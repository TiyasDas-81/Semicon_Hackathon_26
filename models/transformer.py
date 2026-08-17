import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): Window size
    Returns:
        windows: (num_windows * B, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows * B, window_size * window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA) module."""
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (each shape: B_, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class DWFFN(nn.Module):
    """Depth-wise Convolutional Feed-Forward Network to share info across windows."""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super(DWFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        # Depth-wise convolution mixes spatial info across windows
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        
    def forward(self, x):
        # x is (B, C, H, W)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class SwinBlockLight(nn.Module):
    """A single Transformer block utilizing W-MSA and a DW-FFN for local modeling."""
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=2.0):
        super(SwinBlockLight, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim) # equivalent to LayerNorm over spatial dimensions
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.ffn = DWFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Branch 1: Window Attention
        shortcut = x
        x_norm = self.norm1(x)
        
        # Partition into windows
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, pad_w, 0, pad_h))
            
        Hp, Wp = H + pad_h, W + pad_w
        x_windows = window_partition(x_norm, self.window_size) # (num_windows*B, window_size*window_size, C)
        
        # Attention
        attn_windows = self.attn(x_windows)
        
        # Reconstruct windows
        x_attn = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            x_attn = x_attn[:, :, :H, :W]
            
        x = shortcut + x_attn
        
        # Branch 2: DW-FFN
        x = x + self.ffn(self.norm2(x))
        return x

class SwinIRLight(nn.Module):
    """Lightweight SwinIR-inspired image restoration network for resource-constrained training."""
    def __init__(self, scale=4, embed_dim=48, depths=[4, 4], num_heads=[4, 4], window_size=8, mlp_ratio=2.0, in_channels=1, out_channels=1, global_residual=False):
        super(SwinIRLight, self).__init__()
        self.scale = scale
        self.global_residual = global_residual
        
        # 1. Shallow Feature Extraction
        self.shallow_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # 2. Deep Feature Extraction (multiple blocks)
        self.blocks1 = nn.Sequential(*[
            SwinBlockLight(dim=embed_dim, num_heads=num_heads[0], window_size=window_size, mlp_ratio=mlp_ratio)
            for _ in range(depths[0])
        ])
        
        self.blocks2 = nn.Sequential(*[
            SwinBlockLight(dim=embed_dim, num_heads=num_heads[1], window_size=window_size, mlp_ratio=mlp_ratio)
            for _ in range(depths[1])
        ])
        
        # Conv layer before upsampling to incorporate residual structure
        self.mid_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # 3. Upsampling and Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        # Global residual: bicubic-upscale input as the low-frequency base
        if self.global_residual:
            bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # Shallow features
        feat = self.shallow_conv(x)
        
        # Deep blocks
        x_deep = self.blocks1(feat)
        x_deep = self.blocks2(x_deep)
        x_deep = self.mid_conv(x_deep)
        
        # Add skip connection from shallow feature
        feat = feat + x_deep
        
        # Reconstruct output
        out = self.upsample(feat)

        # Add bicubic base: model only learns the high-frequency residual
        if self.global_residual:
            out = out + bicubic

        # Clamp output to valid [0,1] range — use clamp not sigmoid to preserve full contrast
        out = torch.clamp(out, 0.0, 1.0)
        return out
