import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block without Batch Normalization (EDSR-style)."""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class EDSRLight(nn.Module):
    """Lightweight EDSR with Global Residual Learning for semiconductor image restoration."""
    def __init__(self, scale=4, num_res_blocks=8, num_channels=48, in_channels=1, out_channels=1, global_residual=True):
        super(EDSRLight, self).__init__()
        self.scale = scale
        self.global_residual = global_residual
        
        # 1. Feature extraction
        self.head = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=True)
        
        # 2. Residual blocks sequence
        self.body = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # 3. Intermediate convolution before upsampling
        self.conv_after_body = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True)
        
        # 4. Upsampling Module (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, out_channels * (scale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        # Global residual: bicubic-upscale input as the low-frequency base
        if self.global_residual:
            bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # Initial feature extraction
        feat = self.head(x)
        
        # Body
        body_out = self.body(feat)
        body_out = self.conv_after_body(body_out)
        
        # Long skip connection
        feat = feat + body_out
        
        # Upsampling and reconstruction
        out = self.upsample(feat)
        
        # Add bicubic base: model only learns the high-frequency residual
        if self.global_residual:
            out = out + bicubic
        
        # Clamp output to valid [0,1] range
        out = torch.clamp(out, 0.0, 1.0)
        return out
