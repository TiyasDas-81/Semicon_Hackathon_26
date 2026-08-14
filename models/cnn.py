import torch
import torch.nn as nn

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
    """A lightweight version of EDSR customized for rapid training and 4GB GPU constraints."""
    def __init__(self, scale=4, num_res_blocks=6, num_channels=32, in_channels=1, out_channels=1):
        super(EDSRLight, self).__init__()
        self.scale = scale
        
        # 1. Feature extraction
        self.head = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=True)
        
        # 2. Residual blocks sequence
        self.body = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # 3. Intermediate convolution before upsampling
        self.conv_after_body = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True)
        
        # 4. Upsampling Module
        # For scale=4, we use PixelShuffle. We need num_channels -> out_channels * scale^2 channels.
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, out_channels * (scale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        # Initial feature extraction
        feat = self.head(x)
        
        # Body
        body_out = self.body(feat)
        body_out = self.conv_after_body(body_out)
        
        # Long skip connection
        feat = feat + body_out
        
        # Upsampling and reconstruction
        out = self.upsample(feat)
        return out
