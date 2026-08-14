import torch
import torch.nn as nn
import torch.nn.functional as F

class BicubicBaseline(nn.Module):
    """Bicubic interpolation baseline for single image super-resolution."""
    def __init__(self, scale=4):
        super(BicubicBaseline, self).__init__()
        self.scale = scale
        
    def forward(self, x):
        # x is expected to be [B, C, H, W]
        # We use align_corners=False to match standard downsampling behavior
        return F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
