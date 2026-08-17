import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    """Sobel edge preservation loss to enforce sharp boundary structures and gradient magnitude matching."""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel kernels for X and Y directions
        kx = torch.FloatTensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        ky = torch.FloatTensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        
    def forward(self, pred, gt):
        # Assumes input is grayscale [B, 1, H, W]
        p_grad_x = F.conv2d(pred, self.kx, padding=1)
        p_grad_y = F.conv2d(pred, self.ky, padding=1)
        
        g_grad_x = F.conv2d(gt, self.kx, padding=1)
        g_grad_y = F.conv2d(gt, self.ky, padding=1)
        
        # Gradient magnitude matching
        p_mag = torch.sqrt(p_grad_x**2 + p_grad_y**2 + 1e-8)
        g_mag = torch.sqrt(g_grad_x**2 + g_grad_y**2 + 1e-8)
        
        loss_grad = torch.mean(torch.abs(p_grad_x - g_grad_x)) + torch.mean(torch.abs(p_grad_y - g_grad_y))
        loss_mag = torch.mean(torch.abs(p_mag - g_mag))
        
        return loss_grad + loss_mag

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

import math
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) Loss in PyTorch."""
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', create_window(window_size, self.channel))

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)

class CompoundRestorationLoss(nn.Module):
    """Aggregated loss function: L_total = L_1 + lambda_ssim * L_SSIM + lambda_edge * L_edge"""
    def __init__(self, lambda_ssim=1.0, lambda_edge=1.0):
        super(CompoundRestorationLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        
    def forward(self, pred, gt):
        loss_l1 = self.l1(pred, gt)
        loss_ssim = self.ssim(pred, gt)
        loss_edge = self.edge(pred, gt)
        
        total_loss = loss_l1 + self.lambda_ssim * loss_ssim + self.lambda_edge * loss_edge
        
        return total_loss, {
            "l1": loss_l1.item(),
            "ssim": loss_ssim.item(),
            "edge": loss_edge.item(),
            "total": total_loss.item()
        }
