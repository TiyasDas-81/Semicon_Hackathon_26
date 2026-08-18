#!/usr/bin/env python3
"""
Official KLA SemiCon AI Hackathon 2026 — Track 2 Production Inference Script
Model: EDSR2x Super-Resolution Network (0.78M parameters)
Usage:
    python fresh_training/final_submission/inference.py --input <input_directory_or_file> --output <output_directory>
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_channels=64, res_scale=1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSR2x(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_res_blocks=8, num_channels=64, res_scale=1.0):
        super(EDSR2x, self).__init__()
        self.head = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=True)
        body_blocks = [ResBlock(num_channels, res_scale) for _ in range(num_res_blocks)]
        body_blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*body_blocks)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        bicubic_skip = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        head_feats = self.head(x)
        body_feats = self.body(head_feats)
        deep_feats = head_feats + body_feats
        up_feats = self.upsample(deep_feats)
        learned_res = self.tail(up_feats)
        return torch.clamp(bicubic_skip + learned_res, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="KLA Image Restoration Production Inference Script")
    parser.add_argument("--input", required=True, help="Path to input .npy file or directory containing .npy files")
    parser.add_argument("--output", required=True, help="Path to output directory to save restored .npy files")
    parser.add_argument("--weights", default=None, help="Path to EDSR2x model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Running on device: {device}")

    # Determine weights path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.weights and os.path.exists(args.weights):
        weights_path = args.weights
    elif os.path.exists(os.path.join(script_dir, "model", "best_kla_2x.pth")):
        weights_path = os.path.join(script_dir, "model", "best_kla_2x.pth")
    elif os.path.exists(os.path.join(script_dir, "..", "checkpoints", "best_kla_2x.pth")):
        weights_path = os.path.join(script_dir, "..", "checkpoints", "best_kla_2x.pth")
    else:
        raise FileNotFoundError("Could not locate EDSR2x checkpoint file!")

    model = EDSR2x(num_res_blocks=8, num_channels=64).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Inference] Loaded EDSR2x model weights from {weights_path}")

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = sorted(glob.glob(os.path.join(args.input, "*.npy")))

    print(f"[Inference] Found {len(files)} .npy files to process.")
    t0 = time.time()

    with torch.no_grad():
        for filepath in files:
            fname = os.path.basename(filepath)
            img_np = np.load(filepath).astype(np.float32)

            # Ensure (B, C, H, W) tensor shape
            if img_np.ndim == 2:
                tensor_in = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            elif img_np.ndim == 3:
                tensor_in = torch.from_numpy(img_np).unsqueeze(0).to(device)
            else:
                tensor_in = torch.from_numpy(img_np).to(device)

            pred_tensor = model(tensor_in)
            pred_np = pred_tensor.squeeze().cpu().numpy().astype(np.float32)

            # Bounds & NaN safety check
            np.nan_to_num(pred_np, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            np.clip(pred_np, 0.0, 1.0, out=pred_np)

            out_path = os.path.join(args.output, fname)
            np.save(out_path, pred_np)

    t1 = time.time()
    total_sec = t1 - t0
    avg_ms = (total_sec / len(files)) * 1000.0 if files else 0.0
    print(f"[Inference] Completed processing {len(files)} files in {total_sec:.2f}s (Avg: {avg_ms:.2f} ms/image).")

if __name__ == "__main__":
    main()
