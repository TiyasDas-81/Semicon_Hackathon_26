#!/usr/bin/env python3
"""
Final EDSR2x Demo & Visual Showcase Generator
KLA SemiCon AI Hackathon 2026 — Track 2 Official Submission Demo

Generates:
  1. 10 High-Definition (1920x1080) Presentation Slides under fresh_training/final_submission/demo_assets/
     - demo_01_problem.png
     - demo_02_architecture.png
     - demo_03_comparison_01.png
     - demo_04_comparison_02.png
     - demo_05_comparison_003103.png
     - demo_06_comparison_002728.png
     - demo_07_metrics.png
     - demo_08_efficiency.png
     - demo_09_experiment_comparison.png
     - demo_10_final.png
  2. Official Demo Video: fresh_training/final_submission/EDSR2x_demo.mp4 (75s, 1080p, 30 FPS, H.264)
  3. Official Demo Script: fresh_training/final_submission/demo_script.md
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim_fn

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fresh_training.models.edsr_2x import EDSR2x
from fresh_training.dataset import get_train_val_test_split, KLASemiconDataset

# -----------------------------------------------------------------------------
# METRIC FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))

def calculate_ssim(img1, img2):
    return float(ssim_fn(img1, img2, data_range=1.0))

# -----------------------------------------------------------------------------
# SLIDE RENDERING FUNCTIONS (1920x1080, 150 DPI)
# -----------------------------------------------------------------------------
def setup_dark_figure():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#0B0F19') # Sleek dark background
    return fig

def add_header(fig, title, subtitle):
    fig.text(0.05, 0.93, title, fontsize=24, fontweight='bold', color='#00F0FF', ha='left', va='top', fontfamily='sans-serif')
    fig.text(0.05, 0.87, subtitle, fontsize=14, color='#A0AEC0', ha='left', va='top', fontfamily='sans-serif')

def add_footer(fig):
    fig.text(0.05, 0.04, "KLA SemiCon AI Hackathon 2026 | AIvengers Track 2 Submission", fontsize=11, color='#4A5568', ha='left')
    fig.text(0.95, 0.04, "EDSR2x Production Model", fontsize=11, fontweight='bold', color='#00F0FF', ha='right')

def main():
    print("=" * 80, flush=True)
    print("FINAL EDSR2x DEMO & VISUAL ASSETS GENERATOR", flush=True)
    print("================================================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})", flush=True)

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(exp_dir, "demo_assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Load Winning Checkpoint & Extract Real Outputs
    ckpt_path = os.path.join(PROJECT_ROOT, "fresh_training", "checkpoints", "best_kla_2x.pth")
    model = EDSR2x(num_res_blocks=8, num_channels=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.eval()

    train_files, val_files, test_files, gt_dir, lr_dir = get_train_val_test_split(PROJECT_ROOT, seed=42)
    val_ds = KLASemiconDataset(val_files, gt_dir, lr_dir, cache_in_ram=True)

    # Lookup dict for specific validation samples
    val_samples_dict = {}
    with torch.no_grad():
        for i in range(len(val_ds)):
            lr_t, gt_t, fname = val_ds[i]
            fname_str = fname if isinstance(fname, str) else fname[0]
            lr_np = lr_t.squeeze().numpy()
            gt_np = gt_t.squeeze().numpy()
            lr_in = lr_t.unsqueeze(0).to(device)
            pred_t = model(lr_in)
            pred_np = pred_t.squeeze().cpu().numpy()

            bic_np = cv2.resize(lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)
            bic_np = np.clip(bic_np, 0.0, 1.0)

            val_samples_dict[fname_str] = {
                "lr": lr_np, "gt": gt_np, "bicubic": bic_np, "edsr": pred_np,
                "bic_psnr": calculate_psnr(bic_np, gt_np), "bic_ssim": calculate_ssim(bic_np, gt_np),
                "edsr_psnr": calculate_psnr(pred_np, gt_np), "edsr_ssim": calculate_ssim(pred_np, gt_np)
            }

    print(f"Extracted real EDSR2x model outputs for {len(val_samples_dict)} validation samples.", flush=True)

    # Pick representative samples
    sample_normal = val_samples_dict.get("003001.npy", list(val_samples_dict.values())[0])
    sample_texture = val_samples_dict.get("003010.npy", list(val_samples_dict.values())[10])
    sample_003103 = val_samples_dict.get("003103.npy", list(val_samples_dict.values())[20])
    sample_002728 = val_samples_dict.get("002728.npy", list(val_samples_dict.values())[30])

    # -------------------------------------------------------------------------
    # SLIDE 1: INTRODUCTION & TITLE (demo_01_problem.png / Title)
    # -------------------------------------------------------------------------
    print("Rendering Slide 1: Introduction & Title...", flush=True)
    fig = setup_dark_figure()
    fig.text(0.5, 0.65, "AI-Based Restoration of Degraded Semiconductor Images", fontsize=28, fontweight='bold', color='#FFFFFF', ha='center', va='center')
    fig.text(0.5, 0.55, "EDSR2x — Lightweight Deep Learning Restoration Pipeline", fontsize=18, color='#00F0FF', ha='center', va='center')
    
    # 3-step pipeline graphic
    ax1 = fig.add_axes([0.15, 0.20, 0.20, 0.25])
    ax1.imshow(sample_normal["lr"], cmap='gray')
    ax1.set_title("1. Degraded Input (128x128)", color='#A0AEC0', fontsize=12, pad=10)
    ax1.axis('off')

    fig.text(0.40, 0.325, "──▶   EDSR2x Deep Restoration   ──▶", fontsize=14, fontweight='bold', color='#00F0FF', ha='center', va='center')

    ax2 = fig.add_axes([0.65, 0.20, 0.20, 0.25])
    ax2.imshow(sample_normal["edsr"], cmap='gray', vmin=0, vmax=1)
    ax2.set_title("2. Restored Output (256x256)", color='#00FF66', fontsize=12, pad=10)
    ax2.axis('off')

    add_footer(fig)
    slide1_path = os.path.join(assets_dir, "demo_01_problem.png")
    fig.savefig(slide1_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 2: THE PROBLEM (Degraded SEM Analysis)
    # -------------------------------------------------------------------------
    print("Rendering Slide 2: The Problem (SEM Degradation)...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "The Challenge: Low-Dose SEM Degradation", "Semiconductor inspection images suffer from severe electron shot noise and spatial downsampling")

    ax1 = fig.add_axes([0.10, 0.22, 0.40, 0.60])
    ax1.imshow(sample_normal["lr"], cmap='gray')
    ax1.set_title("Degraded Low-Resolution SEM Image (128x128)", color='#FFFFFF', fontsize=14, pad=12)
    ax1.axis('off')

    # Zoomed patch callout
    rect = Rectangle((40, 40), 32, 32, linewidth=2, edgecolor='#FF0055', facecolor='none')
    ax1.add_patch(rect)

    ax2 = fig.add_axes([0.55, 0.35, 0.35, 0.45])
    crop_lr = sample_normal["lr"][40:72, 40:72]
    ax2.imshow(crop_lr, cmap='gray')
    ax2.set_title("Magnified Micro-Structure Region", color='#FF0055', fontsize=14, pad=12)
    ax2.axis('off')

    # Key degradation text box
    box_text = "Key Degradations Identified:\n\n• High Poisson-Gaussian Shot Noise (σ ≈ 0.0875)\n• 2x Spatial Resolution Loss (128x128 -> 256x256)\n• Micro-texture Overlap in High-Frequency Spectrum\n• Edge Blur & Contrast Reduction"
    fig.text(0.55, 0.24, box_text, fontsize=13, color='#E2E8F0', va='top', bbox=dict(boxstyle="round,pad=0.8", facecolor="#1A202C", edgecolor="#FF0055", alpha=0.9))

    add_footer(fig)
    slide2_path = os.path.join(assets_dir, "demo_02_architecture.png")
    fig.savefig(slide2_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 3: OUR SOLUTION (EDSR2x Architecture Diagram)
    # -------------------------------------------------------------------------
    print("Rendering Slide 3: Our Solution (EDSR2x Architecture)...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "Our Solution: EDSR2x Neural Architecture", "Single-stage deep residual network engineered for efficient 2x super-resolution")

    arch_box = "EDSR2x Model Specifications:\n\n• 8 Residual Blocks (64 Feature Channels)\n• 2x PixelShuffle Upsampling Layer\n• Global Bicubic Skip Connection\n• Parameters: 776,705 (~0.78M params)\n• Training Loss: Pure L1 Reconstruction Loss\n• Inference Latency: ~2.95 ms / image (RTX 3050 GPU)"
    fig.text(0.10, 0.45, arch_box, fontsize=14, color='#FFFFFF', va='center', bbox=dict(boxstyle="round,pad=1.0", facecolor="#171923", edgecolor="#00F0FF", alpha=0.9))

    # Visual pipeline blocks on the right
    pipeline_text = "NoisyLR (128x128)\n       │\n       ▼\n[ Conv 3x3 (64 ch) ]\n       │\n       ▼\n[ 8x ResBlocks + Conv ]\n       │\n       ▼\n[ PixelShuffle 2x ] ──▶ [ Conv Tail ] ──▶ Learned Residual\n                                                │\n                                                + ◄── Bicubic 2x Skip\n                                                │\n                                                ▼\n                                        Restored Output (256x256)"
    fig.text(0.60, 0.45, pipeline_text, fontsize=13, fontfamily='monospace', color='#00FF66', va='center', bbox=dict(boxstyle="round,pad=1.0", facecolor="#0D1117", edgecolor="#00FF66", alpha=0.9))

    add_footer(fig)
    slide3_path = os.path.join(assets_dir, "demo_03_comparison_01.png")
    fig.savefig(slide3_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 4: BEFORE / AFTER COMPARISON — NORMAL & TEXTURE CASES
    # -------------------------------------------------------------------------
    print("Rendering Slide 4: Side-by-Side Comparisons (Wafer Structures)...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "Visual Quality: Side-by-Side Restoration", "Real validation sample comparison: Bicubic 2x vs EDSR2x vs Ground Truth")

    # Top Row: Normal Case
    ax1 = fig.add_axes([0.08, 0.52, 0.26, 0.32])
    ax1.imshow(sample_normal["lr"], cmap='gray')
    ax1.set_title("NoisyLR Input (128x128)", color='#A0AEC0', fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_axes([0.37, 0.52, 0.26, 0.32])
    ax2.imshow(sample_normal["bicubic"], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"Bicubic 2x ({sample_normal['bic_psnr']:.2f}dB / {sample_normal['bic_ssim']:.3f})", color='#CBD5E0', fontsize=11)
    ax2.axis('off')

    ax3 = fig.add_axes([0.66, 0.52, 0.26, 0.32])
    ax3.imshow(sample_normal["edsr"], cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f"EDSR2x Restored ({sample_normal['edsr_psnr']:.2f}dB / {sample_normal['edsr_ssim']:.3f})", color='#00FF66', fontsize=11, fontweight='bold')
    ax3.axis('off')

    # Bottom Row: Texture Case
    ax4 = fig.add_axes([0.08, 0.12, 0.26, 0.32])
    ax4.imshow(sample_texture["lr"], cmap='gray')
    ax4.set_title("NoisyLR Input (128x128)", color='#A0AEC0', fontsize=11)
    ax4.axis('off')

    ax5 = fig.add_axes([0.37, 0.12, 0.26, 0.32])
    ax5.imshow(sample_texture["bicubic"], cmap='gray', vmin=0, vmax=1)
    ax5.set_title(f"Bicubic 2x ({sample_texture['bic_psnr']:.2f}dB / {sample_texture['bic_ssim']:.3f})", color='#CBD5E0', fontsize=11)
    ax5.axis('off')

    ax6 = fig.add_axes([0.66, 0.12, 0.26, 0.32])
    ax6.imshow(sample_texture["edsr"], cmap='gray', vmin=0, vmax=1)
    ax6.set_title(f"EDSR2x Restored ({sample_texture['edsr_psnr']:.2f}dB / {sample_texture['edsr_ssim']:.3f})", color='#00FF66', fontsize=11, fontweight='bold')
    ax6.axis('off')

    add_footer(fig)
    slide4_path = os.path.join(assets_dir, "demo_04_comparison_02.png")
    fig.savefig(slide4_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 5: DIFFICULT CASE 1 — SAMPLE 003103.npy
    # -------------------------------------------------------------------------
    print("Rendering Slide 5: Difficult Case Forensic (003103.npy)...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "Complex Structure Forensic: Sample 003103.npy", "Dense contact hole array with high-frequency spatial detail")

    ax1 = fig.add_axes([0.06, 0.25, 0.20, 0.55])
    ax1.imshow(sample_003103["lr"], cmap='gray')
    ax1.set_title("1. NoisyLR Input", color='#A0AEC0', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_axes([0.29, 0.25, 0.20, 0.55])
    ax2.imshow(sample_003103["bicubic"], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"2. Bicubic 2x\n{sample_003103['bic_psnr']:.2f}dB / {sample_003103['bic_ssim']:.4f}", color='#CBD5E0', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_axes([0.52, 0.25, 0.20, 0.55])
    ax3.imshow(sample_003103["edsr"], cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f"3. EDSR2x Restored\n{sample_003103['edsr_psnr']:.2f}dB / {sample_003103['edsr_ssim']:.4f}", color='#00F0FF', fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_axes([0.75, 0.25, 0.20, 0.55])
    ax4.imshow(sample_003103["gt"], cmap='gray', vmin=0, vmax=1)
    ax4.set_title("4. Ground Truth HR\nTarget Reference", color='#00FF66', fontsize=12, fontweight='bold')
    ax4.axis('off')

    add_footer(fig)
    slide5_path = os.path.join(assets_dir, "demo_05_comparison_003103.png")
    fig.savefig(slide5_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 6: DIFFICULT CASE 2 — SAMPLE 002728.npy
    # -------------------------------------------------------------------------
    print("Rendering Slide 6: Difficult Case Forensic (002728.npy)...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "High-Frequency Grating Forensic: Sample 002728.npy", "Extreme high-frequency grating structure with heavy shot noise corruption")

    ax1 = fig.add_axes([0.06, 0.25, 0.20, 0.55])
    ax1.imshow(sample_002728["lr"], cmap='gray')
    ax1.set_title("1. NoisyLR Input", color='#A0AEC0', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_axes([0.29, 0.25, 0.20, 0.55])
    ax2.imshow(sample_002728["bicubic"], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"2. Bicubic 2x\n{sample_002728['bic_psnr']:.2f}dB / {sample_002728['bic_ssim']:.4f}", color='#CBD5E0', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_axes([0.52, 0.25, 0.20, 0.55])
    ax3.imshow(sample_002728["edsr"], cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f"3. EDSR2x Restored\n{sample_002728['edsr_psnr']:.2f}dB / {sample_002728['edsr_ssim']:.4f}", color='#00F0FF', fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_axes([0.75, 0.25, 0.20, 0.55])
    ax4.imshow(sample_002728["gt"], cmap='gray', vmin=0, vmax=1)
    ax4.set_title("4. Ground Truth HR\nTarget Reference", color='#00FF66', fontsize=12, fontweight='bold')
    ax4.axis('off')

    add_footer(fig)
    slide6_path = os.path.join(assets_dir, "demo_06_comparison_002728.png")
    fig.savefig(slide6_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 7: QUANTITATIVE RESULTS
    # -------------------------------------------------------------------------
    print("Rendering Slide 7: Quantitative Benchmark Results...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "Quantitative Performance Metrics", "Rigorous evaluation across 320 official validation & 320 held-out test samples")

    table_data = [
        ["Evaluation Split", "Method", "PSNR (dB)", "SSIM", "MAE", "Edge Score", "PSNR Win Rate"],
        ["Validation (320 samples)", "Bicubic 2x Baseline", "22.59 dB", "0.5166", "0.0597", "0.4727", "—"],
        ["Validation (320 samples)", "EDSR2x (Production)", "27.42 dB", "0.7357", "0.0349", "0.6639", "320/320 (100%)"],
        ["Validation Net Gain", "EDSR2x vs Bicubic", "+4.83 dB", "+0.2191", "-0.0248", "+0.1912", "100.0%"],
        ["Held-Out Test (320 samples)", "Bicubic 2x Baseline", "22.98 dB", "0.5243", "0.0572", "0.4781", "—"],
        ["Held-Out Test (320 samples)", "EDSR2x (Production)", "27.93 dB", "0.7408", "0.0310", "0.6684", "320/320 (100%)"],
        ["Held-Out Test Net Gain", "EDSR2x vs Bicubic", "+4.95 dB", "+0.2165", "-0.0262", "+0.1903", "100.0%"]
    ]

    table_ax = fig.add_axes([0.08, 0.20, 0.84, 0.60])
    table_ax.axis('off')
    tbl = table_ax.table(cellText=table_data, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)

    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = tbl[(i, j)]
            cell.set_height(0.12)
            if i == 0:
                cell.set_facecolor('#1E293B')
                cell.set_text_props(weight='bold', color='#00F0FF')
            elif i in [2, 5]:
                cell.set_facecolor('#0F291E')
                cell.set_text_props(weight='bold', color='#00FF66')
            elif i in [3, 6]:
                cell.set_facecolor('#1E1B4B')
                cell.set_text_props(weight='bold', color='#A855F7')
            else:
                cell.set_facecolor('#0F172A')
                cell.set_text_props(color='#E2E8F0')

    add_footer(fig)
    slide7_path = os.path.join(assets_dir, "demo_07_metrics.png")
    fig.savefig(slide7_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 8: MODEL EFFICIENCY & SPEED
    # -------------------------------------------------------------------------
    print("Rendering Slide 8: Model Efficiency & Hardware Audit...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "Model Efficiency & Hardware Latency", "Engineered for real-time semiconductor line inspection integration")

    metrics_boxes = [
        ("Parameter Count", "776,705", "~0.78M Parameters", "#00F0FF"),
        ("Inference Latency", "2.95 ms", "per image (RTX 3050)", "#00FF66"),
        ("Batch Throughput", "88.1 FPS", "continuous processing", "#A855F7"),
        ("Peak GPU VRAM", "111.6 MB", "< 3% of 4GB VRAM", "#F59E0B")
    ]

    for idx, (title, val, sub, color) in enumerate(metrics_boxes):
        left_pos = 0.08 + idx * 0.22
        b_text = f"{title}\n\n{val}\n\n{sub}"
        fig.text(left_pos + 0.09, 0.50, b_text, fontsize=15, color='#FFFFFF', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=1.2", facecolor="#1E293B", edgecolor=color, alpha=0.9))

    fig.text(0.5, 0.22, "Designed for efficient, reproducible, high-throughput wafer defect inspection", fontsize=16, fontweight='bold', color='#00FF66', ha='center')

    add_footer(fig)
    slide8_path = os.path.join(assets_dir, "demo_08_efficiency.png")
    fig.savefig(slide8_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 9: CONTROLLED RESEARCH COMPARISON
    # -------------------------------------------------------------------------
    print("Rendering Slide 9: Controlled Research Exploration...", flush=True)
    fig = setup_dark_figure()
    add_header(fig, "Controlled Research: Strategy Comparison", "Systematic evaluation of 6 distinct architectural hypotheses")

    research_data = [
        ["Model Architecture / Strategy", "Params", "PSNR (dB)", "SSIM", "Latency", "Final Decision"],
        ["EDSR2x + L1 Loss (Production)", "0.78M", "27.42 dB", "0.7357", "2.95 ms", "🏆 WINNER (Best Overall)"],
        ["RCAN-Light 2x (Channel Attn)", "1.52M", "27.36 dB", "0.7352", "7.48 ms", "Rejected (2.5x Slower)"],
        ["Two-Stage Wavelet Pipeline", "0.90M", "27.27 dB", "0.7288", "3.62 ms", "Rejected (High-Freq Loss)"],
        ["HAT-Small 2x (Hybrid Attn)", "0.29M", "27.26 dB", "0.7324", "10.77 ms", "Rejected (Complex Tradeoff)"],
        ["SwinIR-Light 2x (Window Attn)", "0.22M", "27.13 dB", "0.7273", "10.77 ms", "Rejected (Noise Amplification)"],
        ["EDSR2x + Multi-loss (Charb+SSIM)", "0.78M", "25.20 dB", "0.7061", "2.95 ms", "Rejected (Gradient Conflict)"]
    ]

    table_ax = fig.add_axes([0.08, 0.20, 0.84, 0.60])
    table_ax.axis('off')
    tbl = table_ax.table(cellText=research_data, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for i in range(len(research_data)):
        for j in range(len(research_data[0])):
            cell = tbl[(i, j)]
            cell.set_height(0.12)
            if i == 0:
                cell.set_facecolor('#1E293B')
                cell.set_text_props(weight='bold', color='#00F0FF')
            elif i == 1:
                cell.set_facecolor('#0F291E')
                cell.set_text_props(weight='bold', color='#00FF66')
            else:
                cell.set_facecolor('#1E1E2E')
                cell.set_text_props(color='#E2E8F0')

    add_footer(fig)
    slide9_path = os.path.join(assets_dir, "demo_09_experiment_comparison.png")
    fig.savefig(slide9_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SLIDE 10: FINAL CONCLUSION & SUMMARY (demo_10_final.png)
    # -------------------------------------------------------------------------
    print("Rendering Slide 10: Final Summary & Conclusion...", flush=True)
    fig = setup_dark_figure()

    fig.text(0.5, 0.78, "From Degraded SEM Imagery  ──▶  High-Resolution Restoration", fontsize=24, fontweight='bold', color='#FFFFFF', ha='center')
    fig.text(0.5, 0.68, "EDSR2x + L1 Reconstruction Model", fontsize=20, fontweight='bold', color='#00F0FF', ha='center')

    summary_box = "Validation PSNR : 27.42 dB  (+4.83 dB vs Bicubic)\nValidation SSIM : 0.7357     (+0.2191 vs Bicubic)\nModel Parameters: 776,705    (~0.78M Params)\nGPU Speed       : 2.95 ms    (88.1 FPS)"
    fig.text(0.5, 0.45, summary_box, fontsize=16, fontfamily='monospace', color='#00FF66', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=1.2", facecolor="#0F291E", edgecolor="#00FF66", alpha=0.95))

    fig.text(0.5, 0.22, "Lightweight.  Accurate.  Reproducible.", fontsize=26, fontweight='bold', color='#FFFFFF', ha='center')

    add_footer(fig)
    slide10_path = os.path.join(assets_dir, "demo_10_final.png")
    fig.savefig(slide10_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    print("\nAll 10 High-Definition Demo Asset Slides rendered successfully!", flush=True)

    # -------------------------------------------------------------------------
    # VIDEO ASSEMBLY: EDSR2x_demo.mp4 (75 seconds, 1080p, 30 FPS)
    # -------------------------------------------------------------------------
    video_path = os.path.join(PROJECT_ROOT, "fresh_training", "final_submission", "EDSR2x_demo.mp4")
    print(f"\nAssembling Demo Video: {video_path}...", flush=True)

    slide_files = [
        (slide1_path, 7),  # Section 1: Intro (7s)
        (slide2_path, 9),  # Section 2: Problem (9s)
        (slide3_path, 9),  # Section 3: Solution (9s)
        (slide4_path, 15), # Section 4: Before/After Normal & Texture (15s)
        (slide5_path, 7),  # Section 4: Difficult Case 003103 (7s)
        (slide6_path, 7),  # Section 4: Difficult Case 002728 (7s)
        (slide7_path, 8),  # Section 5: Quantitative Results (8s)
        (slide8_path, 7),  # Section 6: Efficiency (7s)
        (slide9_path, 8),  # Section 7: Controlled Research (8s)
        (slide10_path, 6)  # Section 8: Final Summary (6s)
    ]
    # Total Duration: 7 + 9 + 9 + 15 + 7 + 7 + 8 + 7 + 8 + 6 = 83 seconds (Well within 60-90s window!)

    fps = 30
    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not out_video.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")

    total_frames_written = 0

    for s_idx, (s_path, duration_sec) in enumerate(slide_files):
        img_bgr = cv2.imread(s_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read slide image at {s_path}")
        img_bgr = cv2.resize(img_bgr, (width, height))

        num_frames = duration_sec * fps
        fade_frames = 15 # 0.5s fade transition between slides

        if s_idx > 0 and len(slide_files) > 1:
            prev_bgr = cv2.imread(slide_files[s_idx-1][0])
            prev_bgr = cv2.resize(prev_bgr, (width, height))
            for f_i in range(fade_frames):
                alpha = f_i / float(fade_frames)
                blended = cv2.addWeighted(prev_bgr, 1.0 - alpha, img_bgr, alpha, 0)
                out_video.write(blended)
                total_frames_written += 1
            remaining_frames = num_frames - fade_frames
        else:
            remaining_frames = num_frames

        for _ in range(remaining_frames):
            out_video.write(img_bgr)
            total_frames_written += 1

    out_video.release()
    total_sec = total_frames_written / fps
    print(f"  • Video Output Path : {video_path}")
    print(f"  • Video Resolution  : {width}x{height}")
    print(f"  • Total Frames      : {total_frames_written}")
    print(f"  • Total Duration    : {total_sec:.2f} seconds")

    # -------------------------------------------------------------------------
    # DEMO SCRIPT GENERATION: fresh_training/final_submission/demo_script.md
    # -------------------------------------------------------------------------
    script_path = os.path.join(PROJECT_ROOT, "fresh_training", "final_submission", "demo_script.md")
    print(f"\nGenerating Official Demo Script: {script_path}...", flush=True)

    script_md = """# Official EDSR2x Hackathon Demo Script & Storyboard
## KLA SemiCon AI Hackathon 2026 — Track 2 Presentation Package

### Video Overview
- **File**: [`fresh_training/final_submission/EDSR2x_demo.mp4`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/EDSR2x_demo.mp4)
- **Duration**: `83 seconds` (`01:23`)
- **Resolution**: `1920x1080` (1080p, 30 FPS)
- **Target Audience**: KLA Hackathon Evaluators, AI Engineers, Semiconductor Domain Experts

---

### Timed Storyboard & Narration Breakdown

| Timestamp | Section | Visual Asset Used | Screen Text & Visual Content | Narration / Voiceover Script |
| :--- | :--- | :--- | :--- | :--- |
| **00:00 – 00:07** | **Section 1: Title** | `demo_01_problem.png` | "AI-Based Restoration of Degraded Semiconductor Images" | *"Semiconductor inspection images can suffer from severe electron shot noise and limited spatial resolution. Our solution uses EDSR2x to restore degraded 128 by 128 images into 256 by 256 high-resolution outputs."* |
| **00:07 – 00:16** | **Section 2: Problem** | `demo_02_architecture.png` | "The Challenge: Low-Dose SEM Degradation" (Micro-structure zoom callout) | *"Low-dose Scanning Electron Microscopy exhibits heavy Poisson-Gaussian shot noise and high-frequency spatial attenuation, severely degrading feature visibility."* |
| **00:16 – 00:25** | **Section 3: Solution** | `demo_03_comparison_01.png` | "Our Solution: EDSR2x Neural Architecture" (~0.78M params) | *"We engineered EDSR2x—a lightweight single-stage residual network with 8 residual blocks and global bicubic residual skip connections, trained on pure L1 reconstruction loss."* |
| **00:25 – 00:40** | **Section 4: Visuals** | `demo_04_comparison_02.png` | Side-by-side wafer structure restoration (Bicubic vs EDSR2x) | *"Across standard wafer structures, EDSR2x cleanly removes Poisson-Gaussian noise while preserving edge boundaries, achieving a 4.83 dB PSNR gain over bicubic interpolation."* |
| **00:40 – 00:47** | **Section 4: Forensic 1** | `demo_05_comparison_003103.png` | Complex Contact Hole Array Forensic (`003103.npy`) | *"On dense contact hole arrays, EDSR2x maintains structural alignment without introducing artificial artifacts or over-smoothing."* |
| **00:47 – 00:54** | **Section 4: Forensic 2** | `demo_06_comparison_002728.png` | Extreme High-Freq Grating Forensic (`002728.npy`) | *"Even under extreme high-frequency wafer grating details, the model accurately recovers line edge profiles."* |
| **00:54 – 01:02** | **Section 5: Metrics** | `demo_07_metrics.png` | Quantitative Benchmark Table (Validation: 27.42 dB / Test: 27.93 dB) | *"On our 320 validation samples, EDSR2x achieved 27.42 dB PSNR and 0.7357 SSIM. On held-out internal test samples, performance scaled consistently to 27.93 dB PSNR."* |
| **01:02 – 01:09** | **Section 6: Efficiency** | `demo_08_efficiency.png` | Hardware Audit (2.95 ms/image, 88.1 FPS, 111.6 MB VRAM) | *"Containing just 0.78 million parameters, EDSR2x processes images in 2.95 milliseconds on an RTX 3050 GPU, using under 112 megabytes of VRAM."* |
| **01:09 – 01:17** | **Section 7: Research** | `demo_09_experiment_comparison.png` | Controlled Strategy Matrix (EDSR2x vs RCAN vs SwinIR vs HAT) | *"Our systematic evaluation proved that EDSR2x outperforms complex window-attention and multi-stage models on this dataset."* |
| **01:17 – 01:23** | **Section 8: Summary** | `demo_10_final.png` | "Lightweight. Accurate. Reproducible. — EDSR2x + L1" | *"Our experiments confirm that EDSR2x provides the strongest overall balance of accuracy, efficiency, and reproducibility for semiconductor image restoration."* |
"""

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_md)

    # -------------------------------------------------------------------------
    # UPDATE README.md IN fresh_training/final_submission/
    # -------------------------------------------------------------------------
    readme_path = os.path.join(PROJECT_ROOT, "fresh_training", "final_submission", "README.md")
    print(f"\nUpdating Submission README: {readme_path}...", flush=True)

    readme_content = """# KLA SemiCon AI Hackathon 2026 — Track 2 Official Submission Package
## AIvengers Team Submission

### Model Architecture: EDSR2x + L1 Loss
- **Scale Factor**: 2x Super-Resolution (128x128 NoisyLR -> 256x256 Restored HR)
- **Parameters**: `776,705` (`0.78M` params)
- **Blocks**: 8 Residual Blocks, 64 Feature Channels, 2x PixelShuffle
- **Global Residual**: `bicubic_2x(input) + learned_residual`

### Performance Metrics (Official Validation Split - 320 Samples)
- **PSNR**: `27.42 dB` (vs Bicubic `22.59 dB`)
- **SSIM**: `0.7357` (vs Bicubic `0.5166`)
- **MAE**: `0.0349`
- **Edge Preservation Score**: `0.6639`
- **PSNR Win Rate vs Bicubic**: `320/320` (`100.0%`)
- **SSIM Win Rate vs Bicubic**: `284/320` (`88.8%`)

### Held-Out Internal Test Set (320 Samples)
- **PSNR**: `27.93 dB` (vs Bicubic `22.98 dB`)
- **SSIM**: `0.7408` (vs Bicubic `0.5243`)
- **MAE**: `0.0310`

### Execution & Latency
- **GPU Latency**: `2.95 ms / image` on NVIDIA RTX 3050 GPU (`88.1 FPS` batch throughput)
- **Peak VRAM**: `111.6 MB`

### Official Presentation Demo Video
- **Demo Video**: [`fresh_training/final_submission/EDSR2x_demo.mp4`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/EDSR2x_demo.mp4) (`83s`, 1080p, 30 FPS)
- **Demo Assets**: [`fresh_training/final_submission/demo_assets/`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/demo_assets)
- **Demo Script**: [`fresh_training/final_submission/demo_script.md`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/demo_script.md)

### Usage Command
```bash
python fresh_training/final_submission/inference.py --input <path_to_input_dir> --output <path_to_output_dir>
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\nAll demo assets, video, script, and README update completed successfully!", flush=True)

if __name__ == "__main__":
    main()
