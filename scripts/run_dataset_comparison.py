import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import json
import numpy as np
import cv2
import torch

from datasets.adapter import CarinthiaAdapter, MIICAdapter, NISTAdapter
from inference.restorer import SemiconImageRestorer
from scripts.run_degradation_study import calculate_psnr, calculate_ssim, calculate_edge_preservation

def calculate_local_contrast(img):
    """Calculates RMS local contrast."""
    return float(np.std(img))

def calculate_high_frequency_score(img):
    """Calculates relative high frequency energy using FFT2 magnitude."""
    h, w = img.shape
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    mask = (x*x + y*y) > (min(h, w)/4)**2
    return float(np.mean(magnitude[mask]))

def main():
    print("Initializing comparative benchmark study...")
    checkpoint_dir = "checkpoints"
    cnn_path = os.path.join(checkpoint_dir, "best_cnn.pth")
    transformer_path = os.path.join(checkpoint_dir, "best_transformer.pth")
    
    # Check model checkpoints
    if not (os.path.exists(cnn_path) and os.path.exists(transformer_path)):
        print("ERROR: Models must be trained before running benchmarks.")
        sys.exit(1)
        
    # Load restorers
    restorers = {
        "Bicubic": SemiconImageRestorer(model_type="bicubic"),
        "CNN": SemiconImageRestorer(model_type="cnn", checkpoint_path=cnn_path),
        "Transformer": SemiconImageRestorer(model_type="transformer", checkpoint_path=transformer_path)
    }
    
    # Define test configurations
    # We will test 5 samples per condition for fast, thorough benchmarking
    num_samples = 5
    
    adapters = {
        "Carinthia (Paired, Synthetic)": CarinthiaAdapter(degradation_level=2, num_samples=num_samples),
        "MIIC (Paired, Synthetic)": MIICAdapter(degradation_level=2, num_samples=num_samples),
        "NIST (Paired, Ground Truth)": NISTAdapter(set_num=1, paired=True, num_samples=num_samples),
        "NIST (Blind, Unpaired)": NISTAdapter(set_num=1, paired=False, num_samples=num_samples)
    }
    
    csv_fields = [
        "dataset", "training_type", "model", "degradation", "scale", 
        "PSNR", "SSIM", "edge_score", "consistency_error", "inference_time", "notes"
    ]
    
    csv_rows = []
    
    print("\nStarting evaluation loop...")
    for dataset_name, adapter in adapters.items():
        print(f"\nEvaluating on: {dataset_name}")
        is_blind = "Blind" in dataset_name
        
        for model_name, restorer in restorers.items():
            psnrs, ssims, edges, consistencies, times = [], [], [], [], []
            
            for idx in range(len(adapter)):
                obs = adapter[idx]
                lr_np = obs.degraded_image
                hr_np = obs.ground_truth
                
                # Perform restoration and measure time
                start_time = time.time()
                restored, confidence, deviation, risk = restorer.restore_image(lr_np, patch_size=None)
                elapsed = (time.time() - start_time) * 1000.0  # ms
                
                # Consistency error is 1 - confidence mean
                const_err = float(1.0 - confidence.mean())
                
                consistencies.append(const_err)
                times.append(elapsed)
                
                # Metric calculation
                if not is_blind and hr_np is not None:
                    # Paired metrics (both brought to same resolution)
                    # For NIST, LR and HR are both 512x512, but model restores at 4x (2048x2048).
                    # To calculate metrics, we downsample restored to match ground truth resolution (512x512).
                    gt_h, gt_w = hr_np.shape
                    if restored.shape != hr_np.shape:
                        restored_eval = cv2.resize(restored, (gt_w, gt_h), interpolation=cv2.INTER_AREA)
                    else:
                        restored_eval = restored
                        
                    psnr = calculate_psnr(restored_eval, hr_np)
                    ssim = calculate_ssim(restored_eval, hr_np)
                    edge = calculate_edge_preservation(restored_eval, hr_np)
                    
                    psnrs.append(psnr)
                    ssims.append(ssim)
                    edges.append(edge)
                else:
                    # Blind metrics (no ground truth used!)
                    # Compute edge score relative to input
                    # Downscale restored to LR to compare edges directly at input scale
                    lr_h, lr_w = lr_np.shape
                    restored_eval = cv2.resize(restored, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
                    edge = calculate_edge_preservation(restored_eval, lr_np)
                    edges.append(edge)
            
            # Aggregate metrics
            mean_psnr = f"{np.mean(psnrs):.2f}" if (psnrs and not is_blind) else "N/A - Blind"
            mean_ssim = f"{np.mean(ssims):.4f}" if (ssims and not is_blind) else "N/A - Blind"
            mean_edge = f"{np.mean(edges):.4f}"
            mean_const = f"{np.mean(consistencies):.4f}"
            mean_time = f"{np.mean(times):.1f}"
            
            notes = "Synthetic 4x scaling" if "Synthetic" in dataset_name else "ARTIMAGEN simulated noise/contrast"
            if is_blind:
                notes += " (Blind Metrology Mode)"
                
            row = {
                "dataset": dataset_name,
                "training_type": "Supervised" if model_name != "Bicubic" else "None",
                "model": model_name,
                "degradation": "Controlled" if "Synthetic" in dataset_name else "NIST Noise Grid",
                "scale": "4x",
                "PSNR": mean_psnr,
                "SSIM": mean_ssim,
                "edge_score": mean_edge,
                "consistency_error": mean_const,
                "inference_time": mean_time,
                "notes": notes
            }
            csv_rows.append(row)
            
            print(f"  Model: {model_name:12s} | PSNR: {mean_psnr:12s} | SSIM: {mean_ssim:12s} | Edge: {mean_edge} | ConstErr: {mean_const} | Time: {mean_time}ms")
            
    # Save to experiments/dataset_comparison.csv
    csv_path = "experiments/dataset_comparison.csv"
    os.makedirs("experiments", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"\nSuccessfully wrote benchmarking results to {csv_path}")

if __name__ == "__main__":
    main()
