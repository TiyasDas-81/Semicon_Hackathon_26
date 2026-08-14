import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import glob
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from inference.restorer import SemiconImageRestorer
from evaluation.evaluator import calculate_psnr, calculate_ssim, calculate_edge_preservation
from scripts.generate_degradation_levels import generate_study_set

def main():
    # 1. Generate study set if not present
    study_base = "data/degraded_study"
    if not os.path.exists(os.path.join(study_base, "level1", "lr")):
        print("Study set not found. Synthesizing from Carinthia dataset...")
        # Check for Carinthia raw files
        carinthia_files = glob.glob("data/raw/carinthia/**/*.jpg", recursive=True) + \
                          glob.glob("data/raw/carinthia/**/*.png", recursive=True)
                          
        if not carinthia_files:
            print("ERROR: No Carinthia files found in data/raw/carinthia. Run scripts/download_datasets.py first.")
            sys.exit(1)
            
        success = generate_study_set("data/raw/carinthia/**/*", study_base, num_samples=30)
        if not success:
            print("Failed to generate degradation levels.")
            sys.exit(1)
            
    # 2. Setup restorers
    checkpoint_dir = "checkpoints"
    print("Loading models for degradation study...")
    
    models = {
        "Bicubic": SemiconImageRestorer(model_type="bicubic"),
        "CNN": SemiconImageRestorer(model_type="cnn", checkpoint_path=os.path.join(checkpoint_dir, "best_cnn.pth")),
        "Transformer": SemiconImageRestorer(model_type="transformer", checkpoint_path=os.path.join(checkpoint_dir, "best_transformer.pth"))
    }
    
    levels = [1, 2, 3, 4]
    results_list = []
    
    # 3. Evaluate each model at each degradation level
    for model_name, restorer in models.items():
        print(f"Evaluating Model: {model_name}...")
        for lvl in levels:
            lvl_lr_dir = os.path.join(study_base, f"level{lvl}", "lr")
            lvl_hr_dir = os.path.join(study_base, f"level{lvl}", "hr")
            
            lr_files = sorted(glob.glob(os.path.join(lvl_lr_dir, "*.png")))
            hr_files = sorted(glob.glob(os.path.join(lvl_hr_dir, "*.png")))
            
            psnrs, ssims, edges, confidences = [], [], [], []
            
            for lr_f, hr_f in zip(lr_files, hr_files):
                lr_np = cv2.imread(lr_f, cv2.IMREAD_GRAYSCALE) / 255.0
                hr_np = cv2.imread(hr_f, cv2.IMREAD_GRAYSCALE) / 255.0
                
                restored, confidence_map, deviation_map = restorer.restore_image(lr_np, patch_size=None)
                
                # Metrics
                psnr = calculate_psnr(restored, hr_np)
                ssim = calculate_ssim(restored, hr_np)
                edge = calculate_edge_preservation(restored, hr_np)
                
                psnrs.append(psnr)
                ssims.append(ssim)
                edges.append(edge)
                confidences.append(confidence_map.mean())
                
            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            avg_edge = np.mean(edges)
            avg_conf = np.mean(confidences)
            
            print(f"  Level {lvl} -> PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, Edge: {avg_edge:.4f}, Conf: {avg_conf:.4f}")
            
            results_list.append({
                "Model": model_name,
                "Level": lvl,
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "EdgeScore": avg_edge,
                "MeanConfidence": avg_conf
            })
            
    # 4. Save results and plot curves
    df = pd.DataFrame(results_list)
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/degradation_study.csv", index=False)
    print("Saved study metrics to experiments/degradation_study.csv")
    
    # Generate matplotlib plot
    plt.figure(figsize=(12, 5))
    
    # PSNR Plot
    plt.subplot(1, 2, 1)
    for model_name in models.keys():
        sub = df[df["Model"] == model_name]
        plt.plot(sub["Level"], sub["PSNR"], marker='o', label=model_name)
    plt.title("Restoration PSNR vs. Degradation Level")
    plt.xlabel("Degradation Level (1: Mild, 4: Extreme)")
    plt.ylabel("PSNR (dB)")
    plt.xticks(levels)
    plt.grid(True)
    plt.legend()
    
    # SSIM Plot
    plt.subplot(1, 2, 2)
    for model_name in models.keys():
        sub = df[df["Model"] == model_name]
        plt.plot(sub["Level"], sub["SSIM"], marker='s', label=model_name)
    plt.title("Restoration SSIM vs. Degradation Level")
    plt.xlabel("Degradation Level (1: Mild, 4: Extreme)")
    plt.ylabel("SSIM")
    plt.xticks(levels)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("experiments/degradation_study.png", dpi=150)
    plt.close()
    print("Saved comparison chart to experiments/degradation_study.png")
    
    # 5. Determine warning thresholds empirically based on Transformer performance
    trans_results = df[df["Model"] == "Transformer"]
    
    # We find the level where performance significantly drops.
    # E.g. find level where PSNR drops below 19.0 dB or SSIM drops below 0.35, or Confidence drops below 0.88
    # Let's write the warning bounds
    warning_conf_threshold = 0.88
    warning_psnr_threshold = 19.0
    
    for _, row in trans_results.iterrows():
        lvl = int(row["Level"])
        # If severe (Level 3) or extreme (Level 4) performance drops, we extract their values
        if lvl == 3:
            # Set the warning threshold to Level 3's mean confidence minus a small safety buffer (e.g. 0.02)
            warning_conf_threshold = float(row["MeanConfidence"]) - 0.02
            warning_psnr_threshold = float(row["PSNR"]) - 0.5
            break
            
    thresholds = {
        "warning_confidence": warning_conf_threshold,
        "warning_psnr": warning_psnr_threshold,
        "source": "Threshold selected empirically from validation experiments (Level 3 Severe Threshold)."
    }
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/warning_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=4)
        
    print(f"\nEmpirical warning thresholds established:")
    print(f"  Confidence Threshold: {warning_conf_threshold:.4f}")
    print(f"  PSNR Threshold: {warning_psnr_threshold:.2f} dB")
    print("Written thresholds to configs/warning_thresholds.json")

if __name__ == "__main__":
    main()
