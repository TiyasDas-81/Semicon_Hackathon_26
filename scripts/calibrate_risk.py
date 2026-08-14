import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import numpy as np
import cv2
import torch

from datasets.adapter import CarinthiaAdapter
from inference.restorer import SemiconImageRestorer

def main():
    print("Initializing calibration process...")
    # Load 20 validation observations from Carinthia adapter (degradation level 2)
    adapter = CarinthiaAdapter(root_dir="data/raw/carinthia", degradation_level=2, num_samples=20)
    
    checkpoint_dir = "checkpoints"
    transformer_path = os.path.join(checkpoint_dir, "best_transformer.pth")
    
    if not os.path.exists(transformer_path):
        print(f"ERROR: Transformer checkpoint not found at {transformer_path}. Please train the model first.")
        sys.exit(1)
        
    print("Loading SwinIR-Light Restorer for calibration...")
    restorer = SemiconImageRestorer(model_type="transformer", checkpoint_path=transformer_path)
    
    modifications = []
    consistencies = []
    
    print(f"Running inference over {len(adapter)} validation observations...")
    for idx in range(len(adapter)):
        obs = adapter[idx]
        lr_np = obs.degraded_image
        hr_np = obs.ground_truth
        
        # Run restoration
        restored, _, _ = restorer.restore_image(lr_np, patch_size=None)
        
        # 1. AI Modification (HR resolution): |F(Y) - Bicubic(Y)|
        hr_h, hr_w = restored.shape
        bicubic_hr = cv2.resize(lr_np, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        ai_mod = np.abs(restored - bicubic_hr)
        modifications.append(ai_mod.mean())
        
        # 2. Consistency Error (LR resolution): |Y - D(F(Y))|
        lr_h, lr_w = lr_np.shape
        re_degraded = cv2.resize(restored, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        const_err = np.abs(lr_np - re_degraded)
        consistencies.append(const_err.mean())
        
    # Calculate calibration statistics
    mu_mod = float(np.mean(modifications))
    mu_const = float(np.mean(consistencies))
    
    calibration_data = {
        "mu_modification": mu_mod,
        "mu_consistency": mu_const,
        "calibration_date": "2026-08-14",
        "validation_samples_count": len(adapter),
        "source_dataset": "Carinthia SEM Defect Dataset",
        "description": "Validation-set calibration parameters used to normalize risk mapping telemetry."
    }
    
    # Save to experiments/
    os.makedirs("experiments", exist_ok=True)
    calib_exp_path = "experiments/risk_calibration.json"
    with open(calib_exp_path, "w") as f:
        json.dump(calibration_data, f, indent=4)
    print(f"Saved calibration parameters to {calib_exp_path}")
    
    # Copy to configs/
    os.makedirs("configs", exist_ok=True)
    calib_conf_path = "configs/risk_calibration.json"
    with open(calib_conf_path, "w") as f:
        json.dump(calibration_data, f, indent=4)
    print(f"Copied calibration parameters to {calib_conf_path}")
    
    print("\nCalibration successfully completed!")
    print(f"  AI Modification Mean (mu_mod): {mu_mod:.5f}")
    print(f"  Consistency Error Mean (mu_const): {mu_const:.5f}")

if __name__ == "__main__":
    main()
