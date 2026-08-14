import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import yaml

from models.baseline import BicubicBaseline
from models.cnn import EDSRLight
from models.transformer import SwinIRLight

class SemiconImageRestorer:
    """Handles full-size image inference using patch-based restoration and confidence mapping."""
    def __init__(self, model_type="transformer", checkpoint_path=None, config_path="configs/default.yaml", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.scale = self.cfg.get("scale", 4)
        self.model_type = model_type
        
        # Load calibration parameters if available
        import json
        self.calib_path = "configs/risk_calibration.json"
        self.mu_modification = 0.07543
        self.mu_consistency = 0.07739
        if os.path.exists(self.calib_path):
            try:
                with open(self.calib_path, "r") as f:
                    calib_data = json.load(f)
                    self.mu_modification = calib_data.get("mu_modification", 0.07543)
                    self.mu_consistency = calib_data.get("mu_consistency", 0.07739)
            except Exception:
                pass
        
        # Load model
        if model_type == "bicubic":
            self.model = BicubicBaseline(scale=self.scale)
        elif model_type == "cnn":
            cnn_cfg = self.cfg.get("model", {}).get("cnn", {})
            self.model = EDSRLight(
                scale=self.scale,
                num_res_blocks=cnn_cfg.get("num_res_blocks", 6),
                num_channels=cnn_cfg.get("num_channels", 32)
            )
        elif model_type == "transformer":
            trans_cfg = self.cfg.get("model", {}).get("transformer", {})
            self.model = SwinIRLight(
                scale=self.scale,
                embed_dim=trans_cfg.get("embed_dim", 48),
                depths=trans_cfg.get("depths", [4, 4]),
                num_heads=trans_cfg.get("num_heads", [4, 4]),
                window_size=trans_cfg.get("window_size", 8),
                mlp_ratio=trans_cfg.get("mlp_ratio", 2.0)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if model_type != "bicubic" and checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"WARNING: Checkpoint path {checkpoint_path} not found! Model will use initialized weights.")
                
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def restore_image(self, lr_np, patch_size=64, overlap=16):
        """Restores a low-resolution grayscale numpy image [0, 1] using overlapping patch inference."""
        h, w = lr_np.shape
        scale = self.scale
        
        # If image is small enough or patch_size is None, restore in a single forward pass
        if patch_size is None or (h <= patch_size and w <= patch_size):
            lr_t = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
            # Ensure model warm-up/inference
            pred_t = self.model(lr_t)
            pred_np = pred_t.squeeze().cpu().numpy()
            pred_np = np.clip(pred_np, 0.0, 1.0)
            
            # Compute confidence maps
            confidence, deviation, risk = self.compute_confidence_maps(lr_np, pred_np)
            return pred_np, confidence, deviation, risk
            
        # Overlapping patch-based tiling & stitching
        out_h, out_w = h * scale, w * scale
        restored = np.zeros((out_h, out_w), dtype=np.float32)
        weight_mask = np.zeros((out_h, out_w), dtype=np.float32)
        
        # Create a linear blending window weight mask for patches
        patch_w = patch_size * scale
        win = np.ones((patch_w, patch_w), dtype=np.float32)
        # Create ramp borders
        ramp = np.linspace(0, 1, overlap * scale)
        for i in range(overlap * scale):
            win[i, :] *= ramp[i]
            win[-i-1, :] *= ramp[i]
            win[:, i] *= ramp[i]
            win[:, -i-1] *= ramp[i]
            
        # Iterate over patches
        stride = patch_size - overlap
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Boundary adjustment
                y_start = min(y, h - patch_size)
                x_start = min(x, w - patch_size)
                
                # Crop LR patch
                lr_patch = lr_np[y_start : y_start + patch_size, x_start : x_start + patch_size]
                lr_t = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0).float().to(self.device)
                
                # Model inference
                pred_patch_t = self.model(lr_t)
                pred_patch_np = pred_patch_t.squeeze().cpu().numpy()
                pred_patch_np = np.clip(pred_patch_np, 0.0, 1.0)
                
                # Stitch HR patch back using window weights
                hy_start = y_start * scale
                hx_start = x_start * scale
                
                restored[hy_start : hy_start + patch_w, hx_start : hx_start + patch_w] += pred_patch_np * win
                weight_mask[hy_start : hy_start + patch_w, hx_start : hx_start + patch_w] += win
                
        # Normalize stitched output
        restored = restored / (weight_mask + 1e-8)
        restored = np.clip(restored, 0.0, 1.0)
        
        # Compute confidence maps
        confidence, deviation, risk = self.compute_confidence_maps(lr_np, restored)
        
        return restored, confidence, deviation, risk

    def compute_confidence_maps(self, lr_np, restored_np):
        """Computes AI modification, cycle-consistency reconstruction error,
        normalized confidence score, and potential hallucination risk map.
        """
        scale = self.scale
        h, w = lr_np.shape
        out_h, out_w = restored_np.shape
        
        # 1. AI Modification (HR resolution): |F(Y) - Bicubic(Y)|
        # Measures structural updates relative to the standard upscaling baseline
        bicubic_upscaled = cv2.resize(lr_np, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        deviation = np.abs(restored_np - bicubic_upscaled)
        deviation = np.clip(deviation, 0.0, 1.0)
        
        # 2. Consistency Error (LR resolution): |Y - D(F(Y))|
        # Downscale the restored image back using Area interpolation to match raw input resolution
        downscaled = cv2.resize(restored_np, (w, h), interpolation=cv2.INTER_AREA)
        reconstruction_error = np.abs(lr_np - downscaled)
        
        # 3. Upscale consistency error to HR resolution for visualization mapping
        reconstruction_error_hr = cv2.resize(reconstruction_error, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        
        # 4. Normalize maps using validation set statistics
        normalized_mod = deviation / (self.mu_modification + 1e-8)
        normalized_const_hr = reconstruction_error_hr / (self.mu_consistency + 1e-8)
        
        # 5. Potential Hallucination Risk Map: NormalizedMod * NormalizedConst
        # High AI change + Low consistency = potentially unreliable restoration region requiring inspection.
        potential_risk = normalized_mod * normalized_const_hr
        potential_risk = np.clip(potential_risk, 0.0, 1.0)  # Bound for visual output display
        
        # 6. Global normalized confidence score
        # Bounded between [0, 1] based on consistency error relative to validation mean
        confidence = 1.0 - np.clip(normalized_const_hr, 0.0, 1.0)
        
        return confidence, deviation, potential_risk
