#!/usr/bin/env python3
"""
KLA Track 2 — AI-Based SEM Image Restoration (Offline Inference Entry Point)

Usage:
    python run.py <input-dir> <output-dir>
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F

# Ensure model definition can be imported relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.model_definition import EDSRLight

def load_model(checkpoint_path, device):
    """Loads trained EDSR model weights offline without any external downloads."""
    model = EDSRLight(
        scale=4,
        num_res_blocks=8,
        num_channels=48,
        in_channels=1,
        out_channels=1,
        global_residual=True
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint missing at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_array(arr):
    """
    Validates and converts input numpy array to [0, 1] float32 grayscale tensor (1, 1, H, W).
    Tracks original dimensional structure (2D vs 3D).
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input is not a valid numpy array.")
        
    orig_shape = arr.shape
    is_2d = len(orig_shape) == 2
    
    if len(orig_shape) == 2:
        h, w = orig_shape
    elif len(orig_shape) == 3 and orig_shape[2] == 1:
        h, w = orig_shape[0], orig_shape[1]
        arr = arr[:, :, 0]
    elif len(orig_shape) == 3 and orig_shape[0] == 1:
        h, w = orig_shape[1], orig_shape[2]
        arr = arr[0, :, :]
    else:
        raise ValueError(f"Unsupported array shape: {orig_shape}. Expected (H, W) or (H, W, 1).")
        
    # Handle data types and range normalization safely
    if arr.dtype == np.uint8:
        img_float = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        img_float = arr.astype(np.float32) / 65535.0
    else:
        img_float = arr.astype(np.float32)
        # Check if values are in [0, 255] range instead of [0, 1]
        max_val = np.nanmax(img_float) if img_float.size > 0 else 0.0
        if max_val > 1.5:
            img_float = img_float / 255.0
            
    # Clean any input NaN/Inf
    img_float = np.nan_to_num(img_float, nan=0.0, posinf=1.0, neginf=0.0)
    img_float = np.clip(img_float, 0.0, 1.0)
    
    # Format to (1, 1, H, W) PyTorch Tensor
    tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
    return tensor, is_2d, (h, w)

def restore_image(model, tensor, device, scale=4):
    """Executes single-pass or patch-based super-resolution inference."""
    _, _, h, w = tensor.shape
    
    with torch.no_grad():
        tensor = tensor.to(device)
        # EDSR is fully convolutional and handles arbitrary spatial dimensions
        if h <= 512 and w <= 512:
            out_tensor = model(tensor)
        else:
            # Overlapping patch tiling for very large input images
            patch_size = 128
            overlap = 32
            stride = patch_size - overlap
            out_h, out_w = h * scale, w * scale
            patch_w = patch_size * scale
            
            restored = torch.zeros((1, 1, out_h, out_w), device=device)
            weights = torch.zeros((1, 1, out_h, out_w), device=device)
            
            # Linear blending window
            win = torch.ones((patch_w, patch_w), device=device)
            ramp = torch.linspace(0, 1, overlap * scale, device=device)
            for i in range(overlap * scale):
                win[i, :] *= ramp[i]
                win[-i-1, :] *= ramp[i]
                win[:, i] *= ramp[i]
                win[:, -i-1] *= ramp[i]
            win = win.unsqueeze(0).unsqueeze(0)
            
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y_s = min(y, h - patch_size)
                    x_s = min(x, w - patch_size)
                    
                    patch = tensor[:, :, y_s : y_s + patch_size, x_s : x_s + patch_size]
                    patch_out = model(patch)
                    
                    hy_s, hx_s = y_s * scale, x_s * scale
                    restored[:, :, hy_s : hy_s + patch_w, hx_s : hx_s + patch_w] += patch_out * win
                    weights[:, :, hy_s : hy_s + patch_w, hx_s : hx_s + patch_w] += win
                    
            out_tensor = restored / (weights + 1e-8)
            
    out_np = out_tensor.squeeze().cpu().numpy()
    return out_np

def postprocess_array(out_np, is_2d):
    """Sanitizes output array, enforces [0, 1] range, float32 dtype, and target shape."""
    out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
    out_np = np.clip(out_np, 0.0, 1.0).astype(np.float32)
    
    if not is_2d:
        out_np = np.expand_dims(out_np, axis=-1)
        
    return out_np

def main():
    parser = argparse.ArgumentParser(description="Offline SEM Image Restoration Inference (KLA Track 2)")
    parser.add_argument("input_dir", type=str, help="Path to directory containing input .npy files")
    parser.add_argument("output_dir", type=str, help="Path to directory where restored .npy files will be saved")
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Device selection (Automatic CUDA detection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # Load model
    checkpoint_path = os.path.join(SCRIPT_DIR, "models", "best_cnn.pth")
    try:
        model = load_model(checkpoint_path, device)
        print(f"Loaded restoration model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Find all .npy files
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        sys.exit(0)
        
    print(f"Found {len(npy_files)} .npy files to process.")
    
    success_count = 0
    for idx, filepath in enumerate(npy_files, 1):
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(output_dir, filename)
        
        try:
            arr = np.load(filepath)
            tensor, is_2d, (in_h, in_w) = preprocess_array(arr)
            
            restored_np = restore_image(model, tensor, device, scale=4)
            final_output = postprocess_array(restored_np, is_2d)
            
            np.save(out_filepath, final_output)
            success_count += 1
            print(f"[{idx}/{len(npy_files)}] Processed: {filename} -> Shape: {final_output.shape}, Dtype: {final_output.dtype}")
        except Exception as e:
            print(f"[{idx}/{len(npy_files)}] Error processing {filename}: {e}")
            
    print(f"\nSuccessfully processed {success_count}/{len(npy_files)} files into {output_dir}")

if __name__ == "__main__":
    main()
