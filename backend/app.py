import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import time
import cv2
import numpy as np
import yaml
import torch
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from inference.restorer import SemiconImageRestorer
from evaluation.evaluator import calculate_psnr, calculate_ssim, calculate_edge_preservation

app = FastAPI(title="Semicon Image Restoration API")

# Global cache for restorers to avoid loading weights on every request
RESTORER_CACHE = {}
CONFIG_PATH = "configs/default.yaml"

def get_restorer(model_type: str):
    """Loads and caches restorer models to optimize inference speeds."""
    if model_type not in RESTORER_CACHE:
        checkpoint_name = "best_cnn.pth" if model_type == "cnn" else "best_transformer.pth"
        checkpoint_path = os.path.join("checkpoints", checkpoint_name)
        
        # If no checkpoint exists (e.g. before training), restorer falls back to initialized weights
        if model_type == "bicubic":
            checkpoint_path = None
            
        RESTORER_CACHE[model_type] = SemiconImageRestorer(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            config_path=CONFIG_PATH
        )
    return RESTORER_CACHE[model_type]

def array_to_base64_png(arr):
    """Converts a grayscale float array [0,1] to base64 PNG."""
    img_u8 = np.uint8(np.clip(arr, 0.0, 1.0) * 255)
    _, buffer = cv2.imencode('.png', img_u8)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def colormap_to_base64_png(arr, colormap=cv2.COLORMAP_JET):
    """Applies a colormap to a grayscale float array [0,1] and converts to base64 PNG."""
    img_u8 = np.uint8(np.clip(arr, 0.0, 1.0) * 255)
    colored = cv2.applyColorMap(img_u8, colormap)
    _, buffer = cv2.imencode('.png', colored)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def find_ground_truth(filename: str):
    """Looks for matching ground truth HR image in test or val dirs based on LR filename."""
    # LR filename typically looks like: grating_0001_lr.png
    # HR filename is: grating_0001_hr.png
    hr_name = filename.replace("_lr.png", "_hr.png")
    
    # Check test and val folders
    search_paths = [
        os.path.join("data", "test", "hr", hr_name),
        os.path.join("data", "val", "hr", hr_name),
        os.path.join("data", "train", "hr", hr_name)
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

@app.get("/api/health")
def health():
    return {"status": "ok", "gpu_available": torch.cuda.is_available() if 'torch' in globals() or 'torch' in sys.modules else False}

@app.get("/api/models")
def list_models():
    return {
        "models": [
            {"id": "bicubic", "name": "Bicubic Interpolation (Baseline)"},
            {"id": "cnn", "name": "EDSR-Light CNN"},
            {"id": "transformer", "name": "SwinIR-Light Transformer (Main)"}
        ]
    }

@app.post("/api/restore")
async def restore(image: UploadFile = File(...), model: str = Form("transformer")):
    if model not in ["bicubic", "cnn", "transformer"]:
        raise HTTPException(status_code=400, detail="Invalid model selection")
        
    try:
        # Read uploaded image bytes
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
            
        lr_np = np.float32(img) / 255.0
        
        # Get restorer
        restorer = get_restorer(model)
        
        # Perform restoration
        start_time = time.time()
        restored, confidence, deviation = restorer.restore_image(lr_np, patch_size=64)
        elapsed_time = time.time() - start_time
        
        # Look for ground truth to calculate metrics (for interactive demo evaluation)
        gt_path = find_ground_truth(image.filename)
        psnr_val = None
        ssim_val = None
        edge_val = None
        
        if gt_path is not None:
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None and gt_img.shape == restored.shape:
                gt_np = np.float32(gt_img) / 255.0
                psnr_val = calculate_psnr(restored, gt_np)
                ssim_val = calculate_ssim(restored, gt_np)
                edge_val = calculate_edge_preservation(restored, gt_np)
                
        # Prepare response
        return {
            "filename": image.filename,
            "model": model,
            "inference_time_ms": elapsed_time * 1000.0,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "edge_preservation": edge_val,
            "restored_b64": array_to_base64_png(restored),
            "confidence_b64": colormap_to_base64_png(confidence, cv2.COLORMAP_JET),
            "deviation_b64": colormap_to_base64_png(deviation, cv2.COLORMAP_HOT),
            "original_b64": array_to_base64_png(lr_np)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")

@app.get("/api/presets/{filename}")
def get_preset(filename: str):
    """Delivers preset LR sample images from test folder to UI dashboard."""
    preset_path = os.path.join("data", "test", "lr", filename)
    if not os.path.exists(preset_path):
        raise HTTPException(status_code=404, detail="Preset file not found. Ensure dataset generation is run first.")
    return FileResponse(preset_path)

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join("backend", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index file not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()
