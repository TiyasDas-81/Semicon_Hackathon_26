import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import time
import json
import cv2
import numpy as np
import yaml
import torch
import glob
import re
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
        if model_type == "bicubic":
            checkpoint_path = None
        else:
            checkpoint_name = "best_cnn.pth" if model_type == "cnn" else "best_transformer.pth"
            checkpoint_path = os.path.join("checkpoints", checkpoint_name)
            if not os.path.exists(checkpoint_path):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model checkpoint unavailable: {checkpoint_name} is missing. Please run model training first."
                )
            
        RESTORER_CACHE[model_type] = SemiconImageRestorer(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            config_path=CONFIG_PATH
        )
    return RESTORER_CACHE[model_type]

def get_safe_path(base_dir: str, relative_path: str) -> str:
    """Combines base_dir and relative_path, ensuring no directory traversal occurs."""
    base_dir_abs = os.path.abspath(base_dir)
    joined_path = os.path.join(base_dir_abs, relative_path)
    target_path = os.path.abspath(joined_path)
    
    # Restrict lookup to target directory
    if not target_path.startswith(base_dir_abs + os.sep) and target_path != base_dir_abs:
        raise HTTPException(
            status_code=400, 
            detail="Access denied: invalid path traversal or unsafe file request."
        )
    return target_path

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
            {"id": "cnn", "name": "EDSR-Light CNN (Main / Verified)"},
            {"id": "bicubic", "name": "Bicubic Interpolation (Baseline)"},
            {"id": "transformer", "name": "SwinIR-Light Transformer (Experimental)"}
        ]
    }

@app.get("/api/datasets")
def list_datasets():
    """Returns available datasets and their availability state."""
    return {
        "datasets": [
            {
                "id": "carinthia",
                "name": "Carinthia SEM Defect Dataset",
                "available": os.path.exists("data/raw/carinthia/data/images"),
                "mode": "synthetic"
            },
            {
                "id": "miic",
                "name": "MIIC IC-SEM Interconnect Dataset",
                "available": os.path.exists("data/raw/miic"),
                "mode": "synthetic"
            },
            {
                "id": "nist",
                "name": "NIST SEM Contrast/Noise Stress Test",
                "available": os.path.exists("data/raw/nist"),
                "mode": "real"
            }
        ]
    }

@app.get("/api/datasets/{dataset_id}/samples")
def list_samples(dataset_id: str):
    """Returns a list of available samples for the specified dataset."""
    if dataset_id not in ["carinthia", "miic", "nist"]:
        raise HTTPException(status_code=400, detail=f"Invalid dataset ID: {dataset_id}")
        
    # Check availability
    if dataset_id == "carinthia" and not os.path.exists("data/raw/carinthia/data/images"):
        raise HTTPException(status_code=400, detail="Dataset unavailable — run dataset acquisition.")
    elif dataset_id == "miic" and not os.path.exists("data/raw/miic"):
        raise HTTPException(status_code=400, detail="Dataset unavailable — run dataset acquisition.")
    elif dataset_id == "nist" and not os.path.exists("data/raw/nist"):
        raise HTTPException(status_code=400, detail="Dataset unavailable — run dataset acquisition.")
        
    samples = []
    
    if dataset_id == "carinthia":
        folder = "data/raw/carinthia/data/images"
        paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        for p in paths[:50]: # Return first 50 files for UI loading speed
            fname = os.path.basename(p)
            samples.append({"id": fname, "name": fname})
            
    elif dataset_id == "miic":
        folder = "data/raw/miic"
        paths = sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))
        for p in paths[:50]:
            rel = os.path.relpath(p, folder).replace('\\', '/')
            samples.append({"id": rel, "name": rel})
            
    elif dataset_id == "nist":
        folder = "data/raw/nist"
        paths = sorted(glob.glob(os.path.join(folder, "set*", "*.tiff")))
        for p in paths[:100]: # Allow more for noise/contrast combinations
            rel = os.path.relpath(p, folder).replace('\\', '/')
            samples.append({"id": rel, "name": rel})
            
    return {
        "dataset": dataset_id,
        "samples": samples
    }

@app.get("/api/datasets/{dataset_id}/samples/{sample_id:path}")
def get_sample_metadata(dataset_id: str, sample_id: str):
    """Returns metadata and preview details for a specific sample."""
    if dataset_id not in ["carinthia", "miic", "nist"]:
        raise HTTPException(status_code=400, detail=f"Invalid dataset ID: {dataset_id}")
        
    if dataset_id == "carinthia":
        base_dir = "data/raw/carinthia/data/images"
        sample_path = get_safe_path(base_dir, os.path.basename(sample_id))
    elif dataset_id == "miic":
        base_dir = "data/raw/miic"
        sample_path = get_safe_path(base_dir, sample_id)
    elif dataset_id == "nist":
        base_dir = "data/raw/nist"
        sample_path = get_safe_path(base_dir, sample_id)
        
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found on disk.")
        
    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to parse sample image file.")
        
    h, w = img.shape
    metadata = {
        "id": sample_id,
        "dataset": dataset_id,
        "name": os.path.basename(sample_path),
        "dimensions": f"{w}x{h}",
    }
    
    if dataset_id == "carinthia":
        metadata["source"] = "Carinthia SEM Defect Dataset"
        metadata["degradation_mode"] = "Synthetic (controlled)"
        metadata["reference_status"] = "Original acts as HR reference"
        # Attempt to extract defect class from folder structure if available
        metadata["defect_class"] = "SEM wafer defect"
        
    elif dataset_id == "miic":
        metadata["source"] = "MIIC IC-SEM Interconnect Dataset"
        metadata["degradation_mode"] = "Synthetic (controlled)"
        metadata["reference_status"] = "Original acts as HR reference"
        # Extract category from path
        parts = sample_id.replace("\\", "/").split("/")
        if len(parts) >= 2:
            metadata["category"] = parts[0]
        
    elif dataset_id == "nist":
        metadata["source"] = "NIST SEM Contrast/Noise Stress Test"
        metadata["degradation_mode"] = "Real (in-situ noise/contrast variation)"
        metadata["reference_status"] = "No paired ground truth"
        filename = os.path.basename(sample_path)
        match = re.search(r'noise_(\d+)_contrast_(\d+)', filename)
        if match:
            metadata["noise_level"] = int(match.group(1))
            metadata["contrast_level"] = int(match.group(2))
        # Extract set number
        set_match = re.search(r'set(\d+)', sample_id)
        if set_match:
            metadata["nist_set"] = int(set_match.group(1))
            
    return metadata

@app.post("/api/restore")
async def restore(
    image: UploadFile = File(None), 
    model: str = Form("cnn"), 
    mode: str = Form("synthetic"),
    dataset: str = Form(None),
    sample_id: str = Form(None)
):
    if model not in ["bicubic", "cnn", "transformer"]:
        raise HTTPException(status_code=400, detail="Invalid model selection")
        
    try:
        lr_np = None
        hr_reference_np = None
        filename_label = "custom_upload.png"
        
        # Load sample from dataset if specified
        if dataset:
            if dataset not in ["carinthia", "miic", "nist"]:
                raise HTTPException(status_code=400, detail=f"Invalid dataset ID: {dataset}")
                
            # Verify dataset existence
            if dataset == "carinthia" and not os.path.exists("data/raw/carinthia/data/images"):
                raise HTTPException(status_code=400, detail="Dataset unavailable — run dataset acquisition.")
            elif dataset == "miic" and not os.path.exists("data/raw/miic"):
                raise HTTPException(status_code=400, detail="Dataset unavailable — run dataset acquisition.")
            elif dataset == "nist" and not os.path.exists("data/raw/nist"):
                raise HTTPException(status_code=400, detail="Dataset unavailable — run dataset acquisition.")
                
            # Resolve safe sample path
            if dataset == "carinthia":
                sample_path = get_safe_path("data/raw/carinthia/data/images", os.path.basename(sample_id))
            elif dataset == "miic":
                sample_path = get_safe_path("data/raw/miic", sample_id)
            elif dataset == "nist":
                sample_path = get_safe_path("data/raw/nist", sample_id)
                
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=400, detail=f"Sample not found: {sample_id}")
                
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to load sample image.")
                
            filename_label = os.path.basename(sample_path)
            
            # Handle NIST reference and synthetic degradation
            if dataset == "nist" and mode == "synthetic":
                # Find matching set reference
                match = re.search(r'set(\d+)', sample_id)
                set_num = int(match.group(1)) if match else 1
                ref_path = os.path.join("data/raw/nist", "masks", "masks", f"set{set_num}_cex_noise_000_contrast_100.tiff")
                
                if os.path.exists(ref_path):
                    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                    if ref_img is not None:
                        hr_reference_np = np.float32(ref_img) / 255.0
                        # Synthesize a moderate LR degradation
                        from scripts.generate_degradation_levels import apply_controlled_degradation
                        lr_np, _ = apply_controlled_degradation(hr_reference_np, 1)
                else:
                    # Fallback to direct read if ref is missing
                    lr_np = np.float32(img) / 255.0
            
            elif mode == "synthetic" and dataset in ["carinthia", "miic"]:
                # Original high-quality images act as HR Reference Source Images
                # Standardize to 256x256
                h, w = img.shape
                if h < 256 or w < 256:
                    hr = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                else:
                    ch, cw = h // 2, w // 2
                    hr = img[ch-128:ch+128, cw-128:cw+128]
                hr_reference_np = np.float32(hr) / 255.0
                from scripts.generate_degradation_levels import apply_controlled_degradation
                lr_np, _ = apply_controlled_degradation(hr_reference_np, 1)
                
            else:
                # Real Blind Mode: treat loaded image as degraded directly
                lr_np = np.float32(img) / 255.0
                
        else:
            # File Upload Ingest
            if not image:
                raise HTTPException(status_code=400, detail="No input file or dataset sample provided.")
                
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
                
            lr_np = np.float32(img) / 255.0
            filename_label = image.filename
            
        # Get restorer
        restorer = get_restorer(model)
        
        # Perform restoration
        start_time = time.time()
        restored, confidence, deviation, risk = restorer.restore_image(lr_np, patch_size=64)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics (only if mode is synthetic)
        psnr_val = None
        ssim_val = None
        edge_val = None
        
        if mode == "synthetic":
            # If we generated synthetic degradation from a reference
            if hr_reference_np is not None:
                # Align dimensions
                gt_h, gt_w = hr_reference_np.shape
                if restored.shape != hr_reference_np.shape:
                    restored_eval = cv2.resize(restored, (gt_w, gt_h), interpolation=cv2.INTER_AREA)
                else:
                    restored_eval = restored
                psnr_val = float(calculate_psnr(restored_eval, hr_reference_np))
                ssim_val = float(calculate_ssim(restored_eval, hr_reference_np))
                edge_val = float(calculate_edge_preservation(restored_eval, hr_reference_np))
            else:
                # Look for custom upload ground truth
                gt_path = find_ground_truth(filename_label)
                if gt_path is not None:
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_img is not None and gt_img.shape == restored.shape:
                        gt_np = np.float32(gt_img) / 255.0
                        psnr_val = float(calculate_psnr(restored, gt_np))
                        ssim_val = float(calculate_ssim(restored, gt_np))
                        edge_val = float(calculate_edge_preservation(restored, gt_np))
                        
        # Load warning thresholds if available
        thresholds_path = "configs/warning_thresholds.json"
        warning_conf = 0.6236
        warning_msg = ""
        is_warning = False
        
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, "r") as f:
                    t_data = json.load(f)
                    warning_conf = t_data.get("warning_confidence", 0.6236)
            except Exception:
                pass
                
        mean_conf = float(confidence.mean())
        if mean_conf < warning_conf:
            is_warning = True
            warning_msg = f"Reconstruction consistency ({mean_conf:.3f}) falls below empirical threshold ({warning_conf:.3f}). Inspection required."

        # Prepare response
        res_dict = {
            "filename": filename_label,
            "model": model,
            "mode": mode,
            "inference_time_ms": elapsed_time * 1000.0,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "edge_preservation": edge_val,
            "warning_flag": is_warning,
            "warning_message": warning_msg,
            "warning_threshold": warning_conf,
            "mean_confidence": mean_conf,
            "restored_b64": array_to_base64_png(restored),
            "confidence_b64": colormap_to_base64_png(confidence, cv2.COLORMAP_VIRIDIS),
            "deviation_b64": colormap_to_base64_png(deviation, cv2.COLORMAP_INFERNO),
            "risk_b64": colormap_to_base64_png(risk, cv2.COLORMAP_MAGMA),
            "original_b64": array_to_base64_png(lr_np)
        }
        return res_dict
        
    except HTTPException as he:
        raise he
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
