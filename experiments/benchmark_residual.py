import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.semicon_dataset import SemiconDataset
from models.cnn import EDSRLight
from models.baseline import BicubicBaseline

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim_fn(img1, img2, data_range=1.0)

def compute_sobel_gradients(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobelx**2 + sobely**2)

def calculate_edge_score(img1, img2):
    g1 = compute_sobel_gradients(img1)
    g2 = compute_sobel_gradients(img2)
    g1_flat, g2_flat = g1.flatten(), g2.flatten()
    corr = np.corrcoef(g1_flat, g2_flat)[0, 1] if g1_flat.std() > 1e-8 and g2_flat.std() > 1e-8 else 0.0
    corr = max(0.0, float(corr))
    mae = float(np.mean(np.abs(g1 - g2))) / (float(g2.mean()) + 1e-8)
    return float(0.5 * corr + 0.5 / (1.0 + mae))

def calculate_laplacian_variance(image):
    """Measures image focus / sharpness via variance of Laplacian."""
    lap = cv2.Laplacian(np.float64(image), cv2.CV_64F)
    return float(lap.var())

def calculate_gradient_energy(image):
    """Measures average gradient power."""
    grad = compute_sobel_gradients(image)
    return float(np.mean(grad**2))

def calculate_hf_energy_ratio(image, cutoff_frac=0.25):
    """Radial FFT high-frequency energy fraction."""
    h, w = image.shape
    f = np.fft.fftshift(np.fft.fft2(image))
    mag = np.abs(f)**2
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    radial = np.zeros(max_r)
    counts = np.zeros(max_r)
    for y in range(h):
        for x in range(w):
            r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
            if r < max_r:
                radial[r] += mag[y, x]
                counts[r] += 1
    counts[counts == 0] = 1
    radial /= counts
    cutoff = int(max_r * (1 - cutoff_frac))
    total = radial.sum()
    if total < 1e-12:
        return 0.0
    return float(radial[cutoff:].sum() / total)

def main():
    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking models on device: {device}")
    
    scale = cfg.get("scale", 4)
    test_dir = cfg["dataset"]["test_dir"]
    test_dataset = SemiconDataset(test_dir, scale=scale, patch_size=None, is_train=False)
    
    cnn_cfg = cfg.get("model", {}).get("cnn", {})
    
    # 1. Current Production EDSR (Without global residual)
    current_edsr = EDSRLight(
        scale=scale,
        num_res_blocks=cnn_cfg.get("num_res_blocks", 8),
        num_channels=cnn_cfg.get("num_channels", 48),
        global_residual=False
    ).to(device)
    
    prod_chk = os.path.join(PROJECT_ROOT, "checkpoints", "best_cnn.pth")
    if os.path.exists(prod_chk):
        ckpt = torch.load(prod_chk, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        current_edsr.load_state_dict(sd)
    current_edsr.eval()
    
    # 2. Improved Experimental EDSR (With global residual)
    improved_edsr = EDSRLight(
        scale=scale,
        num_res_blocks=cnn_cfg.get("num_res_blocks", 8),
        num_channels=cnn_cfg.get("num_channels", 48),
        global_residual=True
    ).to(device)
    
    exp_chk = os.path.join(PROJECT_ROOT, "checkpoints", "experimental_edsr_residual.pth")
    if os.path.exists(exp_chk):
        ckpt = torch.load(exp_chk, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        improved_edsr.load_state_dict(sd)
        print(f"Loaded experimental checkpoint from {exp_chk}")
    else:
        print(f"ERROR: Experimental checkpoint {exp_chk} not found!")
        return
        
    improved_edsr.eval()
    
    models = {
        "Bicubic": None,
        "Current EDSR": current_edsr,
        "Improved EDSR (Residual)": improved_edsr
    }
    
    results = {
        m: {
            "psnr": [], "ssim": [], "edge": [],
            "lap_var": [], "grad_energy": [], "hf_energy": []
        } for m in models.keys()
    }
    
    # Track HR stats as reference
    hr_stats = {"lap_var": [], "grad_energy": [], "hf_energy": []}
    
    for idx in range(len(test_dataset)):
        lr_tensor, hr_tensor = test_dataset[idx]
        lr_np = lr_tensor.squeeze().numpy()
        hr_np = hr_tensor.squeeze().numpy()
        
        # Ground Truth HR reference stats
        hr_stats["lap_var"].append(calculate_laplacian_variance(hr_np))
        hr_stats["grad_energy"].append(calculate_gradient_energy(hr_np))
        hr_stats["hf_energy"].append(calculate_hf_energy_ratio(hr_np))
        
        lr_t = lr_tensor.unsqueeze(0).to(device)
        
        outputs = {}
        # Bicubic
        bic_np = cv2.resize(lr_np, (hr_np.shape[1], hr_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        outputs["Bicubic"] = bic_np
        
        # Models (RAW output, NO post-processing!)
        with torch.no_grad():
            c_out = current_edsr(lr_t).squeeze().cpu().numpy()
            outputs["Current EDSR"] = np.clip(c_out, 0.0, 1.0)
            
            i_out = improved_edsr(lr_t).squeeze().cpu().numpy()
            outputs["Improved EDSR (Residual)"] = np.clip(i_out, 0.0, 1.0)
            
        for name, pred in outputs.items():
            results[name]["psnr"].append(calculate_psnr(pred, hr_np))
            results[name]["ssim"].append(calculate_ssim(pred, hr_np))
            results[name]["edge"].append(calculate_edge_score(pred, hr_np))
            results[name]["lap_var"].append(calculate_laplacian_variance(pred))
            results[name]["grad_energy"].append(calculate_gradient_energy(pred))
            results[name]["hf_energy"].append(calculate_hf_energy_ratio(pred))
            
        # Save visual comparison for sample 0, 1, 2
        if idx < 3:
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # 1. LR
            lr_up = cv2.resize(lr_np, (hr_np.shape[1], hr_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            axes[0].imshow(lr_up, cmap="gray")
            axes[0].set_title(f"LR Input (Pixelated)")
            axes[0].axis("off")
            
            # 2. Bicubic
            axes[1].imshow(outputs["Bicubic"], cmap="gray")
            axes[1].set_title(f"Bicubic\nPSNR: {results['Bicubic']['psnr'][-1]:.2f}dB")
            axes[1].axis("off")
            
            # 3. Current EDSR
            axes[2].imshow(outputs["Current EDSR"], cmap="gray")
            axes[2].set_title(f"Current EDSR\nPSNR: {results['Current EDSR']['psnr'][-1]:.2f}dB")
            axes[2].axis("off")
            
            # 4. Improved EDSR
            axes[3].imshow(outputs["Improved EDSR (Residual)"], cmap="gray")
            axes[3].set_title(f"Improved EDSR (Residual)\nPSNR: {results['Improved EDSR (Residual)']['psnr'][-1]:.2f}dB")
            axes[3].axis("off")
            
            # 5. HR Target
            axes[4].imshow(hr_np, cmap="gray")
            axes[4].set_title("Ground Truth HR")
            axes[4].axis("off")
            
            plt.tight_layout()
            out_img_path = os.path.join(PROJECT_ROOT, "experiments", f"controlled_benchmark_sample_{idx}.png")
            plt.savefig(out_img_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visual benchmark to {out_img_path}")

    print("\n" + "="*75)
    print("CONTROLLED BENCHMARK RESULTS (RAW Outputs - NO Post-Processing)")
    print("="*75)
    print(f"Ground Truth HR Reference Stats:")
    print(f"  Laplacian Variance:  {np.mean(hr_stats['lap_var']):.6f}")
    print(f"  Gradient Energy:     {np.mean(hr_stats['grad_energy']):.6f}")
    print(f"  HF Energy Ratio:     {np.mean(hr_stats['hf_energy']):.8f}")
    print("-" * 75)
    
    summary_rows = []
    for name, m in results.items():
        row = {
            "Method": name,
            "PSNR (dB)": np.mean(m["psnr"]),
            "SSIM": np.mean(m["ssim"]),
            "Edge Score": np.mean(m["edge"]),
            "Laplacian Var": np.mean(m["lap_var"]),
            "Gradient Energy": np.mean(m["grad_energy"]),
            "HF Energy Ratio": np.mean(m["hf_energy"])
        }
        summary_rows.append(row)
        print(f"Model: {name}")
        print(f"  PSNR:             {row['PSNR (dB)']:.4f} dB")
        print(f"  SSIM:             {row['SSIM']:.4f}")
        print(f"  Edge Score:       {row['Edge Score']:.4f}")
        print(f"  Laplacian Var:    {row['Laplacian Var']:.6f}")
        print(f"  Gradient Energy:  {row['Gradient Energy']:.6f}")
        print(f"  HF Energy Ratio:  {row['HF Energy Ratio']:.8f}")
        print("-" * 75)
        
    df = pd.DataFrame(summary_rows)
    csv_out = os.path.join(PROJECT_ROOT, "experiments", "residual_benchmark_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved benchmark metrics to {csv_out}")

if __name__ == "__main__":
    main()
