import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import yaml

from datasets.semicon_dataset import SemiconDataset
from inference.restorer import SemiconImageRestorer

def compute_sobel_gradients(image):
    """Computes Sobel gradient magnitude of a grayscale image."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    return grad

def calculate_psnr(img1, img2):
    """Calculates PSNR for numpy arrays [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculates structural similarity index using OpenCV's structural comparison or scikit-image."""
    from skimage.metrics import structural_similarity as ssim
    # Images are in [0, 1] range
    return ssim(img1, img2, data_range=1.0)

def calculate_edge_preservation(img1, img2):
    """Calculates Edge Preservation Score based on L1 difference between Sobel gradient maps."""
    grad1 = compute_sobel_gradients(img1)
    grad2 = compute_sobel_gradients(img2)
    
    # Normalize gradients
    grad1_max = grad1.max()
    grad2_max = grad2.max()
    if grad1_max > 0: grad1 = grad1 / grad1_max
    if grad2_max > 0: grad2 = grad2 / grad2_max
    
    mae_grad = np.mean(np.abs(grad1 - grad2))
    score = 1.0 - np.clip(mae_grad, 0.0, 1.0)
    return score

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark Semiconductor Image Restoration Models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    os.makedirs("experiments", exist_ok=True)
    scale = cfg.get("scale", 4)
    test_dir = cfg["dataset"]["test_dir"]
    
    # Load test dataset
    test_dataset = SemiconDataset(test_dir, scale=scale, patch_size=None, is_train=False)
    print(f"Loaded {len(test_dataset)} test samples.")
    
    # Load restorers
    checkpoint_dir = cfg.get("train", {}).get("checkpoint_dir", "checkpoints")
    
    print("Loading restorers...")
    restorers = {
        "Bicubic": SemiconImageRestorer(model_type="bicubic", config_path=args.config),
        "CNN (EDSR)": SemiconImageRestorer(model_type="cnn", checkpoint_path=os.path.join(checkpoint_dir, "best_cnn.pth"), config_path=args.config),
        "Transformer (SwinIR-Light)": SemiconImageRestorer(model_type="transformer", checkpoint_path=os.path.join(checkpoint_dir, "best_transformer.pth"), config_path=args.config)
    }
    
    # Results dictionary
    results = {m: {"psnr": [], "ssim": [], "edge": [], "time": []} for m in restorers.keys()}
    
    # Evaluate sample by sample
    for idx in range(len(test_dataset)):
        # Load LR and HR images
        # Datasets outputs tensors, let's load them as numpy arrays
        lr_tensor, hr_tensor = test_dataset[idx]
        lr_np = lr_tensor.squeeze().numpy()
        hr_np = hr_tensor.squeeze().numpy()
        
        # Determine sample name from dataset file list
        sample_path = test_dataset.lr_paths[idx]
        sample_name = os.path.basename(sample_path).replace("_lr.png", "")
        
        saved_visuals = {}
        confidence_map = None
        deviation_map = None
        
        for name, restorer in restorers.items():
            start_time = time.time()
            # Perform restoration (disable patching for test size 256x256 as it fits in memory easily)
            restored, conf, dev = restorer.restore_image(lr_np, patch_size=None)
            inference_time = time.time() - start_time
            
            psnr = calculate_psnr(restored, hr_np)
            ssim_val = calculate_ssim(restored, hr_np)
            edge_score = calculate_edge_preservation(restored, hr_np)
            
            results[name]["psnr"].append(psnr)
            results[name]["ssim"].append(ssim_val)
            results[name]["edge"].append(edge_score)
            results[name]["time"].append(inference_time)
            
            saved_visuals[name] = restored
            if name == "Transformer (SwinIR-Light)":
                confidence_map = conf
                deviation_map = dev
                
        # Generate and save comparison plot for first few test samples
        if idx < 3:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            axes[0, 0].imshow(hr_np, cmap="gray")
            axes[0, 0].set_title("Ground Truth HR")
            axes[0, 0].axis("off")
            
            axes[0, 1].imshow(lr_np, cmap="gray")
            axes[0, 1].set_title(f"Degraded LR ({lr_np.shape[0]}x{lr_np.shape[1]})")
            axes[0, 1].axis("off")
            
            axes[0, 2].imshow(saved_visuals["Bicubic"], cmap="gray")
            axes[0, 2].set_title(f"Bicubic\nPSNR: {results['Bicubic']['psnr'][-1]:.2f}dB")
            axes[0, 2].axis("off")
            
            axes[0, 3].imshow(saved_visuals["CNN (EDSR)"], cmap="gray")
            axes[0, 3].set_title(f"CNN (EDSR)\nPSNR: {results['CNN (EDSR)']['psnr'][-1]:.2f}dB")
            axes[0, 3].axis("off")
            
            axes[1, 0].imshow(saved_visuals["Transformer (SwinIR-Light)"], cmap="gray")
            axes[1, 0].set_title(f"Transformer\nPSNR: {results['Transformer (SwinIR-Light)']['psnr'][-1]:.2f}dB")
            axes[1, 0].axis("off")
            
            # Confidence map
            im_conf = axes[1, 1].imshow(confidence_map, cmap="jet", vmin=0.0, vmax=1.0)
            axes[1, 1].set_title("Confidence Map\n(Consistency)")
            axes[1, 1].axis("off")
            plt.colorbar(im_conf, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Deviation / details added map
            im_dev = axes[1, 2].imshow(deviation_map, cmap="hot")
            axes[1, 2].set_title("Deviation Map\n(|Transformer - Bicubic|)")
            axes[1, 2].axis("off")
            plt.colorbar(im_dev, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            # Edge gradient map
            trans_grad = compute_sobel_gradients(saved_visuals["Transformer (SwinIR-Light)"])
            axes[1, 3].imshow(trans_grad, cmap="gray")
            axes[1, 3].set_title("Restored Edge Map\n(Sobel Gradient)")
            axes[1, 3].axis("off")
            
            plt.tight_layout()
            plt.savefig(f"experiments/comparison_sample_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save Zoomed comparison figure (focus on a 64x64 sub-region of HR center)
            center = hr_np.shape[0] // 2
            crop_size_hr = 64
            crop_size_lr = crop_size_hr // scale
            
            c_y_hr, c_x_hr = center, center
            c_y_lr, c_x_lr = center // scale, center // scale
            
            crop_hr = hr_np[c_y_hr - crop_size_hr//2 : c_y_hr + crop_size_hr//2, c_x_hr - crop_size_hr//2 : c_x_hr + crop_size_hr//2]
            crop_lr = lr_np[c_y_lr - crop_size_lr//2 : c_y_lr + crop_size_lr//2, c_x_lr - crop_size_lr//2 : c_x_lr + crop_size_lr//2]
            crop_bic = saved_visuals["Bicubic"][c_y_hr - crop_size_hr//2 : c_y_hr + crop_size_hr//2, c_x_hr - crop_size_hr//2 : c_x_hr + crop_size_hr//2]
            crop_cnn = saved_visuals["CNN (EDSR)"][c_y_hr - crop_size_hr//2 : c_y_hr + crop_size_hr//2, c_x_hr - crop_size_hr//2 : c_x_hr + crop_size_hr//2]
            crop_trans = saved_visuals["Transformer (SwinIR-Light)"][c_y_hr - crop_size_hr//2 : c_y_hr + crop_size_hr//2, c_x_hr - crop_size_hr//2 : c_x_hr + crop_size_hr//2]
            
            # We plot them next to each other
            fig_crop, axes_crop = plt.subplots(1, 5, figsize=(15, 3.5))
            axes_crop[0].imshow(crop_hr, cmap="gray")
            axes_crop[0].set_title("GT HR (Zoom)")
            axes_crop[0].axis("off")
            
            # Resize LR crop using nearest neighbor to show pixelation
            crop_lr_up = cv2.resize(crop_lr, (crop_size_hr, crop_size_hr), interpolation=cv2.INTER_NEAREST)
            axes_crop[1].imshow(crop_lr_up, cmap="gray")
            axes_crop[1].set_title("LR Pixelated")
            axes_crop[1].axis("off")
            
            axes_crop[2].imshow(crop_bic, cmap="gray")
            axes_crop[2].set_title("Bicubic")
            axes_crop[2].axis("off")
            
            axes_crop[3].imshow(crop_cnn, cmap="gray")
            axes_crop[3].set_title("CNN (EDSR)")
            axes_crop[3].axis("off")
            
            axes_crop[4].imshow(crop_trans, cmap="gray")
            axes_crop[4].set_title("Transformer")
            axes_crop[4].axis("off")
            
            plt.tight_layout()
            plt.savefig(f"experiments/zoomed_comparison_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
    # Calculate averages and report
    summary_data = []
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Averages across test set):")
    print("="*50)
    
    for name, metrics in results.items():
        avg_psnr = np.mean(metrics["psnr"])
        avg_ssim = np.mean(metrics["ssim"])
        avg_edge = np.mean(metrics["edge"])
        avg_time = np.mean(metrics["time"])
        
        print(f"Model: {name}")
        print(f"  PSNR: {avg_psnr:.4f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  Edge Preservation Score: {avg_edge:.4f}")
        print(f"  Avg Inference Time: {avg_time * 1000.0:.2f} ms")
        print("-"*50)
        
        summary_data.append({
            "Method": name,
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "Edge Score": avg_edge,
            "Inference Time (ms)": avg_time * 1000.0
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("experiments/model_comparison.csv", index=False)
    print("Saved evaluation summary to experiments/model_comparison.csv")

if __name__ == "__main__":
    main()
