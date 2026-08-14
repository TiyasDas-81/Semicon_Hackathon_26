import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.semicon_dataset import SemiconDataset
from models.cnn import EDSRLight
from models.transformer import SwinIRLight
from losses.restoration_losses import CompoundRestorationLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) for [0, 1] normalized PyTorch tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculates a quick structural similarity index (SSIM) metric using kornia-like window formulation."""
    # We can use the SSIM Loss class internally here by using 1.0 - loss
    from losses.restoration_losses import SSIMLoss
    ssim_metric = SSIMLoss(channel=1).to(img1.device)
    loss_val = ssim_metric(img1, img2)
    return 1.0 - loss_val.item()

def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_total = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_edge = 0.0
    
    for lr, hr in tqdm(loader, desc="  Training", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr)
            loss, loss_dict = criterion(pred, hr)
            loss.backward()
            optimizer.step()
            
        running_total += loss.item() * lr.size(0)
        running_l1 += loss_dict["l1"] * lr.size(0)
        running_ssim += loss_dict["ssim"] * lr.size(0)
        running_edge += loss_dict["edge"] * lr.size(0)
        
    num_samples = len(loader.dataset)
    return {
        "loss": running_total / num_samples,
        "l1": running_l1 / num_samples,
        "ssim": running_ssim / num_samples,
        "edge": running_edge / num_samples
    }

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_total = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    
    for lr, hr in tqdm(loader, desc="  Validation", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        
        loss, _ = criterion(pred, hr)
        running_total += loss.item() * lr.size(0)
        
        # Calculate evaluation metrics
        for b in range(lr.size(0)):
            p_img = pred[b:b+1]
            g_img = hr[b:b+1]
            running_psnr += calculate_psnr(p_img, g_img).item()
            running_ssim += calculate_ssim(p_img, g_img)
            
    num_samples = len(loader.dataset)
    return {
        "loss": running_total / num_samples,
        "psnr": running_psnr / num_samples,
        "ssim": running_ssim / num_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Train Super-Resolution Models for Wafer Inspection")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "transformer"], help="Model type to train")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    set_seed(cfg.get("seed", 42))
    
    # Output directories setup
    chk_dir = cfg.get("train", {}).get("checkpoint_dir", "checkpoints")
    log_dir = cfg.get("train", {}).get("log_dir", "logs")
    os.makedirs(chk_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloaders setup
    scale = cfg.get("scale", 4)
    patch_size = cfg.get("dataset", {}).get("patch_size", 64)
    batch_size = cfg.get("train", {}).get("batch_size", 16)
    
    train_dataset = SemiconDataset(cfg["dataset"]["train_dir"], scale=scale, patch_size=patch_size, is_train=True)
    val_dataset = SemiconDataset(cfg["dataset"]["val_dir"], scale=scale, patch_size=None, is_train=False) # validate on full size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Instantiate Model
    if args.model == "cnn":
        cnn_cfg = cfg.get("model", {}).get("cnn", {})
        model = EDSRLight(
            scale=scale,
            num_res_blocks=cnn_cfg.get("num_res_blocks", 6),
            num_channels=cnn_cfg.get("num_channels", 32)
        )
    else:
        trans_cfg = cfg.get("model", {}).get("transformer", {})
        model = SwinIRLight(
            scale=scale,
            embed_dim=trans_cfg.get("embed_dim", 48),
            depths=trans_cfg.get("depths", [4, 4]),
            num_heads=trans_cfg.get("num_heads", [4, 4]),
            window_size=trans_cfg.get("window_size", 8),
            mlp_ratio=trans_cfg.get("mlp_ratio", 2.0)
        )
        
    model = model.to(device)
    print(f"Initialized {args.model.upper()} model. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & Optimizer
    criterion = CompoundRestorationLoss().to(device)
    lr = float(cfg.get("train", {}).get("learning_rate", 0.0005))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate cosine annealing scheduler
    epochs = cfg.get("train", {}).get("epochs", 15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    use_amp = cfg.get("train", {}).get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    
    # Logs tracking
    history = []
    best_psnr = 0.0
    patience = cfg.get("train", {}).get("early_stopping_patience", 5)
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        scheduler.step()
        
        # Validation
        if epoch == 1 or epoch % cfg.get("train", {}).get("val_interval", 2) == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"  [Train] Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, SSIM: {train_metrics['ssim']:.4f}, Edge: {train_metrics['edge']:.4f})")
            print(f"  [Val]   Loss: {val_metrics['loss']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
            
            # Log history
            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_l1": train_metrics["l1"],
                "train_ssim_loss": train_metrics["ssim"],
                "train_edge_loss": train_metrics["edge"],
                "val_loss": val_metrics["loss"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "lr": optimizer.param_groups[0]["lr"]
            })
            
            # Save best checkpoint
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                patience_counter = 0
                checkpoint_path = os.path.join(chk_dir, f"best_{args.model}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": best_psnr,
                    "val_ssim": val_metrics["ssim"]
                }, checkpoint_path)
                print(f"  ==> Saved new BEST checkpoint (PSNR: {best_psnr:.2f} dB) to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered after {epoch} epochs.")
                    break
        else:
            print(f"  [Train] Loss: {train_metrics['loss']:.4f}")
            
    # Save final checkpoint
    final_path = os.path.join(chk_dir, f"final_{args.model}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")
    
    # Save training history to CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(log_dir, f"history_{args.model}.csv"), index=False)
    print(f"Saved training history to {os.path.join(log_dir, f'history_{args.model}.csv')}")

if __name__ == "__main__":
    main()
