import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconDataset(Dataset):
    """PyTorch Dataset for paired HR-LR semiconductor wafer images."""
    def __init__(self, data_dir, scale=4, patch_size=64, is_train=True, augment=True):
        self.data_dir = data_dir
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment
        
        # Load all LR and HR file paths
        self.hr_paths = sorted(glob.glob(os.path.join(data_dir, "hr", "*_hr.png")))
        self.lr_paths = sorted(glob.glob(os.path.join(data_dir, "lr", "*_lr.png")))
        
        assert len(self.hr_paths) == len(self.lr_paths), \
            f"Mismatched number of HR ({len(self.hr_paths)}) and LR ({len(self.lr_paths)}) images!"
            
    def __len__(self):
        return len(self.hr_paths)
        
    def __getitem__(self, idx):
        # Load grayscale images
        hr_img = cv2.imread(self.hr_paths[idx], cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(self.lr_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Normalize to [0, 1]
        hr_img = np.float32(hr_img) / 255.0
        lr_img = np.float32(lr_img) / 255.0
        
        # Perform random patching during training
        if self.is_train and self.patch_size is not None:
            lr_h, lr_w = lr_img.shape
            lr_patch_size = self.patch_size
            hr_patch_size = lr_patch_size * self.scale
            
            # Select random top-left corner in LR coordinate space
            y_lr = np.random.randint(0, lr_h - lr_patch_size + 1)
            x_lr = np.random.randint(0, lr_w - lr_patch_size + 1)
            
            # Aligned crop
            lr_patch = lr_img[y_lr : y_lr + lr_patch_size, x_lr : x_lr + lr_patch_size]
            
            y_hr = y_lr * self.scale
            x_hr = x_lr * self.scale
            hr_patch = hr_img[y_hr : y_hr + hr_patch_size, x_hr : x_hr + hr_patch_size]
        else:
            lr_patch = lr_img
            hr_patch = hr_img
            
        # Data augmentation
        if self.is_train and self.augment:
            # 1. Random horizontal flip
            if np.random.random() > 0.5:
                lr_patch = np.fliplr(lr_patch)
                hr_patch = np.fliplr(hr_patch)
            # 2. Random vertical flip
            if np.random.random() > 0.5:
                lr_patch = np.flipud(lr_patch)
                hr_patch = np.flipud(hr_patch)
            # 3. Random 90-degree rotations
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                lr_patch = np.rot90(lr_patch, rot_k)
                hr_patch = np.rot90(hr_patch, rot_k)
                
        # Convert to C, H, W tensors (PyTorch format)
        # Ensure contiguous arrays for PyTorch
        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_patch)).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_patch)).unsqueeze(0).float()
        
        return lr_tensor, hr_tensor
