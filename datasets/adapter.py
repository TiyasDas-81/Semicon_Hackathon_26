import os
import glob
import cv2
import numpy as np
from abc import ABC, abstractmethod
from scripts.generate_degradation_levels import apply_controlled_degradation

class SEMObservation:
    """Standardized representation of a Scanning Electron Microscopy observation."""
    def __init__(self, degraded_image, ground_truth=None, is_paired=False, metadata=None):
        self.degraded_image = degraded_image  # np.ndarray (LR image, float32, range [0, 1])
        self.ground_truth = ground_truth      # np.ndarray or None (HR image, float32, range [0, 1])
        self.is_paired = is_paired            # bool (True if ground_truth is available and aligned)
        self.metadata = metadata or {}        # dict (extra metrics, noise parameters, labels)

class DatasetAdapter(ABC):
    """Abstract Base Class for all SEM Dataset Adapters."""
    @abstractmethod
    def __len__(self):
        pass
        
    @abstractmethod
    def __getitem__(self, idx):
        pass

class CarinthiaAdapter(DatasetAdapter):
    """Adapter for Carinthia SEM Defect Dataset (Primary Semiconductor)."""
    def __init__(self, root_dir="data/raw/carinthia", degradation_level=2, num_samples=None):
        self.root_dir = root_dir
        self.degradation_level = degradation_level
        # Carinthia images are located under data/raw/carinthia/data/images/
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "data", "images", "*.jpg")))
        if num_samples:
            self.image_paths = self.image_paths[:num_samples]
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load Carinthia image: {path}")
            
        # Standardize HR to 256x256
        h, w = img.shape
        if h < 256 or w < 256:
            hr = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else:
            ch, cw = h // 2, w // 2
            hr = img[ch-128:ch+128, cw-128:cw+128]
            
        hr_float = np.float32(hr) / 255.0
        
        # Apply synthetic degradation to create LR
        lr_float, scale = apply_controlled_degradation(hr_float, self.degradation_level)
        
        metadata = {
            "dataset": "Carinthia",
            "source_path": path,
            "degradation_type": f"Synthetic Level {self.degradation_level}",
            "original_dimensions": f"{w}x{h}"
        }
        
        return SEMObservation(
            degraded_image=lr_float,
            ground_truth=hr_float,
            is_paired=True,
            metadata=metadata
        )

class MIICAdapter(DatasetAdapter):
    """Adapter for MIIC Microscopic Images of Integrated Circuits (IC SEM Domain)."""
    def __init__(self, root_dir="data/raw/miic", degradation_level=2, num_samples=None):
        self.root_dir = root_dir
        self.degradation_level = degradation_level
        # MIIC images are located under Anomaly_test or Inpainting_test as .jpg files
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True))
        if num_samples:
            self.image_paths = self.image_paths[:num_samples]
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load MIIC image: {path}")
            
        # Standardize HR to 256x256
        h, w = img.shape
        if h < 256 or w < 256:
            hr = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else:
            ch, cw = h // 2, w // 2
            hr = img[ch-128:ch+128, cw-128:cw+128]
            
        hr_float = np.float32(hr) / 255.0
        lr_float, scale = apply_controlled_degradation(hr_float, self.degradation_level)
        
        metadata = {
            "dataset": "MIIC",
            "source_path": path,
            "degradation_type": f"Synthetic Level {self.degradation_level}",
            "original_dimensions": f"{w}x{h}",
            "role": "High-quality source image used to generate synthetic degradation"
        }
        
        return SEMObservation(
            degraded_image=lr_float,
            ground_truth=hr_float,
            is_paired=True,
            metadata=metadata
        )

class NISTAdapter(DatasetAdapter):
    """Adapter for NIST Detection Limits for SEM Image Segmentation (Noise/Contrast Controlled)."""
    def __init__(self, root_dir="data/raw/nist", set_num=1, paired=True, num_samples=None):
        self.root_dir = root_dir
        self.set_num = set_num
        self.paired = paired
        
        # NIST images are located under data/raw/nist/set*/
        self.folder_path = os.path.join(root_dir, f"set{set_num}")
        self.image_paths = sorted(glob.glob(os.path.join(self.folder_path, "*.tiff")))
        if num_samples:
            self.image_paths = self.image_paths[:num_samples]
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load NIST image: {path}")
            
        lr_float = np.float32(img) / 255.0
        
        # Parse noise and contrast parameters from filename
        # e.g., set1_cex_noise_007_contrast_001.tiff
        filename = os.path.basename(path)
        import re
        match = re.search(r'noise_(\d+)_contrast_(\d+)', filename)
        noise_param = int(match.group(1)) if match else 0
        contrast_param = int(match.group(2)) if match else 100
        
        hr_float = None
        is_paired = False
        
        if self.paired:
            # We look for the reference image in the masks subfolder
            # e.g., masks/set1_cex_noise_000_contrast_100.tiff
            ref_path = os.path.join(self.root_dir, "masks", "masks", f"set{self.set_num}_cex_noise_000_contrast_100.tiff")
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                if ref_img is not None:
                    hr_float = np.float32(ref_img) / 255.0
                    is_paired = True
                    
        metadata = {
            "dataset": f"NIST Set {self.set_num}",
            "source_path": path,
            "noise_level": noise_param,
            "contrast_level": contrast_param,
            "degradation_type": "Real Controlled (ARTIMAGEN SEM Simulation)",
            "paired_ground_truth": "Yes (set*_cex_noise_000_contrast_100.tiff)" if is_paired else "No"
        }
        
        return SEMObservation(
            degraded_image=lr_float,
            ground_truth=hr_float,
            is_paired=is_paired,
            metadata=metadata
        )
