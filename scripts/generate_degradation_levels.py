import os
import cv2
import numpy as np
import glob
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def apply_controlled_degradation(image, level, scale=4):
    """Applies a controlled level of SEM degradation to clean HR image at a fixed scale factor (4x):
    Level 1: Mild (low blur/noise)
    Level 2: Moderate (moderate blur/noise)
    Level 3: Severe (high blur/noise)
    Level 4: Extreme (extreme blur/noise)
    """
    h, w = image.shape
    
    # Define degradation parameters per level
    if level == 1:
        kernel_size = 3
        blur_sigma = 0.6
        poisson_scale = 250.0  # High electron count = low noise
        gauss_sigma = 0.005
        jpeg_quality = 90
    elif level == 2:
        kernel_size = 5
        blur_sigma = 1.2
        poisson_scale = 100.0  # Moderate electron count
        gauss_sigma = 0.015
        jpeg_quality = 70
    elif level == 3:
        kernel_size = 7
        blur_sigma = 2.0
        poisson_scale = 50.0   # Low electron count = high noise
        gauss_sigma = 0.03
        jpeg_quality = 50
    elif level == 4:
        kernel_size = 9
        blur_sigma = 3.0
        poisson_scale = 20.0   # Extreme noise
        gauss_sigma = 0.05
        jpeg_quality = 30
    else:
        raise ValueError(f"Unknown degradation level: {level}")
        
    # 1. Blur
    blur_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_sigma)
    
    # 2. Downsample
    lr_h, lr_w = h // scale, w // scale
    lr_img = cv2.resize(blur_img, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    
    # 3. Poisson Noise
    noisy_poisson = np.random.poisson(np.clip(lr_img, 0.0, 1.0) * poisson_scale) / poisson_scale
    
    # 4. Gaussian Noise
    noise_gauss = np.random.normal(0, gauss_sigma, (lr_h, lr_w))
    noisy_img = noisy_poisson + noise_gauss
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    
    # 5. JPEG Compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', np.uint8(noisy_img * 255), encode_param)
    lr_final = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    lr_final = np.float32(lr_final) / 255.0
    
    return lr_final, scale

def generate_study_set(src_pattern, dest_dir, num_samples=30):
    """Generates study pairs across 4 degradation levels from source images."""
    set_seed(42)
    files = glob.glob(src_pattern, recursive=True)
    if not files:
        print(f"No source files matched pattern: {src_pattern}")
        return False
        
    # Select a random subset of files for the study
    selected_files = random.sample(files, min(num_samples, len(files)))
    print(f"Selected {len(selected_files)} images for degradation study from {os.path.dirname(src_pattern)}")
    
    for level in [1, 2, 3, 4]:
        lvl_dir_hr = os.path.join(dest_dir, f"level{level}", "hr")
        lvl_dir_lr = os.path.join(dest_dir, f"level{level}", "lr")
        os.makedirs(lvl_dir_hr, exist_ok=True)
        os.makedirs(lvl_dir_lr, exist_ok=True)
        
        for idx, f in enumerate(selected_files):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Crop/Resize source to a standard HR size e.g. 256x256
            # Ensure it is divisible by 8 (max scale factor)
            h, w = img.shape
            if h < 256 or w < 256:
                img_hr = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            else:
                # Center crop to 256x256
                ch, cw = h // 2, w // 2
                img_hr = img[ch-128:ch+128, cw-128:cw+128]
                
            img_hr_float = np.float32(img_hr) / 255.0
            
            # Apply degradation
            img_lr_float, scale = apply_controlled_degradation(img_hr_float, level)
            
            # Save files
            cv2.imwrite(os.path.join(lvl_dir_hr, f"sample_{idx:03d}_hr.png"), img_hr)
            cv2.imwrite(os.path.join(lvl_dir_lr, f"sample_{idx:03d}_lr.png"), np.uint8(img_lr_float * 255))
            
        print(f"Generated Degradation Level {level} (scale: {scale}x) at {lvl_dir_lr}")
    return True

if __name__ == "__main__":
    # We will invoke this once datasets are downloaded
    # Source pattern: data/raw/carinthia/**/*.jpg or data/raw/carinthia/**/*.png
    # Destination directory: data/degraded_study
    print("Controlled Degradation Levels script ready.")
