import os
import cv2
import numpy as np
import yaml
import argparse
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_noise_texture(width, height):
    """Generates a subtle high-frequency texture simulating the silicon wafer substrate."""
    # Base background noise
    bg = np.random.normal(0.4, 0.02, (height, width))
    # Add a bit of low-frequency waviness
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    wave = 0.01 * np.sin(xx / 30.0) + 0.01 * np.cos(yy / 30.0)
    bg = bg + wave
    return np.clip(bg, 0.0, 1.0)

def add_sem_edge_effect(image, edge_intensity=0.25, edge_width=1):
    """Simulates the SEM edge effect where vertical or inclined structures appear brighter due to secondary electron emission."""
    # Find edges using Sobel filters
    sobelx = cv2.Sobel(np.uint8(image * 255), cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(np.uint8(image * 255), cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    grad = grad / (grad.max() + 1e-8)
    
    # Dilate and blur edges for a smoother emission profile
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(np.float32(grad), kernel, iterations=edge_width)
    blurred_edges = cv2.GaussianBlur(dilated_edges, (3, 3), 0.8)
    
    # Add edge highlight
    result = image + edge_intensity * blurred_edges
    return np.clip(result, 0.0, 1.0)

def generate_grating_pattern(width, height, pitch=32, line_width=14, ler_amp=1.5, ler_freq=0.15):
    """Generates line-space gratings with Line Edge Roughness (LER) and random bridge/break defects."""
    canvas = np.zeros((height, width), dtype=np.float32)
    bg = generate_noise_texture(width, height)
    canvas[:] = bg
    
    num_lines = int(width / pitch) + 2
    x_positions = [int(i * pitch - pitch/2) for i in range(num_lines)]
    
    # Randomly select a few lines to have defects
    defect_line_idx = random.sample(range(num_lines), k=min(2, num_lines))
    defect_types = ["bridge", "break", "none"]
    
    for idx, start_x in enumerate(x_positions):
        defect = "none"
        if idx in defect_line_idx:
            defect = random.choice(defect_types)
            
        # Draw the line with edge roughness
        y_coords = np.arange(height)
        # Create correlated line edge roughness using sine waves
        ler_left = ler_amp * np.sin(y_coords * ler_freq + random.uniform(0, 2*np.pi)) + \
                   (ler_amp/2.0) * np.sin(y_coords * ler_freq * 2.5 + random.uniform(0, 2*np.pi))
        ler_right = ler_amp * np.sin(y_coords * ler_freq + random.uniform(0, 2*np.pi)) + \
                    (ler_amp/2.0) * np.sin(y_coords * ler_freq * 2.5 + random.uniform(0, 2*np.pi))
        
        for y in range(height):
            xl = int(start_x - line_width/2 + ler_left[y])
            xr = int(start_x + line_width/2 + ler_right[y])
            
            # Apply line break defect
            if defect == "break" and 100 < y < 140:
                continue # leave substrate
                
            xl = max(0, min(width - 1, xl))
            xr = max(0, min(width - 1, xr))
            canvas[y, xl:xr] = 0.7 # Photoresist gray level
            
        # Apply line bridge defect to next line
        if defect == "bridge":
            bridge_y = random.randint(50, height - 80)
            bridge_h = random.randint(8, 16)
            next_x = start_x + pitch
            for y in range(bridge_y, bridge_y + bridge_h):
                xl = int(start_x + line_width/2)
                xr = int(next_x - line_width/2)
                xl = max(0, min(width - 1, xl))
                xr = max(0, min(width - 1, xr))
                canvas[y, xl:xr] = 0.65
                
    # Add SEM edge effect
    canvas = add_sem_edge_effect(canvas, edge_intensity=0.2, edge_width=1)
    return canvas

def generate_vias_pattern(width, height, spacing=48, radius=12, defect_rate=0.1):
    """Generates contact holes (vias) on a grid with missing contacts and edge roughness."""
    canvas = np.zeros((height, width), dtype=np.float32)
    bg = generate_noise_texture(width, height)
    canvas[:] = bg
    
    x_coords = np.arange(spacing/2, width, spacing)
    y_coords = np.arange(spacing/2, height, spacing)
    
    for cx in x_coords:
        for cy in y_coords:
            if random.random() < defect_rate:
                continue # Missing via defect
                
            # Draw circle with slight border roughness
            # We can draw it as a filled polygon with perturbed radius
            num_points = 32
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            poly_pts = []
            
            # Random roughness parameters
            r_noise = np.random.normal(0, 0.8, num_points)
            
            for i, angle in enumerate(angles):
                r = radius + r_noise[i]
                px = int(cx + r * np.cos(angle))
                py = int(cy + r * np.sin(angle))
                poly_pts.append((px, py))
                
            cv2.fillPoly(canvas, [np.array(poly_pts, dtype=np.int32)], 0.15) # oxide via depth is darker
            
    # Add SEM edge effect
    canvas = add_sem_edge_effect(canvas, edge_intensity=0.35, edge_width=1)
    return canvas

def generate_logic_pattern(width, height, num_shapes=8):
    """Generates complex logic pattern layout with rectangular features and random particle contaminations."""
    canvas = np.zeros((height, width), dtype=np.float32)
    bg = generate_noise_texture(width, height)
    canvas[:] = bg
    
    # Draw logic shapes
    for _ in range(num_shapes):
        sx = random.randint(20, width - 60)
        sy = random.randint(20, height - 60)
        sw = random.randint(30, 80)
        sh = random.randint(20, 50)
        
        # Draw L-shape or rectangular shapes
        gray_level = random.choice([0.6, 0.8])
        if random.random() > 0.5:
            # L-shape
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), gray_level, -1)
            offset = random.choice([15, 25])
            cv2.rectangle(canvas, (sx, sy), (sx + offset, sy + sh + offset), gray_level, -1)
        else:
            # simple rectangle
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), gray_level, -1)
            
    # Add particle defect (contamination)
    if random.random() > 0.3:
        px = random.randint(40, width - 40)
        py = random.randint(40, height - 40)
        pr = random.randint(5, 12)
        # Random polygon representing dust/particle
        num_pts = random.randint(5, 8)
        pts = []
        for i in range(num_pts):
            angle = i * (2 * np.pi / num_pts)
            dist = pr * random.uniform(0.6, 1.4)
            pts.append((int(px + dist * np.cos(angle)), int(py + dist * np.sin(angle))))
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 0.95) # particle is bright/conductive
        
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5) # slightly round the lithographic corners
    # Add SEM edge effect
    canvas = add_sem_edge_effect(canvas, edge_intensity=0.25, edge_width=1)
    return canvas

def apply_sem_degradation(image, scale=4, cfg=None):
    """Applies realistic Scanning Electron Microscope (SEM) degradations to clean HR image:
    1. Anisotropic Gaussian Blur (defocus + astigmatism)
    2. Downsampling
    3. Poisson Noise (electron shot noise)
    4. Gaussian Noise (amplifier/sensor noise)
    5. Brightness / Contrast shift
    6. JPEG compression
    """
    if cfg is None:
        cfg = {}
        
    # Read config ranges or set defaults
    blur_sigma_min = cfg.get("blur_sigma_min", 0.5)
    blur_sigma_max = cfg.get("blur_sigma_max", 2.0)
    poisson_noise_scale_min = cfg.get("poisson_noise_scale_min", 0.5)
    poisson_noise_scale_max = cfg.get("poisson_noise_scale_max", 2.0)
    gaussian_noise_sigma_min = cfg.get("gaussian_noise_sigma_min", 0.005)
    gaussian_noise_sigma_max = cfg.get("gaussian_noise_sigma_max", 0.03)
    jpeg_quality_min = cfg.get("jpeg_quality_min", 40)
    jpeg_quality_max = cfg.get("jpeg_quality_max", 90)
    contrast_min = cfg.get("contrast_min", 0.7)
    contrast_max = cfg.get("contrast_max", 1.3)
    brightness_min = cfg.get("brightness_min", -0.15)
    brightness_max = cfg.get("brightness_max", 0.15)

    h, w = image.shape
    
    # 1. Anisotropic Blur (simulating astigmatism/defocus)
    sigma_x = random.uniform(blur_sigma_min, blur_sigma_max)
    sigma_y = random.uniform(blur_sigma_min, blur_sigma_max)
    
    # Create anisotropic Gaussian kernel
    kx = int(round(sigma_x * 3.0)) * 2 + 1
    ky = int(round(sigma_y * 3.0)) * 2 + 1
    kx = max(3, min(15, kx))
    ky = max(3, min(15, ky))
    
    blur_img = cv2.GaussianBlur(image, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)
    
    # 2. Downsampling (HR -> LR)
    lr_h, lr_w = h // scale, w // scale
    lr_img = cv2.resize(blur_img, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    
    # 3. Poisson Noise (electron shot noise)
    # Scale to simulate electron count, generate Poisson, then scale back
    poisson_scale = random.uniform(poisson_noise_scale_min, poisson_noise_scale_max) * 100.0
    poisson_img = np.clip(lr_img, 0.0, 1.0)
    noisy_poisson = np.random.poisson(poisson_img * poisson_scale) / poisson_scale
    
    # 4. Gaussian Noise (electronic detector noise)
    gauss_sigma = random.uniform(gaussian_noise_sigma_min, gaussian_noise_sigma_max)
    noise_gauss = np.random.normal(0, gauss_sigma, (lr_h, lr_w))
    noisy_img = noisy_poisson + noise_gauss
    
    # 5. Contrast and Brightness
    c = random.uniform(contrast_min, contrast_max)
    b = random.uniform(brightness_min, brightness_max)
    adjusted_img = (noisy_img - 0.5) * c + 0.5 + b
    adjusted_img = np.clip(adjusted_img, 0.0, 1.0)
    
    # 6. JPEG / compression artifacts
    jpeg_quality = random.randint(jpeg_quality_min, jpeg_quality_max)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', np.uint8(adjusted_img * 255), encode_param)
    lr_final = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    lr_final = np.float32(lr_final) / 255.0
    
    return lr_final

def generate_split(num_samples, split_name, output_dir, scale, cfg):
    """Generates a dataset split (train, val, test) consisting of matching HR and LR images."""
    os.makedirs(os.path.join(output_dir, "hr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lr"), exist_ok=True)
    
    img_size = cfg.get("dataset", {}).get("image_size", 256)
    
    patterns = ["grating", "vias", "logic"]
    
    for i in range(num_samples):
        pattern_type = random.choice(patterns)
        
        # Generate HR pattern
        if pattern_type == "grating":
            pitch = random.randint(28, 48)
            lw = random.randint(10, 20)
            hr = generate_grating_pattern(img_size, img_size, pitch=pitch, line_width=lw)
        elif pattern_type == "vias":
            spacing = random.randint(40, 56)
            rad = random.randint(9, 14)
            hr = generate_vias_pattern(img_size, img_size, spacing=spacing, radius=rad)
        else:
            num_sh = random.randint(5, 10)
            hr = generate_logic_pattern(img_size, img_size, num_shapes=num_sh)
            
        # Apply degradation to generate LR counterpart
        degrade_cfg = cfg.get("degradation", {})
        lr = apply_sem_degradation(hr, scale=scale, cfg=degrade_cfg)
        
        # Save images
        hr_filename = f"{pattern_type}_{i:04d}_hr.png"
        lr_filename = f"{pattern_type}_{i:04d}_lr.png"
        
        cv2.imwrite(os.path.join(output_dir, "hr", hr_filename), np.uint8(hr * 255))
        cv2.imwrite(os.path.join(output_dir, "lr", lr_filename), np.uint8(lr * 255))

def main():
    parser = argparse.ArgumentParser(description="Procedural Semiconductor Image & Degradation Dataset Generator")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    set_seed(cfg.get("seed", 42))
    
    scale = cfg.get("scale", 4)
    train_dir = cfg.get("dataset", {}).get("train_dir", "data/train")
    val_dir = cfg.get("dataset", {}).get("val_dir", "data/val")
    test_dir = cfg.get("dataset", {}).get("test_dir", "data/test")
    
    num_train = cfg.get("dataset", {}).get("num_samples_train", 120)
    num_val = cfg.get("dataset", {}).get("num_samples_val", 30)
    num_test = cfg.get("dataset", {}).get("num_samples_test", 30)
    
    print(f"Generating dataset with scale factor {scale}...")
    print(f"Train samples: {num_train} in {train_dir}")
    print(f"Val samples: {num_val} in {val_dir}")
    print(f"Test samples: {num_test} in {test_dir}")
    
    # Generate splits
    # Set seed differently for each split to ensure variance, but keep it reproducible
    set_seed(cfg.get("seed", 42) + 1)
    generate_split(num_train, "train", train_dir, scale, cfg)
    
    set_seed(cfg.get("seed", 42) + 2)
    generate_split(num_val, "val", val_dir, scale, cfg)
    
    set_seed(cfg.get("seed", 42) + 3)
    generate_split(num_test, "test", test_dir, scale, cfg)
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
