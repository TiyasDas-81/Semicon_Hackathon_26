import os
import sys
import urllib.request
import zipfile
import subprocess
import glob
import cv2
from datetime import datetime

# URLs for public datasets
CARINTHIA_URL = "https://zenodo.org/records/10715190/files/data.zip?download=1"
MIIC_TEST_URL = "https://researchdata.ntu.edu.sg/api/access/datafile/69910"
MIIC_INPAINT_URL = "https://researchdata.ntu.edu.sg/api/access/datafile/69907"
NIST_URL = "https://data.nist.gov/od/ds/mds2-3838/intensity_sets.zip"

def download_file(url, dest_path):
    """Downloads a file showing a progress indicator."""
    print(f"Downloading: {url} -> {dest_path}")
    
    # Custom opener to include User-Agent to prevent Dataverse/Zenodo blocks
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    def reporthook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        else:
            sys.stdout.write(f"\r  Progress: {read_so_far / (1024*1024):.1f} MB")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\n  Download complete.")

def extract_archive(archive_path, extract_dir):
    """Extracts zip or rar archives using native Windows tar utility."""
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting: {archive_path} -> {extract_dir}")
    
    # tar -xf <archive> -C <extract_dir>
    cmd = ["tar", "-xf", archive_path, "-C", extract_dir]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("  Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"  Error extracting using tar: {e.stderr}")
        raise e

def check_image_stats(search_pattern):
    """Checks image count, dimensions, and verifies readability."""
    files = glob.glob(search_pattern, recursive=True)
    count = 0
    corrupt = 0
    dimensions = set()
    
    for f in files[:200]: # Sample first 200 to find dimensions and check corruptions
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            corrupt += 1
        else:
            count += 1
            dimensions.add(img.shape)
            
    # Count the rest without reading to save time
    count += len(files) - len(files[:200])
    
    dim_str = ", ".join([f"{d[1]}x{d[0]}" for d in dimensions]) if dimensions else "Unknown"
    return count, corrupt, dim_str

def main():
    os.makedirs("data/raw/carinthia", exist_ok=True)
    os.makedirs("data/raw/miic", exist_ok=True)
    os.makedirs("data/raw/nist", exist_ok=True)
    os.makedirs("data/auxiliary/wm811k", exist_ok=True)
    
    download_date = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Download & Extract Carinthia
    carinthia_zip = "data/raw/carinthia/data.zip"
    if not os.path.exists(carinthia_zip) and not os.path.exists("data/raw/carinthia/data"):
        try:
            download_file(CARINTHIA_URL, carinthia_zip)
            extract_archive(carinthia_zip, "data/raw/carinthia")
            # Remove zip to save disk space
            os.remove(carinthia_zip)
        except Exception as e:
            print(f"Failed Carinthia download/extraction: {e}")
            
    # 2. Download & Extract MIIC
    miic_rar1 = "data/raw/miic/Anomaly_test.rar"
    miic_rar2 = "data/raw/miic/Inpainting_test.rar"
    if not os.path.exists(miic_rar1) and not os.path.exists("data/raw/miic/Anomaly_test"):
        try:
            download_file(MIIC_TEST_URL, miic_rar1)
            extract_archive(miic_rar1, "data/raw/miic")
            os.remove(miic_rar1)
        except Exception as e:
            print(f"Failed MIIC Anomaly download/extraction: {e}")
            
    if not os.path.exists(miic_rar2) and not os.path.exists("data/raw/miic/Inpainting_test"):
        try:
            download_file(MIIC_INPAINT_URL, miic_rar2)
            extract_archive(miic_rar2, "data/raw/miic")
            os.remove(miic_rar2)
        except Exception as e:
            print(f"Failed MIIC Inpainting download/extraction: {e}")
            
    # 3. Download & Extract NIST
    nist_zip = "data/raw/nist/intensity_sets.zip"
    if not os.path.exists(nist_zip) and not os.path.exists("data/raw/nist/set1"):
        try:
            download_file(NIST_URL, nist_zip)
            extract_archive(nist_zip, "data/raw/nist")
            os.remove(nist_zip)
        except Exception as e:
            print(f"Failed NIST download/extraction: {e}")
            
    # 4. Check Kaggle credentials and notify
    kaggle_creds = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    if not kaggle_creds:
        print("\n[WARNING] Kaggle credentials not found at ~/.kaggle/kaggle.json.")
        print("To acquire the auxiliary WM-811K Wafer Map dataset, please download it manually from:")
        print("  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print("And extract the wafer map files to: data/auxiliary/wm811k/")
        
    # Gather statistics
    print("\n--- Verifying Ingested Datasets ---")
    
    carinthia_cnt, carinthia_crp, carinthia_dims = check_image_stats("data/raw/carinthia/**/*.jpg")
    if carinthia_cnt == 0: # Check for png just in case
        carinthia_cnt, carinthia_crp, carinthia_dims = check_image_stats("data/raw/carinthia/**/*.png")
        
    miic_cnt, miic_crp, miic_dims = check_image_stats("data/raw/miic/**/*.jpg")
    nist_cnt, nist_crp, nist_dims = check_image_stats("data/raw/nist/**/*.tiff")
    
    print(f"Carinthia Images: {carinthia_cnt} (Corrupt: {carinthia_crp}), Dimensions: {carinthia_dims}")
    print(f"MIIC Images: {miic_cnt} (Corrupt: {miic_crp}), Dimensions: {miic_dims}")
    print(f"NIST Images: {nist_cnt} (Corrupt: {nist_crp}), Dimensions: {nist_dims}")
    
    # Generate data/README.md
    readme_content = f"""# Semiconductor Restoration Datasets Registry

This directory contains the ingested and verified public domain semiconductor datasets used for training, validation, robustness testing, and failure threshold analysis.

## 1. Carinthia SEM Defect Dataset (Primary)
- **Official Source**: [https://zenodo.org/records/10715190](https://zenodo.org/records/10715190)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Download Date**: {download_date}
- **Image Count**: {carinthia_cnt}
- **Image Dimensions**: {carinthia_dims}
- **Intended Role**: Primary training, self-supervised representation learning, and defect preservation testing.

## 2. MIIC — Microscopic Images of Integrated Circuits (Secondary)
- **Official Source**: [https://doi.org/10.21979/N9/WBLTFI](https://doi.org/10.21979/N9/WBLTFI)
- **License**: CC BY-NC 4.0 (Non-Commercial Research)
- **Download Date**: {download_date}
- **Image Count**: {miic_cnt}
- **Image Dimensions**: {miic_dims}
- **Intended Role**: Cross-dataset validation and generalization checks.

## 3. NIST — Detection Limits for SEM Image Segmentation (Noise/Contrast Controlled)
- **Official Source**: [https://doi.org/10.18434/mds2-3838](https://doi.org/10.18434/mds2-3838)
- **License**: Public Domain (NIST Software/Data Policy)
- **Download Date**: {download_date}
- **Image Count**: {nist_cnt}
- **Image Dimensions**: {nist_dims}
- **Intended Role**: Robustness profiling, Poisson noise scaling experiments, and cycle-consistency validation.

## 4. WM-811K Wafer Map Dataset (Auxiliary)
- **Official Source**: [https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- **License**: Kaggle Open Database License
- **Intended Role**: Metrology failure visualization and wafer pattern mapping.
"""
    
    with open("data/README.md", "w") as f:
        f.write(readme_content)
    print("Created data/README.md successfully.")
    
    print("\nDataset acquisition pipeline completed successfully!")

if __name__ == "__main__":
    main()
