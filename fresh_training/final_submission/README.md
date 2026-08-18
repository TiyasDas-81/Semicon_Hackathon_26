# KLA SemiCon AI Hackathon 2026 — Track 2 Official Submission Package
## Team AIvengers Submission (PS01)

### 1. Project Title & Overview
**AI-Based Restoration of Degraded Images for Semiconductor Inspection_KLA**
This repository contains the complete production-grade submission package for Track 2 / PS01 of the KLA SemiCon India Hackathon 2026. The solution restores low-dose, degraded semiconductor inspection images (128x128 single-channel grayscale) to clean 256x256 high-resolution images.

### 2. Team Information
- **Team Name**: `AIvengers`
- **Institution**: `Vellore Institute of Technology, Vellore`
- **Team Members**:
  - `Tiyas Das` — Team Leader & ML/AI Lead (Architecture & Loss Optimization)
  - `Soumen Mondal` — Model Development & Training Engineer (EDSR2x Training & Engineering)
  - `Partha Protim Mondal` — Data Analysis & Evaluation Engineer (Degradation Forensics & Metrics)
  - `Aryan Raj` — Deployment, Integration & Presentation Lead (Packaging & Inference Pipeline)

### 3. Problem Statement Addressed
Semiconductor wafer inspection during manufacturing uses Scanning Electron Microscopy (SEM). Low-beam-dose imaging causes severe **Speckle Noise**, **Gaussian Noise**, and **Spatial Resolution Loss (128x128 -> 256x256 2x Super-Resolution)** simultaneously. Conventional spatial filters cause over-smoothing. Our solution jointly removes noise and restores 2x spatial resolution.

### 4. Model Architecture & Specifications
- **Model Name**: `EDSR2x + L1 Loss`
- **Parameter Count**: `776,705` (~0.78M parameters)
- **Structure**: 8 Residual Blocks, 64 Channels, PixelShuffle 2x, Global Bicubic Skip Connection
- **Training Loss**: Pure L1 Reconstruction Loss

### 5. Official Performance Benchmark Results
| Evaluation Split | Method | PSNR (dB) | SSIM | MAE | Edge Score | Win Rate vs Bicubic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official Validation Set (320 samples)** | Bicubic 2x | `22.59 dB` | `0.5166` | `0.0597` | `0.4727` | — |
| **Official Validation Set (320 samples)** | **EDSR2x (Production)** | **`27.42 dB`** | **`0.7357`** | **`0.0349`** | **`0.6639`** | `320/320` (**100.0%**) |
| **Validation Net Gain** | **EDSR2x vs Bicubic** | **`+4.83 dB`** | **`+0.2191`** | **`-0.0248`** | **`+0.1912`** | `88.8%` SSIM Win |
| **Held-Out Internal Test (320 samples)** | Bicubic 2x | `22.98 dB` | `0.5243` | `0.0572` | `0.4781` | — |
| **Held-Out Internal Test (320 samples)** | **EDSR2x (Production)** | **`27.93 dB`** | **`0.7408`** | **`0.0310`** | **`0.6684`** | `320/320` (**100.0%**) |

### 6. Hardware Audit & Speed
- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU
- **Inference Latency**: `2.95 ms / image`
- **Batch Throughput**: `88.1 FPS`
- **Peak VRAM**: `111.6 MB`

### 7. Code Repository & Assets
- **GitHub Repository**: `https://github.com/TiyasDas-81/Semicon_Hackathon_26.git`
- **Demo Video**: [`fresh_training/final_submission/EDSR2x_demo.mp4`](file:///C:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/EDSR2x_demo.mp4) (83s, 1080p, 30 FPS)
- **Demo Script**: [`fresh_training/final_submission/demo_script.md`](file:///C:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/demo_script.md)
- **Standalone Evaluation Script**: `python evaluation.py --input <test_images_dir> --output <restored_output_dir>`

### 8. Package Contents
- `AIvengers_KLA_PS01.pptx` (Official 9-Slide Presentation)
- `AIvengers_KLA_PS01.pdf` (Official PDF Presentation)
- `evaluation.py` (Standalone AS-IS evaluation script)
- `requirements.txt` (Complete Python dependencies)
- `fresh_training/final_submission/model/best_kla_2x.pth` (Production model weights)
- `fresh_training/final_submission/test_outputs/` (400 Restored test sample `.npy` files)
