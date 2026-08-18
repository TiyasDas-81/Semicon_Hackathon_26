# KLA SemiCon AI Hackathon 2026 — Track 2 / PS01 Official Submission
## AI-Based Restoration of Degraded Images for Semiconductor Inspection
### Team AIvengers — Vellore Institute of Technology, Vellore

---

### Executive Summary
This repository contains the complete, production-grade submission package for **Track 2 (PS01)** of the **KLA SemiCon India Hackathon 2026**.

Our winning model—**EDSR2x + L1 Loss** (776,705 parameters, ~0.78M)—restores low-dose degraded semiconductor inspection images (128x128 single-channel float32 .npy files) into high-resolution restored images (256x256 float32 .npy files) jointly removing **Speckle Noise**, **Gaussian Noise**, and performing **2x Spatial Super-Resolution**.

---

### Quick Start: AS-IS Evaluation Command
To run inference on any directory of degraded `.npy` images using our trained model:

```bash
python evaluation.py --input <path_to_test_images_directory> --output <path_to_output_directory>
```

#### Example Execution on Official Test Set:
```bash
python evaluation.py --input Test_NoisyLR/NoisyLR --output scratch/restored_outputs/
```

- **Input Format**: Directory containing $128 \times 128$ float32 `.npy` files.
- **Output Format**: Directory populated with restored $256 \times 256$ float32 `.npy` files bounded strictly in $[0.0, 1.0]$.
- **Execution Time**: $\sim 2.95\text{ ms / image}$ on NVIDIA RTX 3050 GPU ($88.1\text{ FPS}$ batch throughput).

---

### Team AIvengers (Vellore Institute of Technology, Vellore)
1. **Tiyas Das** — Team Leader & ML/AI Lead (*Architecture Design & Loss Optimization*)
2. **Soumen Mondal** — Model Development & Training Engineer (*EDSR2x Model Training & Pipeline*)
3. **Partha Protim Mondal** — Data Analysis & Evaluation Engineer (*Degradation Forensics & Metrics Audit*)
4. **Aryan Raj** — Deployment, Integration & Presentation Lead (*Inference Pipeline & Deliverables Packaging*)

---

### Measured Official Validation & Held-Out Test Benchmarks

| Evaluation Split | Method | PSNR (dB) | SSIM | MAE | Edge Preservation | Win Rate vs Bicubic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official Validation (320 samples)** | Bicubic 2× Baseline | `22.59 dB` | `0.5166` | `0.0597` | `0.4727` | — |
| **Official Validation (320 samples)** | **EDSR2x (Production Winner)** | **`27.42 dB`** | **`0.7357`** | **`0.0349`** | **`0.6639`** | `320/320` (**100.0%**) |
| **Validation Net Improvement** | **EDSR2x vs Bicubic** | **`+4.83 dB`** | **`+0.2191`** | **`-0.0248`** | **`+0.1912`** | `284/320` (**88.8%** SSIM) |
| **Held-Out Internal Test (320 samples)** | Bicubic 2× Baseline | `22.98 dB` | `0.5243` | `0.0572` | `0.4781` | — |
| **Held-Out Internal Test (320 samples)** | **EDSR2x (Production Winner)** | **`27.93 dB`** | **`0.7408`** | **`0.0310`** | **`0.6684`** | `320/320` (**100.0%**) |
| **Held-Out Test Net Improvement** | **EDSR2x vs Bicubic** | **`+4.95 dB`** | **`+0.2165`** | **`-0.0262`** | **`+0.1903`** | `285/320` (**89.1%** SSIM) |

*Note: Official Hackathon Test Set (400 samples in `Test_NoisyLR/`) restored outputs are generated under `fresh_training/final_submission/test_outputs/`. Official benchmark scores will be evaluated by contest organizers.*

---

### Hardware Audit & Model Efficiency
- **Parameters**: `776,705` ($\sim 0.78\text{M}$ parameters)
- **Model Size**: `9.37 MB` PyTorch checkpoint (`fresh_training/final_submission/model/best_kla_2x.pth`)
- **GPU Steady-State Latency**: `2.95 ms / image` on NVIDIA GeForce RTX 3050 Laptop GPU
- **Peak VRAM Usage**: `111.6 MB` ($<3\%$ of 4GB VRAM limit)

---

### Key Deliverables & Repository Structure

```text
Semicon_Hackathon_26/
├── AIvengers_KLA_PS01.pptx        ← Official 9-Slide Presentation
├── AIvengers_KLA_PS01.pdf         ← Official PDF Presentation
├── evaluation.py                  ← Standalone AS-IS Evaluation Script
├── requirements.txt               ← Complete Python dependencies
│
├── fresh_training/final_submission/
│   ├── model/best_kla_2x.pth      ← Verified production checkpoint (SHA-256 matched)
│   ├── test_outputs/              ← 400 Restored official test sample outputs (.npy)
│   ├── training/train_edsr2x.py   ← Reproduction training script
│   ├── demo_assets/               ← 10 High-Definition presentation slides (1080p)
│   ├── EDSR2x_demo.mp4            ← Official 83s 1080p demo presentation video
│   ├── validation_results.json    ← Full metric breakout
│   └── final_report.md            ← Comprehensive final report
└── README.md                      ← Root repository documentation
```

---

### Codebase Links & Presentation Assets
- **Public GitHub Repository**: [https://github.com/TiyasDas-81/Semicon_Hackathon_26.git](https://github.com/TiyasDas-81/Semicon_Hackathon_26.git)
- **Official Demo Video**: [`fresh_training/final_submission/EDSR2x_demo.mp4`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/EDSR2x_demo.mp4)
- **Demo Script & Storyboard**: [`fresh_training/final_submission/demo_script.md`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/demo_script.md)
