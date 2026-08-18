# AIvengers Official Submission Package
## Project: AI-Based Restoration of Degraded Images for Semiconductor Inspection_KLA
### SEMICON India Hackathon 2026 — Track 2 / PS01

---

### Team Information
- **Team Name**: `AIvengers`
- **Institution**: `Vellore Institute of Technology, Vellore`
- **Team Members**:
  1. `Tiyas Das` — Team Leader & ML/AI Lead
  2. `Soumen Mondal` — Model Development & Training Engineer
  3. `Partha Protim Mondal` — Data Analysis & Evaluation Engineer
  4. `Aryan Raj` — Deployment, Integration & Presentation Lead

---

### Official Execution Command
To execute restoration on any directory of degraded `.npy` images using our trained EDSR2x model:

```bash
python run.py <input-dir> <output-dir>
```

#### Example Usage:
```bash
python run.py Test_NoisyLR/NoisyLR scratch/restored_outputs/
```

---

### Model Architecture & Specs
- **Model Name**: `EDSR2x + L1 Loss`
- **Parameters**: `776,705` (~0.78M parameters)
- **Structure**: 8 Residual Blocks, 64 Feature Channels, 2x PixelShuffle Upsampler, Global Bicubic Residual Skip Connection
- **Weights File**: `models/best_kla_2x.pth`

---

### Input & Output Format Specifications
- **Input Format**: Directory containing $128 \times 128$ single-channel float32 `.npy` degraded images.
- **Output Format**: Directory populated with restored $256 \times 256$ single-channel float32 `.npy` images.
- **Numerical Bounds**: Strictly bounded in $[0.0, 1.0]$ with $0$ NaNs and $0$ Infs.
- **GPU Throughput**: $\sim 2.95\text{ ms / image}$ ($88.1\text{ FPS}$) on NVIDIA GeForce RTX 3050 Laptop GPU ($111.6\text{ MB}$ peak VRAM).

---

### Performance Summary
- **Official Validation PSNR**: `27.42 dB` (+$4.83$ dB over Bicubic)
- **Official Validation SSIM**: `0.7357` (+$0.2191$ over Bicubic)
- **Held-Out Internal Test PSNR**: `27.93 dB` (+$4.95$ dB over Bicubic)
- **Held-Out Internal Test SSIM**: `0.7408` (+$0.2165$ over Bicubic)
