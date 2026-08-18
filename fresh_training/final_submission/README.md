# KLA SemiCon AI Hackathon 2026 — Track 2 Official Submission Package
## AIvengers Team Submission

### Model Architecture: EDSR2x + L1 Loss
- **Scale Factor**: 2x Super-Resolution (128x128 NoisyLR -> 256x256 Restored HR)
- **Parameters**: `776,705` (`0.78M` params)
- **Blocks**: 8 Residual Blocks, 64 Feature Channels, 2x PixelShuffle
- **Global Residual**: `bicubic_2x(input) + learned_residual`

### Performance Metrics (Official Validation Split - 320 Samples)
- **PSNR**: `27.42 dB` (vs Bicubic `22.59 dB`)
- **SSIM**: `0.7357` (vs Bicubic `0.5166`)
- **MAE**: `0.0349`
- **Edge Preservation Score**: `0.6639`
- **PSNR Win Rate vs Bicubic**: `320/320` (`100.0%`)
- **SSIM Win Rate vs Bicubic**: `284/320` (`88.8%`)

### Held-Out Internal Test Set (320 Samples)
- **PSNR**: `27.93 dB` (vs Bicubic `22.98 dB`)
- **SSIM**: `0.7408` (vs Bicubic `0.5243`)
- **MAE**: `0.0310`

### Execution & Latency
- **GPU Latency**: `2.95 ms / image` on NVIDIA RTX 3050 GPU (`88.1 FPS` batch throughput)
- **Peak VRAM**: `111.6 MB`

### Official Presentation Demo Video
- **Demo Video**: [`fresh_training/final_submission/EDSR2x_demo.mp4`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/EDSR2x_demo.mp4) (`83s`, 1080p, 30 FPS)
- **Demo Assets**: [`fresh_training/final_submission/demo_assets/`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/demo_assets)
- **Demo Script**: [`fresh_training/final_submission/demo_script.md`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/demo_script.md)

### Usage Command
```bash
python fresh_training/final_submission/inference.py --input <path_to_input_dir> --output <path_to_output_dir>
```
