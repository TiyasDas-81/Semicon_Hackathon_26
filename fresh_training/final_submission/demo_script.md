# Official EDSR2x Hackathon Demo Script & Storyboard
## KLA SemiCon AI Hackathon 2026 — Track 2 Presentation Package

### Video Overview
- **File**: [`fresh_training/final_submission/EDSR2x_demo.mp4`](file:///c:/Users/Asus/Desktop/Semicon/Semicon_Hackathon_26/fresh_training/final_submission/EDSR2x_demo.mp4)
- **Duration**: `83 seconds` (`01:23`)
- **Resolution**: `1920x1080` (1080p, 30 FPS)
- **Target Audience**: KLA Hackathon Evaluators, AI Engineers, Semiconductor Domain Experts

---

### Timed Storyboard & Narration Breakdown

| Timestamp | Section | Visual Asset Used | Screen Text & Visual Content | Narration / Voiceover Script |
| :--- | :--- | :--- | :--- | :--- |
| **00:00 – 00:07** | **Section 1: Title** | `demo_01_problem.png` | "AI-Based Restoration of Degraded Images for Semiconductor Inspection_KLA" | *"Semiconductor inspection images can suffer from severe electron shot noise and limited spatial resolution. Our solution uses EDSR2x to restore degraded 128 by 128 images into 256 by 256 high-resolution outputs."* |
| **00:07 – 00:16** | **Section 2: Problem** | `demo_02_architecture.png` | "The Challenge: Low-Dose SEM Degradation" (Micro-structure zoom callout) | *"Low-dose Scanning Electron Microscopy exhibits heavy Poisson-Gaussian shot noise and high-frequency spatial attenuation, severely degrading feature visibility."* |
| **00:16 – 00:25** | **Section 3: Solution** | `demo_03_comparison_01.png` | "Our Solution: EDSR2x Neural Architecture" (~0.78M params) | *"We engineered EDSR2x—a lightweight single-stage residual network with 8 residual blocks and global bicubic residual skip connections, trained on pure L1 reconstruction loss."* |
| **00:25 – 00:40** | **Section 4: Visuals** | `demo_04_comparison_02.png` | Side-by-side wafer structure restoration (Bicubic vs EDSR2x) | *"Across standard wafer structures, EDSR2x cleanly removes Poisson-Gaussian noise while preserving edge boundaries, achieving a 4.83 dB PSNR gain over bicubic interpolation."* |
| **00:40 – 00:47** | **Section 4: Forensic 1** | `demo_05_comparison_003103.png` | Complex Contact Hole Array Forensic (`003103.npy`) | *"On dense contact hole arrays, EDSR2x maintains structural alignment without introducing artificial artifacts or over-smoothing."* |
| **00:47 – 00:54** | **Section 4: Forensic 2** | `demo_06_comparison_002728.png` | Extreme High-Freq Grating Forensic (`002728.npy`) | *"Even under extreme high-frequency wafer grating details, the model accurately recovers line edge profiles."* |
| **00:54 – 01:02** | **Section 5: Metrics** | `demo_07_metrics.png` | Quantitative Benchmark Table (Validation: 27.42 dB / Test: 27.93 dB) | *"On our 320 validation samples, EDSR2x achieved 27.42 dB PSNR and 0.7357 SSIM. On held-out internal test samples, performance scaled consistently to 27.93 dB PSNR."* |
| **01:02 – 01:09** | **Section 6: Efficiency** | `demo_08_efficiency.png` | Hardware Audit (2.95 ms/image, 88.1 FPS, 111.6 MB VRAM) | *"Containing just 0.78 million parameters, EDSR2x processes images in 2.95 milliseconds on an RTX 3050 GPU, using under 112 megabytes of VRAM."* |
| **01:09 – 01:17** | **Section 7: Research** | `demo_09_experiment_comparison.png` | Controlled Strategy Matrix (EDSR2x vs RCAN vs SwinIR vs HAT) | *"Our systematic evaluation proved that EDSR2x outperforms complex window-attention and multi-stage models on this dataset."* |
| **01:17 – 01:23** | **Section 8: Summary** | `demo_10_final.png` | "Lightweight. Accurate. Reproducible. — EDSR2x + L1" | *"Our experiments confirm that EDSR2x provides the strongest overall balance of accuracy, efficiency, and reproducibility for semiconductor image restoration."* |
