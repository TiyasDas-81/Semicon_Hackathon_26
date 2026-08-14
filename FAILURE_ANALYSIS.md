# FAILURE ANALYSIS & LIMITATIONS: AI-Based Semiconductor Image Restoration

This document provides a systematic analysis of potential failure modes and reconstruction limitations for our Single-Image Super-Resolution (SISR) models under various SEM operating conditions.

## 1. Identified Failure Modes

Through physical modeling and experimentation, we identify three critical degradation thresholds where the restoration models may fail or perform sub-optimally:

### A. High Poisson Shot Noise (Extreme Low-Dose Ingestion)
- **Description**: Scanning Electron Microscopes (SEM) operating at extremely low currents (low electron dose to avoid wafer charge-up and damage) produce images with highly dominant Poisson shot noise.
- **AI Behavior**: When the signal-to-noise ratio (SNR) is extremely low, the model struggles to distinguish tiny defect boundaries (e.g., small particles < 5nm) from noise fluctuations. It tends to smooth out these features or blend them into the substrate, failing to restore them.
- **Mitigation**: The *Cycle-Consistency Confidence Map* flags these regions. A high discrepancy between the downscaled restored output and the noisy input indicates that the model smoothed out high-frequency fluctuations, warning engineers of potential missing defects.

### B. Severe Astigmatism (Beam Asymmetry)
- **Description**: If the electron beam focus is asymmetric ($\sigma_x \gg \sigma_y$ or vice-versa), vertical patterns will blur horizontally, causing adjacent lines to appear connected.
- **AI Behavior**: The model may misinterpret severe directional defocus blur as physical line-bridging (a short defect) and reconstruct a false structural connection between gates.
- **Mitigation**: The *Deviation Map* will show extremely large local modifications in the direction of the astigmatism blur. Engineers can inspect the telemetry and reject the scan, prompting beam recalibration.

### C. Extreme Resolution Aliasing (High Downsampling Scale > 4x)
- **Description**: When the input image spatial resolution is downsampled beyond the Nyquist limit for the pattern pitch, aliasing (Moire patterns) occurs.
- **AI Behavior**: The model may reconstruct lines that have incorrect pitches or phases, shifting the layout lines by a few pixels. This represents a form of structural hallucination.
- **Mitigation**: We enforce a strict scale limit of 4x. For higher scaling ratios, multi-frame or prior-guided restoration is recommended.

## 2. Telemetry and Risk Mitigation

To prevent AI hallucination from causing incorrect metrology or inspection results, our dashboard introduces a **Dual-Map Quality Check**:

1. **Deviation Map ($| \text{Transformer} - \text{Bicubic} |$)**: Tells the inspector exactly *where* and *how much* high-frequency detail the AI model added. High deviation highlights thin line-edges and corners where features are sharpened.
2. **Confidence Map ($1.0 - \text{Consistency Error}$)**: Calculates the mathematical agreement of the restoration with the source input. If a restored pattern cannot be degraded back to match the original LR image, the confidence score drops.

## 3. Empirical Warning Thresholds from Validation Experiments

To establish practical warning boundaries, we evaluated our trained Bicubic, CNN (EDSR-Light), and Transformer (SwinIR-Light) models across 4 controlled degradation levels on the official **Carinthia SEM Defect Dataset** (containing 4,591 real production-layer SEM images).

### Experimental Results Summary (Carinthia SEM Dataset)

| Model | Degradation Level | PSNR (dB) | SSIM | Edge preservation | Mean Consistency Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bicubic** | Level 1 (Mild) | 28.79 | 0.5624 | 0.8543 | 0.9581 |
| | Level 2 (Moderate) | 24.84 | 0.3780 | 0.8458 | 0.9400 |
| | Level 3 (Severe) | 21.91 | 0.2601 | 0.8430 | 0.9213 |
| | Level 4 (Extreme) | 18.05 | 0.1399 | 0.8432 | 0.8804 |
| **CNN** | Level 1 (Mild) | 23.72 | 0.7752 | 0.9277 | 0.7610 |
| | Level 2 (Moderate) | 23.46 | 0.7011 | 0.9269 | 0.7219 |
| | Level 3 (Severe) | 23.04 | 0.6090 | 0.9209 | 0.6709 |
| | Level 4 (Extreme) | 21.90 | 0.4449 | 0.9015 | 0.5618 |
| **Transformer** | Level 1 (Mild) | 22.06 | 0.7301 | 0.9277 | 0.6977 |
| | Level 2 (Moderate) | 21.93 | 0.6444 | 0.9206 | 0.6788 |
| | Level 3 (Severe) | 21.71 | 0.5490 | 0.9089 | 0.6436 |
| | Level 4 (Extreme) | 21.09 | 0.3891 | 0.8803 | 0.5577 |

### Analysis and Warning Boundaries Selection

- **Analysis**: As degradation increases from Mild to Extreme, the structural similarity (SSIM) of both models degrades substantially (e.g. SwinIR-Light drops from `0.7301` to `0.3891`). The Cycle-Consistency Confidence Map mean value drops correspondingly from `0.6977` to `0.5577`.
- **Threshold Selection**: Based on these experiments, we select the warning thresholds at the boundary of **Level 3 (Severe)**, below which structural hallucinations and pattern distortions become critical:
  - **Empirical Warning Confidence Threshold**: `0.6236` (derived from Level 3 Transformer mean confidence `0.6436` minus `0.02` safety margin).
  - **Empirical Warning PSNR Threshold**: `21.21 dB`.
  - **Status Designation**: A restoration output with a mean consistency score `< 0.6236` will trigger `SYS_STATUS: ALARM` in the telemetry console, prompting manual verification.

> [!NOTE]
> *Thresholds selected empirically from validation experiments.*
