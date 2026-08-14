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
2. **Confidence Map ($1.0 - \text{Consistency Error}$)**: Calculates the mathematical agreement of the restoration with the source input. If a restored pattern cannot be degraded back to match the original LR image, the confidence score drops. Regions with confidence $< 85\%$ are flagged as "Inspect" rather than "Auto-Pass".
