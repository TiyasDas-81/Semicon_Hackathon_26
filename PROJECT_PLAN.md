# PROJECT PLAN: AI-Based Restoration of Degraded Semiconductor Images

This document outlines the project goals, architecture, design decisions, and verification strategies implemented for Track 2 – KLA of the Semicon India Hackathon.

## 1. Project Goal

The objective is to develop deep learning models capable of restoring degraded, low-resolution semiconductor images (such as Critical Dimension Scanning Electron Microscopy - CD-SEM - images) to full spatial resolution, recovering fine structures, device patterns, and defect boundaries while strictly avoiding hallucinating non-existent structures.

## 2. Core Architecture

The system operates on an end-to-end vision pipeline:

```
                  Degraded Semiconductor Image
                               |
                               v
                     Image Preprocessing
                               |
                               v
                   Inference Patch Extraction
                               |
                               v
                     AI Restoration Model
                    /                  \
                   /                    \
        EDSR CNN Baseline         SwinIR-Light Transformer
                   \                    /
                    \                  /
                     Blending & Stitching
                               |
                               v
                    Reconstruction Analytics
                   /           |            \
                  /            |             \
            Deviation Map  Confidence Map  Quality Metrics
                  \            |            /
                   \           |           /
                      Final Restored Image
```

### Components

1. **Procedural Pattern Synthesizer**: Generates realistic semiconductor geometries (gratings, contact holes, logic layers) with line-edge roughness (LER) and random defects (shorts, breaks, missing vias, particle contaminations) as high-resolution (HR) ground-truth.
2. **SEM Degradation Pipeline**: Applies anisotropic Gaussian blur (astigmatism), downsampling (4x), Poisson shot noise (electron density), Gaussian sensor noise, contrast degradation, and JPEG compression to produce low-resolution (LR) input images.
3. **Restoration Models**:
   - *Bicubic Interpolation*: Baseline reference.
   - *EDSR-Light (CNN)*: A lightweight residual CNN optimized for fast, block-based super-resolution without Batch Normalization.
   - *SwinIR-Light (Transformer)*: Our main model, using window-based multi-head self-attention (W-MSA) for local spatial modeling and a depth-wise convolutional feed-forward network (DW-FFN) for cross-window interaction.
4. **Metrology Loss Function**: Combines $L_1$ pixel loss, Structural Similarity (SSIM) loss, and Sobel edge-preservation loss to ensure pattern fidelity.
5. **Inference Stitcher**: Handles arbitrary size image inputs via overlapping tiling and linear window blending to prevent block seam artifacts.
6. **Reconstruction Telemetry**: Calculates confidence maps using Cycle Consistency Error ($| \text{LR} - \text{Downscale}(\text{Restored}) |$) to highlight uncertain regions requiring human review.

## 3. Environment & Memory Constraints

- Target hardware: Consumer NVIDIA RTX 3050 Laptop GPU (4GB VRAM).
- Memory optimization features:
  - Small patch training (64x64 LR / 256x256 HR).
  - FP16 Automatic Mixed Precision (AMP) training.
  - Omission of high-memory Batch Normalization layers in the CNN.
  - Linear-time window-based attention in the Transformer.
