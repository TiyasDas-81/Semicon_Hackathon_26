# Semiconductor and Microscopy SEM Restoration Dataset Registry

This document catalogs the verified SEM datasets integrated into this project and candidates identified for future domain expansion.

---

## 1. Carinthia SEM Defect Dataset (Primary)

*   **Official Source**: [Zenodo Record #10715190](https://zenodo.org/records/10715190)
*   **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
*   **Dataset Type**: Real production-layer SEM images of defect wafers.
*   **Image Count**: 4,591 images.
*   **Resolution**: 480x480 pixels (8-bit grayscale).
*   **Degradation Characteristics**: Original images represent high-quality production wafer scans. Low-resolution/degraded inputs are synthesized via our multi-level degradation pipeline.
*   **Paired/Unpaired Status**: Paired (synthetic degradation mapped to clean wafer reference images).
*   **Training Suitability**: Excellent. Used as the primary wafer domain dataset for training EDSR-Light and SwinIR-Light models.
*   **Testing Suitability**: Excellent. Used as the main validation benchmark dataset.
*   **Limitations**: Original wafer images represent structured layouts but do not contain physical acquisition LR-HR pairs from the microscope detector.

---

## 2. MIIC — Microscopic Images of Integrated Circuits (Secondary)

*   **Official Source**: [GitHub: wenbihan/MIIC-IAD](https://github.com/wenbihan/MIIC-IAD)
*   **License**: Academic Non-Commercial Use
*   **Dataset Type**: Real SEM images of manufactured integrated circuits (IC) metal layers.
*   **Image Count**: 1,890 images.
*   **Resolution**: 512x512 pixels (8-bit grayscale).
*   **Degradation Characteristics**: Clean metal layer scans used as high-quality targets. Controlled synthetic degradations are applied programmatically.
*   **Paired/Unpaired Status**: Paired (synthetic degradation mapped to high-quality source images).
*   **Training Suitability**: High. Provides semiconductor domain diversity (metal interconnect structures).
*   **Testing Suitability**: High. Evaluates model generalizability to metal line-spaces and interconnect vias.
*   **Limitations**: No native low-resolution acquisition pairs are provided.

---

## 3. NIST — Detection Limits for SEM Image Segmentation (Stress Control)

*   **Official Source**: [NIST PDR (mds2-3838)](https://doi.org/10.18434/mds2-3838)
*   **Official Citation**: Bajcsy, Peter, Sathe, Pushkar, Vladar, Andras (2025), Detection Limits for SEM Image Segmentation, National Institute of Standards and Technology, DOI: 10.18434/mds2-3838.
*   **License**: Public Domain (U.S. Government Work)
*   **Dataset Type**: Simulated SEM images with controlled contrast and Poisson shot noise grids.
*   **Image Count**: 3,402 degraded images (6 sets of 567 images mapping a 27x21 parameter grid).
*   **Resolution**: 512x512 pixels (8-bit grayscale).
*   **Degradation Characteristics**: Programmatic Poisson shot noise and contrast levels simulated using ARTIMAGEN software.
*   **Paired/Unpaired Status**:
    *   *Paired Mode*: Degraded images are mapped to their noise-free, high-contrast references (`masks/set*_cex_noise_000_contrast_100.tiff`) extracted from the official `mask_sets.zip` archive.
    *   *Blind Mode*: Tested without loading references to run cycle-consistency and risk metrics.
*   **Training Suitability**: Moderate. The simulated grid is optimized for segmentation and detection limits, but useful for domain generalization testing.
*   **Testing Suitability**: High. Serves as a controlled stress test to profile model behavior under severe signal-to-noise ratios.
*   **Limitations**: Geometries are simulated patterns (e.g. lines, contact holes) rather than real wafer acquisitions.

---

## 4. WM-811K Wafer Map Dataset (Auxiliary)

*   **Official Source**: [MIR Lab Wafer Map Dataset](http://mirlab.org/dataset/public/)
*   **License**: Open Academic Research Use
*   **Dataset Type**: Wafer bin maps showing spatial defect patterns (scratch, ring, loc, etc.).
*   **Image Count**: 811,457 maps.
*   **Resolution**: Varied low-resolution maps representing full wafers.
*   **Degradation Characteristics**: Low-resolution bin classification maps.
*   **Paired/Unpaired Status**: Unpaired (semantic bin maps, not SEM micrographs).
*   **Training Suitability**: Unsuitable for super-resolution training.
*   **Testing Suitability**: Unsuitable.
*   **Limitations**: This is a macro wafer defect classification dataset, not high-resolution SEM microscopic imagery.

---

## 5. Candidate Search Results (Future Expansion)

### A. SEMICON-SR-2X-MEMS-v1 (Kaggle)
*   **Official Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/qingyi/semicon-sr-2x-mems-v1) (Community uploaded by "qingyi")
*   **License**: Community Open Source
*   **Dataset Type**: Synthetic benchmark containing paired MEMS semiconductor structures.
*   **Image Count**: Varies (approximately 500 images).
*   **Resolution**: 512x512 pixels.
*   **Paired/Unpaired Status**: Paired (designed for 2x super-resolution training).
*   **Training Suitability**: High. Very relevant to semiconductor structure super-resolution.
*   **Testing Suitability**: High. Useful for generalizability checks on MEMS layout geometries.

### B. Resolution Enhancement of SEM using AI (Zenodo)
*   **Official Source**: [Zenodo Record #11224127](https://doi.org/10.5281/zenodo.11224127)
*   **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
*   **Dataset Type**: Micrographs of dual-phase steel.
*   **Image Count**: Triplet sets (low-res, high-res, and reference images).
*   **Resolution**: Varied.
*   **Paired/Unpaired Status**: Paired.
*   **Training/Testing Suitability**: Useful for testing models on materials SEM domains (metals/grain structures) rather than semiconductor layout geometries.
