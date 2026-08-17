# KLA Track 2 — AI-Based SEM Image Restoration (Submission Package)

This package contains the self-contained offline AI restoration pipeline for Critical Dimension Scanning Electron Microscopy (CD-SEM) images.

## Architecture & Model
- **Model**: EDSR-Light CNN with Global Residual Learning
- **Scale Factor**: 4× Super-Resolution (e.g., $64 \times 64 \rightarrow 256 \times 256$)
- **Parameters**: 360,736 parameters (optimized for sub-second inference)
- **Offline Operation**: 100% self-contained. Requires zero internet connectivity, zero external API keys, and zero runtime downloads.

## Requirements
- Python 3.8+
- PyTorch 2.0+ (CUDA GPU acceleration recommended, CPU supported as fallback)
- NumPy 1.21+

## Installation

```bash
pip install -r requirements.txt
```

## Input Format Assumptions

The offline evaluator entry point `run.py` accepts input directories containing NumPy `.npy` arrays with the following properties:
- **File Type**: `.npy` binary arrays
- **Shape**: Grayscale 2D `(H, W)` or 3D `(H, W, 1)`
- **Data Types**: `float32`, `float64`, `uint8`, or `uint16`
- **Dynamic Range**: Scaled automatically to $[0.0, 1.0]$ standard floating-point representation

## Execution Command

Run the inference script by providing input and output directories:

```bash
python run.py <input-dir> <output-dir>
```

### Example

```bash
python run.py ./sample_inputs ./restored_outputs
```

## Output Specification

For every input `<filename>.npy` in `<input-dir>`, `run.py` generates exactly one corresponding `<filename>.npy` in `<output-dir>`:
- **Filename**: Identical to input filename
- **Spatial Resolution**: $4\times$ spatial upscale ($(H \times 4, W \times 4)$ or $(H \times 4, W \times 4, 1)$ matching input dimensions)
- **Data Type**: `float32`
- **Dynamic Range**: Strictly bounded in $[0.0, 1.0]$
- **Numerical Safety**: Cleaned of any `NaN` or `Inf` values via `np.nan_to_num`

## Offline Guarantee

This submission is guaranteed to operate completely offline:
- No `requests`, `urllib`, or web requests during inference
- No Hugging Face or PyTorch Hub automatic downloads
- No Kaggle dataset fetches
- No interactive prompts or manual configuration required

## Troubleshooting

- **CUDA Out of Memory**: The model automatically performs single-pass inference for images up to $512 \times 512$. For larger images, it uses an overlapping patch tiling mechanism to fit within GPU VRAM.
- **CPU Mode**: If no NVIDIA GPU is detected, PyTorch will automatically fall back to CPU execution without error.
