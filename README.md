# AI-Based Restoration of Degraded Semiconductor Images (KLA Track 2)

This repository contains the end-to-end implementation for our **Semicon India Hackathon Project — Track 2: AI-Based Restoration of Degraded Images**. 

Our solution restores degraded, low-resolution, and noisy semiconductor (CD-SEM) wafer images to full spatial resolution using deep learning. It recovers critical gate patterns, contacts, and defects while using physical cycle-consistency mapping to identify potential hallucination regions for human inspection.

---

## 1. Problem and Architecture

### The Problem
During semiconductor manufacturing, Critical Dimension Scanning Electron Microscope (CD-SEM) systems scan wafers to verify line widths and inspect defects. To prevent wafer charging or damage, low-dose scans are used, introducing heavy Poisson shot noise and blur. Standard mathematical upscalers (Bicubic) cannot recover lost spatial details or sharpen defect boundaries.

### The Solution
We implement a deep-learning-based single-image super-resolution (SISR) pipeline optimized to run on consumer-grade hardware (4GB VRAM GPU) using:
- A compound **Edge-Preserving Metrology Loss** to encourage structural fidelity.
- **Overlap-Stitching Inference** to handle arbitrary high-resolution scans.
- **Cycle-Consistency Confidence Telemetry** to flag low-confidence zones for human inspection.

### System Architecture Diagram
```
            Semiconductor Image
                     |
                     v
            Image Preprocessing
                     |
                     v
           Degradation Analysis
                     |
                     v
          AI Restoration Model
          /                 \
         /                   \
  CNN Baseline          Transformer
         \                   /
          \                 /
           ---- Comparison
                   |
                   v
           Quality Analysis
             /     |      \
            /      |       \
         PSNR     SSIM    Edge
            \      |       /
             \     |      /
              Final Restored
                 Image
```

---

## 2. Repository Structure

```
project/
├── configs/
│   └── default.yaml          # Dataset and training hyperparameters
├── datasets/
│   └── semicon_dataset.py    # Aligned patch loader and augmentations
├── losses/
│   └── restoration_losses.py # L1 + SSIM + Sobel Edge metrology losses
├── models/
│   ├── baseline.py           # Bicubic interpolation module
│   ├── cnn.py                # EDSR-Light CNN Baseline
│   └── transformer.py        # SwinIR-Light Transformer Model (W-MSA + DW-FFN)
├── training/
│   └── trainer.py            # PyTorch AMP-accelerated training pipeline
├── evaluation/
│   └── evaluator.py          # Metric benchmarking and comparative visual generator
├── inference/
│   └── restorer.py           # Overlap patch inference & cycle-consistency maps
├── backend/
│   ├── app.py                # FastAPI web server
│   └── index.html            # Dark-mode industrial inspection console
├── scripts/
│   ├── generate_dataset.py   # Procedural wafer pattern & defect synthesizer
│   └── run_pipeline.py       # Master orchestration script
├── experiments/              # Telemetry CSVs and visualization output
├── checkpoints/              # Saved model checkpoints (CNN & Transformer)
├── PROJECT_PLAN.md           # Engineering design and roadmap
├── FAILURE_ANALYSIS.md       # Limitations and risk mitigation analysis
└── requirements.txt          # Python dependencies list
```

---

## 3. Getting Started

### Installation
Ensure Python 3.11+ is installed, then run:
```bash
# Clone the repository
git clone https://github.com/TiyasDas-81/Semicon_Hackathon_26.git
cd Semicon_Hackathon_26

# Install PyTorch with CUDA 12.1 and other requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Reproducing the Complete Pipeline
You can run the entire workflow (dataset generation -> baseline training -> transformer training -> metrics evaluation) with a single command:
```bash
python scripts/run_pipeline.py
```
*Note: The master configuration is located in `configs/default.yaml`.*

To execute individual components step-by-step:
```bash
# 1. Synthesize Semiconductor Images and Defects
python scripts/generate_dataset.py

# 2. Train the CNN Baseline Model (EDSR-Light)
python training/trainer.py --model cnn

# 3. Train the SwinIR-Light Transformer Model
python training/trainer.py --model transformer

# 4. Benchmarks and Visual Plots Generation
python evaluation/evaluator.py
```

---

## 4. Benchmark Evaluation Results

The models were evaluated across 30 unseen test scenarios. Quantitative averages are reported below:

| Method | PSNR (dB) | SSIM | Edge Score | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Bicubic (Baseline)** | 18.79 | 0.4000 | 0.8484 | **8.07 ms** |
| **EDSR-Light (CNN)** | 19.69 | **0.4793** | **0.8817** | 13.68 ms |
| **SwinIR-Light (Transformer)** | **19.71** | 0.4430 | 0.8751 | 17.08 ms |

### Visual Comparisons & Telescopic Details
For each sample, the evaluation script saves:
1. Side-by-side grids in `experiments/comparison_sample_{idx}.png` showing input, output, confidence, and deviation maps.
2. Pixel-level zooms in `experiments/zoomed_comparison_{idx}.png` highlighting line-edge sharpness.

---

## 5. Launching the Web Workstation Ingestion

Start the FastAPI backend server:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Once the server is running, open your web browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Workstation Features:
- **Preset Ingestion**: Select Gratings, Vias, or Logic layers with defects directly from the sidebar.
- **Dual Colormap Inspection**: Toggle between the Restored image, the **Confidence Map** (Jet scale representing consistency), and the **Deviation Map** (Hot scale indicating AI sharpened details).
- **Synchronized Cursor Magnifier**: Hover over any pixel coordinate on the viewports to see a real-time 4x zoom.
