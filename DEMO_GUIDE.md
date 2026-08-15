# Hackathon Demonstration Guide — Track 2: KLA Image Restoration

This guide outlines a **3-minute presentation script** to showcase the features, robustness, and industrial applicability of the AI-based Semiconductor Image Restoration platform.

---

## Preparation
1. Ensure the backend server is running:
   ```bash
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```
2. Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in the web browser.
3. Keep this guide open for narration cues.

---

## ⏱️ Minute 0:00 – 1:00 | Demo 1: Semiconductor Defect Restoration & Model Comparison
* **Goal**: Show controlled super-resolution on real wafer defect patterns and compare models.
* **Actions**:
  1. Under the sidebar panel **"Quick Demo Samples"**, click **`Carinthia Example`**.
  2. The system automatically selects the Carinthia SEM dataset, loads a wafer scan, sets the mode to **Synthetic Benchmark**, and runs restoration.
  3. **Visual Ingestion**: Show the **Original Ingest** (degraded) and **Output Restored** viewports side-by-side.
  4. **Model Comparison**: Switch the model selector to **EDSR-Light CNN** and re-run. Then switch to **Bicubic** and re-run. Compare the quantitative metrics across all three models using the same input image.
  5. **Cursor Magnification**: Hover the mouse over the defect region. Show the **Detail Magnifier (4x Zoom)** showing the comparison between the pixelated input crop and the restored output.
* **Narration Cue**:
  > *"Here, we see a real wafer defect scan from the production-layer Carinthia dataset. CD-SEM scanners operate at low current dosages to protect wafers, yielding highly pixelated images. Standard Bicubic interpolation smooths defect edges. Our deep learning models recover 4x spatial resolution. The restoration pipeline uses reconstruction consistency and AI modification maps to identify regions requiring human inspection, giving engineers actionable quality indicators alongside restored images."*

---

## ⏱️ Minute 1:00 – 2:00 | Demo 2: Blind SEM Stress Test & Calibrated Risk Map
* **Goal**: Show blind evaluation on extreme Poisson noise with no ground truth.
* **Actions**:
  1. Under the sidebar panel **"Quick Demo Samples"**, click **`NIST High Noise`**.
  2. The system automatically selects the **NIST SEM Contrast/Noise Stress Test**, loads a highly degraded image (`set1_cex_noise_915_contrast_001.tiff`), sets the mode to **Real SEM Mode (Blind)**, and runs restoration.
  3. **Suppress Metrics Check**: Note that PSNR, SSIM, and Edge Score are locked to `N/A - Real Scan` because no Ground Truth exists.
  4. **Active Alarm Warning**: Show the status bar indicating `SYS_STATUS: ALARM (LOW CONFIDENCE)`.
  5. **Inspect maps**: Click the **Potential Risk Map** tab on the Output viewport. The dashboard displays the `magma`-colored potential risk heatmap. Click the **AI Modification Map** tab to see where the model diverges from bicubic baseline.
* **Narration Cue**:
  > *"When dealing with real scans, no ground truth exists. Our system switches to Blind Mode, suppressing PSNR/SSIM scores since they cannot be computed without a reference. Instead, we provide a Calibrated Telemetry warning panel. By evaluating cycle-consistency at LR and AI modification at HR, we generate a Potential Hallucination Risk Map. High-risk regions represent areas where the model made substantial modifications with weak reconstruction consistency — these regions require human inspection. If the global confidence drops below the calibrated threshold, the system triggers a warning alarm, flagging this wafer scan for manual verification by a metrology engineer."*

---

## ⏱️ Minute 2:00 – 3:00 | Demo 3: Quantitative Metrology Benchmark
* **Goal**: Demonstrate model performance with empirical benchmark data.
* **Actions**:
  1. Point the judges to the Metrology Metrics scoreboard and reference the benchmark data in [`experiments/model_comparison.csv`](experiments/model_comparison.csv) and [`experiments/model_comparison.png`](experiments/model_comparison.png).
  2. Present the measured results (20 Carinthia SEM images, 4x super-resolution):

     | Model | Mean PSNR | Mean SSIM | Edge Score | Inference Time |
     |-------|----------|----------|-----------|---------------|
     | Bicubic (Baseline) | 24.73 dB | 0.3694 | 0.8394 | 1.3 ms |
     | EDSR-Light CNN | 23.97 dB | 0.7013 | 0.9231 | 3.4 ms |
     | SwinIR-Light Transformer | 22.35 dB | 0.6444 | 0.9172 | 11.3 ms |

  3. Explain that deep learning models nearly double structural similarity (SSIM from 0.37 to 0.70) and significantly improve edge preservation (from 0.84 to 0.92), while maintaining fast inference times under 12 ms.
  4. **Model selection**: EDSR-Light CNN was selected as the recommended production model based on the quality–latency trade-off: highest SSIM, best edge preservation, and 3x faster than SwinIR-Light. SwinIR-Light remains available as an experimental comparison.
* **Narration Cue**:
  > *"Model selection was based on empirical quality–latency trade-off across 20 Carinthia SEM images. EDSR-Light CNN provides the highest SSIM of 0.70 and the best edge preservation of 0.92, while running at 3.4 ms — making it suitable for integration into live production lines. All three models are available in the console for A/B comparison."*

---

## Notes
- PSNR is higher for Bicubic because it minimizes pixel-level difference by blurring, while DL models restructure edges (increasing SSIM and edge fidelity at the cost of pixel-exact PSNR).
- The Potential Risk Map does **not** prove hallucination occurred. It highlights regions where substantial AI modification coincides with weak reconstruction consistency, serving as a statistical trigger for manual verification.
- All metrics shown in this demo are computed from actual model inference on real public dataset images. No metrics are fabricated or pre-computed.
