# Calibration Methodology — Restoration Confidence and Potential Risk Score

This document outlines the mathematical formulations and calibration procedures used for our **Restoration Confidence / Inspection Risk Indicator (RC-IRI)** and the **Potential Hallucination Risk Map**.

## 1. Spatial-Dimension Invariance

In Phase 1, the raw pixel difference was prone to scale mismatch because the input $Y$ is lower-resolution while the restored output $F(Y)$ is super-resolved. 

To resolve this, all comparisons are aligned mathematically at their respective native resolutions:

1. **AI Modification** ($E_{mod}$): Computed at **HR resolution** to measure the high-frequency structural updates added by the neural network relative to the standard upscaling baseline:
   $$E_{mod}(x_{HR}, y_{HR}) = | F(Y)(x_{HR}, y_{HR}) - \text{Bicubic}(Y)(x_{HR}, y_{HR}) |$$

2. **Consistency Error** ($E_{const}$): Computed at **LR resolution** to verify the model's fidelity to the raw detector measurements. The restored image is downscaled back to LR using an Area-resize operator $D(\cdot)$ and compared directly to the input $Y$:
   $$E_{const}(x_{LR}, y_{LR}) = | Y(x_{LR}, y_{LR}) - D(F(Y))(x_{LR}, y_{LR}) |$$

---

## 2. Normalization & Calibration Logic

To prevent reporting raw, uncalibrated pixel values, each component is normalized against average statistics ($\mu_{mod}, \mu_{const}$) derived from a representative validation set of real SEM images:

- **Normalized AI Modification** ($\overline{E}_{mod}$):
  $$\overline{E}_{mod}(x_{HR}, y_{HR}) = \frac{E_{mod}(x_{HR}, y_{HR})}{\mu_{mod}}$$

- **Normalized Consistency Error** ($\overline{E}_{const}$):
  $$\overline{E}_{const}(x_{LR}, y_{LR}) = \frac{E_{const}(x_{LR}, y_{LR})}{\mu_{const}}$$

---

## 3. Potential Hallucination Risk Map

A region is flagged as having a **Potential Hallucination Risk** if the AI model makes substantial structural modifications to the pattern, yet the resulting structures have poor consistency (i.e. they fail to re-degrade back to the input measurements):

$$\text{PotentialRisk}(x_{HR}, y_{HR}) = \overline{E}_{mod}(x_{HR}, y_{HR}) \times \overline{E}_{const\_HR}(x_{HR}, y_{HR})$$

where $\overline{E}_{const\_HR}$ is the consistency error map upsampled to HR resolution to match dimensions.

### Terminology Interpretation
> [!NOTE]
> High-risk regions indicate areas where the restoration differs substantially from the interpolation baseline and has weak reconstruction consistency. These regions require human inspection. This does not prove structural hallucination, but serves as a statistical trigger for manual verification.

---

## 4. Restoration Confidence / Inspection Risk Indicator (RC-IRI)

The global confidence score is computed as:
$$\text{RC-IRI} = 1.0 - \text{Clip}\left(\frac{\text{Mean}(E_{const})}{\mu_{const}}, 0.0, 1.0\right)$$

This metric is bounded between `[0, 1]` and indicates the overall reconstruction quality normalized by our validation dataset properties.
