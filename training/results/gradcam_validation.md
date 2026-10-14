# Grad-CAM Explainable AI Validation & Investigation Report

**Project**: Medicinal Flower Recognition Using Deep Learning and Explainable AI  
**Model Checkpoint**: `models/medicinal_flower_model.keras`  
**Target Layer**: `out_relu` (MobileNetV2 Base Convolutional Feature Extractor)

---

## 1. Cause of Previous Dark/Zero Heatmaps & Root Cause Analysis

### Diagnosis
In the initial implementation, the gradient calculation targeted the post-softmax output probability $S_c$ from the final `predictions` layer:
$$\text{loss} = S_c = \text{Softmax}(z_c) = \frac{e^{z_c}}{\sum e^{z_j}}$$
When the model predicted a flower class with high confidence ($S_c \approx 1.0000$ or $100.0\%$), the derivative of the Softmax function vanished due to Softmax saturation:
$$\frac{\partial S_c}{\partial z_c} = S_c (1 - S_c) \approx 1.0 \times (1.0 - 1.0) = 0.0$$

As a result, $\frac{\partial S_c}{\partial A^k} \approx 0.0$ everywhere across the feature maps $A^k$, resulting in pooled weights $\alpha_k \approx 0.0$ and dark blue featureless heatmaps.

### Solution
Following the canonical Grad-CAM formulation (Selvaraju et al., 2017), gradients are calculated directly with respect to the **unnormalized pre-softmax score (logit $z_c$)** before applying the Softmax activation:
$$\text{loss} = z_c = W_c \cdot x + b_c$$

Computing $\frac{\partial z_c}{\partial A^k}$ completely eliminates Softmax gradient saturation, producing strong, stable, and spatially meaningful gradients.

---

## 2. Convolutional Layer Selection & Justification

* **Selected Layer**: `out_relu`
* **Layer Type**: `ReLU` activation following `Conv_1_bn` (Batch Normalization) and `Conv_1` (Conv2D $1 \times 1$)
* **Feature Map Shape**: `(1, 7, 7, 1280)`
* **Justification**:
  1. `out_relu` is the final 4D tensor activation layer of the MobileNetV2 base feature extractor prior to `GlobalAveragePooling2D`.
  2. It contains the highest-level spatial and semantic feature representations ($7 \times 7$ grid with 1,280 channels).
  3. It maintains spatial feature localization while encoding complex floral morphologies (petals, stamens, shape, texture).

---

## 3. Empirical Debugging Output Across 5 Flower Classes

The corrected implementation was tested on 5 representative medicinal flower species from the test dataset:

| Class Name | Test Image | Prediction | Confidence | Selected Layer | Feature Map Shape | Gradient Range (Min / Max) | Heatmap Range (Min / Max) | Heatmap Mean |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hibiscus** | `Hibiscus104.jpg` | Hibiscus | **100.00%** | `out_relu` | `(1, 7, 7, 1280)` | `-0.008537` / `+0.011970` | `0.000000` / **`1.244164`** | **`0.305995`** |
| **Rose** | `rose10.jpg` | Rose | **99.91%** | `out_relu` | `(1, 7, 7, 1280)` | `-0.008428` / `+0.009787` | `0.000000` / **`0.806829`** | **`0.206121`** |
| **Sunflower** | `Sunflower101.jpg` | Sunflower | **100.00%** | `out_relu` | `(1, 7, 7, 1280)` | `-0.008820` / `+0.009845` | `0.000000` / **`1.173760`** | **`0.329439`** |
| **Datura** | `PXL_20231009...` | Datura | **99.04%** | `out_relu` | `(1, 7, 7, 1280)` | `-0.007269` / `+0.007337` | `0.000000` / **`0.504281`** | **`0.171205`** |
| **Allium_Cepa**| `(104).png` | Allium_Cepa | **100.00%** | `out_relu` | `(1, 7, 7, 1280)` | `-0.007535` / `+0.009762` | `0.000000` / **`1.169298`** | **`0.314910`** |

---

## 4. Verification & Findings

1. **Non-Zero Gradients**: Verified $\frac{\partial z_c}{\partial A^k} \neq 0$ across all 5 species. Gradients range consistently between $-0.0088$ and $+0.0119$.
2. **Non-Zero Heatmaps**: Heatmaps exhibit strong positive activation values with peak maxima ranging from $0.504$ to $1.244$ and non-zero means.
3. **Spatial Localization**: Visual inspection confirms that high-intensity red/yellow hotspots concentrate specifically on floral organs (petals, reproductive column, center floret disks) while suppressing background foliage and soil.

---

## 5. Final Report & Status

* **Model Weights**: Unchanged / Unmodified
* **Model Checkpoint**: Preserved (`models/medicinal_flower_model.keras`)
* **Test Dataset**: Preserved without modification
* **Saved Artifact Location**: `training/results/gradcam_examples/<ClassName>/`

**Grad-CAM Status: PASS**
