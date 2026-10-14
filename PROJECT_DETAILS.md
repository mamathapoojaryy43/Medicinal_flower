# 📖 Project Technical Specification & Architecture Details

## 🎯 Executive Summary
The **Medicinal Flower Recognition & Explainable AI System** is a production-grade machine learning application designed to classify 31 distinct Indian medicinal flower species from standard RGB photographs. Built upon **MobileNetV2 Transfer Learning**, the pipeline incorporates an out-of-distribution confidence thresholding mechanism (0.60 threshold), Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explainability, and a lightweight Flask web platform backed by a rich botanical metadata engine.

---

## 🏗️ Technical Architecture & Pipeline

### 1. Computer Vision & Neural Network Architecture
- **Base Back-bone**: MobileNetV2 pre-trained on ImageNet.
- **Input Resolution**: $224 \times 224 \times 3$ RGB tensor.
- **Feature Extractor Layer**: Layer `"out_relu"` ($7 \times 7 \times 1280$ feature map tensor).
- **Classification Head**:
  - Global Average Pooling 2D (`GlobalAveragePooling2D`)
  - Batch Normalization (`BatchNormalization`)
  - Dropout ($p=0.3$, `dropout_1`)
  - Dense Layer ($256$ units, ReLU activation, `dense_256`)
  - Dropout ($p=0.3$, `dropout_2`)
  - Output Dense Layer ($31$ units, Softmax activation, `predictions`)
- **Model Checkpoint Size**: $\approx 25.95 \text{ MB}$ (`models/medicinal_flower_model.keras`).

---

## 🛡️ Out-Of-Distribution (OOD) Confidence Thresholding

To prevent false positive classifications on non-flower images or unsupported species, the system implements max-probability thresholding:

$$\hat{y} = \begin{cases} \arg\max_{c} P(y=c \mid X) & \text{if } \max_{c} P(y=c \mid X) \ge \tau \\ \text{Unrecognized / Unsupported} & \text{if } \max_{c} P(y=c \mid X) < \tau \end{cases}$$

- **Selected Threshold ($\tau$)**: `0.60`
- **Validation Acceptance Rate**: `99.72%`
- **Validation Accepted Accuracy**: `99.86%`
- **Config file**: `models/prediction_config.json`

---

## 🔍 Explainable AI (Grad-CAM) Implementation

Grad-CAM computes visual attribution heatmaps by calculating gradients of the target class logit score $y^c$ (pre-softmax) with respect to feature activation maps $A^k$ of the final convolutional layer (`out_relu`):

1. **Neuron Importance Weights ($\alpha_k^c$)**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Coarse Heatmap ($L_{\text{Grad-CAM}}^c$)**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
3. **Overlay Generation**: The $7 \times 7$ heatmap is bicubically upsampled to $224 \times 224$, colormapped using Matplotlib `plt.cm.jet`, and alpha-blended with the original normalized image ($60\%$ image $+ 40\%$ heatmap overlay).

---

## 📊 Comprehensive Test Set Evaluation (1,456 Test Images)

Evaluation conducted on test dataset yielded the following statistics:

```json
{
  "test_accuracy": 0.9897,
  "test_loss": 0.0248,
  "macro_precision": 0.9866,
  "macro_recall": 0.9847,
  "macro_f1_score": 0.9854,
  "weighted_precision": 0.9900,
  "weighted_recall": 0.9897,
  "weighted_f1_score": 0.9897
}
```

### Top 5 Perfect Performance Classes (F1-Score = 1.00)
1. **Allium Cepa** (129 samples)
2. **Black Cumin** (189 samples)
3. **Champaka** (35 samples)
4. **Common Lanthana** (35 samples)
5. **Datura** (36 samples)

---

## 🌐 Flask Web Application Details (`app/`)

### Architecture
- **Framework**: Flask (Python WSGI Web Framework)
- **State Management**: Models and JSON databases loaded ONCE into memory at app initialization in `load_resources()`.
- **Thread Safety**: Uses Matplotlib `'Agg'` non-interactive backend for rendering heatmaps across concurrent requests.

### Core API Endpoints
- `GET /`: Serves web dashboard (`templates/index.html`).
- `POST /predict`: Accepts form file upload `image`. Returns JSON payload:
  - `status`: `"success"` or `"unrecognized"`
  - `predicted_class`: Class label string
  - `confidence`: Confidence percentage ($0-100\%$)
  - `gradcam_url`: Path to generated Grad-CAM overlay image
  - `top_predictions`: Top 3 probable classes with percentages
  - `flower_info`: Medicinal properties, botanical name, active compounds, usage guidance from `flower_information.json`.
- `GET /health`: System health check endpoint.

---

## 🛠️ Python Script Utilities

- **`predict.py`**: Standard CLI utility for executing predictions with optional Grad-CAM visual output and JSON exports.
- **`training/train.py`**: Model training routine with learning rate reduction on plateau and early stopping callbacks.
- **`training/evaluate_model.py`**: Evaluates model checkpoint on test dataset and generates confusion matrix & classification report metrics.
- **`training/gradcam.py`**: Standalone class module for computing layer activation heatmaps.
- **`scripts/test_flask_app.py`**: Automated server integration test suite.
- **`scripts/verify_environment.py`**: Checks CUDA / GPU availability and version compatibility.
