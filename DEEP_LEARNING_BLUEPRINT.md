# ORIEN Neural Synergy: Deep Learning R&D Blueprint (v2.0)

## 🎯 Mastery Vision
To achieve a **98% Mastery State** by fusing **Appearance Features** (CNN) with **Geometric Heatmaps** (Landmarks) and applying **Temporal Smoothing** for real-time stability.

---

## 🔬 1. Deep Research: SOTA Architectures (2024-2025)
Based on recent deep research into State-of-the-Art (SOTA) Facial Emotion Recognition (FER), we are implementing a **Multi-Modal Fusion Framework**.

### 1.1 Spatial Attention via Landmark Heatmaps
*   **Research Insight**: Simple coordinate injection lacks spatial context. SOTA models use landmark heatmaps to guide the CNN's attention.
*   **Implementation**: A secondary input stream will generate Gaussian heatmaps around 68 key facial landmarks. These heatmaps will be multiplied with the CNN's intermediate feature maps to focus the model on high-entropy regions like the mouth and brow.

### 1.2 Meta-Classifier Ensemble (Post-Softmax)
*   **Research Insight**: SoftMax can be overconfident or biased. Replacing/augmenting it with **Support Vector Machines (SVM)** or **XGBoost** on the final latent vector (256-D) improves robustness.
*   **Implementation**: The 256-D bottleneck vector will be exported to a secondary Meta-Classifier trained specifically on hard-to-distinguish classes (e.g., Fear vs. Surprise).

### 1.3 Temporal Sliding Window (HUD Stability)
*   **Research Insight**: Single-frame inference is prone to "jitter."
*   **Implementation**: A **Moving Average Filter** or a **1D-CNN temporal head** will process the last 5-10 frames to produce a smoothed, high-fidelity prediction in the Real-time HUD.

---

## 🛠️ 2. Updated Master Pipeline
### 2.1 Ultra-Fidelity Preprocessing
*   **Resolution**: 224x224px.
*   **Landmark Alignment**: Faces are not just cropped, but **aligned** based on the horizontal eye-line to reduce variance in head tilt.

### 2.2 Dual-Stream Training
1.  **Stream A (Pixels)**: EfficientNetB0 fine-tuned on pixel data.
2.  **Stream B (Geometry)**: Landmark Heatmap generation via MediaPipe.
3.  **Fusion**: Concatenation of Global Average Pooled features with Geometric latent vectors.

---

## 🔭 Telescope: The Path to 98.5% Accuracy
*   **Step 1**: Hyper-parameter optimization (current phase).
*   **Step 2**: **Landmark Heatmap Integration** (Mastery Phase 2).
*   **Step 3**: **Ensemble Meta-Classifier** (Mastery Phase 3).
*   **Step 4**: **Edge-Quantization (Int8)** with QAT for zero-latency deployment.

---
*Blueprint Version: 2.0 | Updated: 2026-04-22 | Status: SOTA Research Integrated*
