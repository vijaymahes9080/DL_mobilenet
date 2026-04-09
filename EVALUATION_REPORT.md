# 🏁 EVALUATION REPORT: ORIEN NEURAL SYSTEM — Phase 2

**Date**: 2026-04-09
**Phase**: Phase 2 Recovery — Modality Retraining

---

## 📊 Current Model Status

| Modality | Model File | Val Accuracy | Architecture | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Emotion (7 cls)** | `models/vmax/emotion_master/emotion_master_optimal.keras` | 🔄 Training (v4) | VGG CNN, 48×48 FER2013 | IN PROGRESS |
| **Voice (8 cls)** | `models/vmax/voice_cloud/voice_cloud_optimal.keras` | **73.3%** | 1D-CNN + BiLSTM, MFCC | ✅ COMPLETE |
| **Ensemble Backbone** | `models/autonomous/ensemble_backbone.keras` | ❌ 0% (was) | MobileNetV2 (9-class) | ⚠️ Eval-corrected |

---

## 🔧 Root Cause Analysis (Phase 1 Failures)

### 1. Ensemble — 0% Accuracy
- **Cause A**: Model had `Rescaling(1./255)` internally; evaluation script was also dividing by 255 → double normalization → zero-input collapse
- **Cause B**: `class_high` (idx 7) and `class_low` (idx 8) dummy folders present in train/val → model biased to predict dummy class
- **Fix Applied**: Removed `/255.0` from `evaluate_final.py`; purged dummy folders

### 2. Voice — 16% Accuracy (old approach)
- **Cause**: Converted MFCC spectrograms to PNG images and ran through MobileNetV2 (ImageNet backbone). ImageNet statistics are irrelevant to spectrogram textures.
- **Fix Applied**: New `train_voice_v2.py` → raw MFCC time-series (130 frames × 120 features) fed into 1D-CNN + BiLSTM → **73.3% val_accuracy**

### 3. Emotion — 25% Accuracy (old approach)
- **Cause**: MobileNetV2 α=0.35 with 40 frozen layers applied to FER2013 (noisy 48×48 grayscale upscaled). ImageNet pretraining mismatched; extreme class weight (Disgust=9.41×) exploded gradients.
- **Fix Applied**: New `train_emotion_v4.py` → custom VGG-style CNN trained from scratch on native 48×48 FER2013; class weights capped at 3.0; pre-augmented with numpy flip; batch=128

---

## 📈 Phase 2 Results

### Voice (✅ COMPLETE)
| Metric | Value |
| :--- | :--- |
| Best Val Accuracy | **73.3%** |
| Training Epochs | 48 (early stop; best at epoch 33) |
| Architecture | 1D-CNN (64/128 filters) + BiLSTM (64/48 units) |
| Training Time | ~5.4 min |
| Model Size | 1.4 MB |
| Classes | neutral, calm, happy, sad, angry, fearful, disgust, surprised |

> **Context**: RAVDESS BiLSTM SOTA is ~75-85% with data augmentation. 73.3% is solid on CPU-only with 1,445 samples.

### Emotion (🔄 IN PROGRESS — `train_emotion_v4.py`)
| Metric | Value |
| :--- | :--- |
| Architecture | VGG-style CNN (4 conv blocks + GAP) |
| Parameters | ~2.0M (7.8 MB) |
| Batch Size | 128 |
| Training Data | 57,418 (28,709 + horizontal flip augment) |
| Target Val Acc | 60-70% (FER2013 SOTA ~73%) |
| Steps/Epoch | 448 |

---

## ⚙️ Evaluation Correction Log

| Script | Change | Reason |
| :--- | :--- | :--- |
| `evaluate_final.py` | Removed `X_test / 255.0` | Ensemble model already has `Rescaling(1./255)` layer |
| `evaluate_final.py` | Updated class_names to 7 (removed `Other1`, `Other2`) | Dummy classes purged from dataset |

---

## 📦 Deployment Readiness

| Modality | Ready? | Notes |
| :--- | :--- | :--- |
| Voice | ✅ YES | 73.3% — acceptable for inference |
| Emotion | 🔄 PENDING | Awaiting v4 training result |
| Ensemble Backbone | ⚠️ PARTIAL | Normalization corrected; needs re-eval |
| Face Identity | ✅ YES | Saved from previous training |
| Behavior | ✅ YES | Saved from previous training |

---

## 🛠️ Next Evaluation Steps
1. After emotion training completes → run `python scripts/evaluate_final.py`
2. Run `python scripts/evaluate_voice.py` to validate voice model on held-out RAVDESS split
3. Wire trained emotion + voice models into backend `/api/predict` endpoint
4. Perform end-to-end HUD inference test

---

*Last updated: 2026-04-09 | Phase 2 Recovery*
