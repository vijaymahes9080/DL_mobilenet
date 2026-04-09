# 🏁 TRAINING REPORT: ORIEN NEURAL ECOSYSTEM — Phase 2

**Date**: 2026-04-09
**Hardware**: CPU-only, 16GB RAM, Windows (TF 2.21.0)
**Status**: 🔄 EMOTION IN PROGRESS | VOICE ✅ COMPLETE

---

## 📊 Phase 2 Training Summary

| Modality | Trainer Script | Architecture | Best Val Acc | Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Emotion** | `train_emotion_v4.py` | VGG CNN (48×48, 7 cls) | 🔄 — | ~22 min/epoch | IN PROGRESS |
| **Voice** | `train_voice_v2.py` | 1D-CNN + BiLSTM (8 cls) | **73.3%** | 5.4 min | ✅ DONE |
| Emotion (v1 — MobileNetV2) | `microbatch_retrain.py` | MobileNetV2 α=0.35 | 25.6% | 43.8 min | ❌ ABANDONED |
| Voice (v1 — MFCC-PNG) | `microbatch_retrain.py` | MobileNetV2 α=0.35 | 16.2% | 2.4 min | ❌ ABANDONED |

---

## 🔄 Iteration Log

### Attempt 1 — MobileNetV2 on Both (Batch 8)
- **Result**: Emotion 25.6%, Voice 16.2%
- **Failure mode**: ImageNet pretrained BatchNorm stats misaligned with FER grayscale and MFCC spectrogram textures. Backbone froze meaningful gradient signal.
- **Decision**: Abandon transfer learning; use domain-specific architectures from scratch.

### Attempt 2 — Voice BiLSTM v1 (Batch 32)
- **Result**: OOM crash mid-epoch (gradient tensor `[46,130,600]` allocation failed)
- **Best before crash**: 75.1% (saved by ModelCheckpoint)
- **Fix**: Reduced BiLSTM units (128→64, 64→48), CNN filters (128→64, 256→128), batch=16

### Attempt 3 — Voice BiLSTM v2 OOM-fixed (Batch 16)
- **Result**: ✅ **73.3% val_accuracy**, early-stop at epoch 48 (best at epoch 33)
- **Saved**: `models/vmax/voice_cloud/voice_cloud_optimal.keras`

### Attempt 4 — Emotion CNN v3 (VGG, on-the-fly aug, batch 64)
- **Result**: 14.2% mid epoch-1, abandoned — tf.data random_crop augmentation 3s/step on CPU
- **Fix**: Pre-augment in numpy (horizontal flip only), use batch=128

### Attempt 5 — Emotion CNN v4 (pre-aug numpy, GAP head, batch 128)
- **Status**: 🔄 Currently running
- **Architecture**: 4 Conv blocks (32/64/128/256) + GlobalAveragePooling2D + Dense(512/256) + Softmax(7)
- **Parameters**: ~2.0M (7.8 MB)
- **Expected**: 60-70% based on FER2013 dataset characteristics

---

## 🧰 Key Technical Decisions

### Why NOT MobileNetV2 for Emotion/Voice?
MobileNetV2 is pretrained on ImageNet RGB photos. FER2013 is 48×48 grayscale faces (noisy, low-res). RAVDESS voice inputs are temporal MFCC spectrograms. The ImageNet BatchNorm statistics (mean/variance) are calibrated for natural photo textures — applying them to these domains causes:
- Early-epoch loss stagnation (gradient signal drowned by BN mismatch)
- High learning rate instability
- Class weight extremes (Disgust=9.4×) causing gradient explosion

### Why VGG-style CNN for Emotion?
- Proven architecture on FER2013 (multiple Kaggle leaderboard winners use VGG-based CNNs)
- Trains from scratch — no BN stat mismatch
- GlobalAveragePooling2D head avoids huge Flatten→Dense gradient tensors (prevents OOM)

### Why 1D-CNN + BiLSTM for Voice?
- MFCC is a 1D time-series (time frames × feature coefficients)
- 1D-CNN captures local n-gram patterns in frequency space
- BiLSTM captures global sequence dependencies (emotion cadence, intonation)
- Avoids converting to PNG which loses temporal ordering and introduces spatial artifacts

---

## 📂 Model Artifacts

| File | Purpose | Format |
| :--- | :--- | :--- |
| `models/vmax/voice_cloud/voice_cloud_optimal.keras` | Voice inference | Keras 3 |
| `models/vmax/voice_cloud/classes.json` | Voice label map | JSON |
| `models/vmax/voice_cloud/voice_mean.npy` | Normalization mean (z-score) | NumPy |
| `models/vmax/voice_cloud/voice_std.npy` | Normalization std (z-score) | NumPy |
| `models/vmax/emotion_master/emotion_master_optimal.keras` | Emotion inference | Keras 3 |
| `models/vmax/emotion_master/classes.json` | Emotion label map (7 classes) | JSON |

---

## ⚙️ Dataset Corrections Applied

| Fix | Script | Effect |
| :--- | :--- | :--- |
| Purged `class_high` + `class_low` | Manual (PowerShell) | Removed dummy bias from emotion training |
| Unpacked RAVDESS WAV → MFCC PNG | `scripts/unpack_voice_mfcc.py` | 1,445 PNGs across 8 emotion classes |
| Removed double `/255` in evaluator | `scripts/evaluate_final.py` | Ensemble normalization now correct |

---

*Last updated: 2026-04-09 | Phase 2 Recovery*
