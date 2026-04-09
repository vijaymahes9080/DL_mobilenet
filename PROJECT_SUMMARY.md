# 💎 PROJECT SUMMARY: ORIEN Neural Ecosystem — Phase 2

### 🛠️ Model Information
- **Project Name**: ORIEN (Observational Responsive Intelligence Engine for Neural Synergy)
- **Version**: Phase 2 Recovery Build
- **Architecture**: Domain-specific CNNs + BiLSTM (replacing MobileNetV2 transfer-learning approach)
- **Updated**: 2026-04-09
- **Status**: 🔄 EMOTION TRAINING IN PROGRESS | VOICE ✅ COMPLETE

---

### 🆙 Phase 2 Changes vs Phase 1

| Issue | Root Cause | Fix Applied |
| :--- | :--- | :--- |
| 0% ensemble accuracy | Double `/255` normalisation + dummy class bias | Removed manual rescaling; purged `class_high`/`class_low` |
| Voice training failed | `voice_cloud/classes/` was empty | Ran `unpack_voice_mfcc.py` → 1,445 MFCC PNGs extracted |
| Emotion stuck at 25% | MobileNetV2 ImageNet weights mismatched to FER grayscale | Replaced with VGG-style custom CNN (train_emotion_v4.py) |
| Voice stuck at 16% | MobileNetV2 on MFCC-PNGs not suited to spectrogram textures | Replaced with 1D-CNN + BiLSTM on raw MFCC time-series |
| OOM at batch 4 | Two training processes sharing 16GB RAM | Sequential training + GlobalAveragePooling head (no Flatten) |

---

### 📝 Project Description
- **Objective**: Real-time emotionally-aware AI companion monitoring Face, Voice, and Behavior to provide proactive support.
- **Scope**: Multi-modal telemetry, memory-backed conversational AI, futuristic HUD frontend.
- **Approach**: Per-modality domain-specific models trained from scratch or with minimal transfer, then fused via a Bayesian Resolver.

---

### 💻 Technical Stack
- **Language**: Python 3.10+ (Backend), JavaScript (Frontend / HUD)
- **Frameworks**: TensorFlow 2.21, scikit-learn, librosa, FastAPI, Three.js, ChromaDB
- **Architectures**: VGG-style CNN (Emotion), 1D-CNN + BiLSTM (Voice), MobileNetV2 α=0.35 (Identity/Gesture)
- **Tools**: WebSockets, REST API, local `.keras` model inference

---

### 🧠 Model Status

| Modality | Script | Val Accuracy | Model Path |
| :--- | :--- | :--- | :--- |
| **Emotion (7 cls)** | `train_emotion_v4.py` | 🔄 Training | `models/vmax/emotion_master/emotion_master_optimal.keras` |
| **Voice (8 cls)** | `train_voice_v2.py` | ✅ **73.3%** | `models/vmax/voice_cloud/voice_cloud_optimal.keras` |
| **Face Identity** | `local_trainer.py` | Saved | `models/vmax/face/face_optimal.keras` |
| **Behavior** | `local_trainer.py` | Saved | `models/vmax/behavior/behavior_optimal.keras` |
| **Gesture** | `local_trainer.py` | Saved | `models/vmax/gesture/gesture_optimal.keras` |

---

### 📂 Training Data

| Modality | Dataset | Size | Preprocessing |
| :--- | :--- | :--- | :--- |
| Emotion | FER2013 CSV | 35,887 samples | Native 48×48 grayscale, class weights capped at 3.0 |
| Voice | RAVDESS WAV | 1,445 files × 24 actors | MFCC (40 coefficients) + delta + delta² → (130, 120) time-series |
| Face Identity | LFW | 26,499+ | 96×96 RGB JPGs |
| Behavior | Balabit | 822 sessions | 14 kinematic features, StandardScaler |

---

### 📊 Accuracy Targets vs Achieved

| Modality | Target | Achieved | Note |
| :--- | :--- | :--- | :--- |
| Emotion | 95% | 🔄 In training | FER2013 SOTA ~73%; realistic target 60-70% |
| Voice | 90% | ✅ 73.3% | RAVDESS BiLSTM; SOTA ~85% with augmentation |
| Face Identity | 95% | Saved | LFW dataset |
| Behavior | 95% | Saved | Balabit mouse dynamics |

---

### 🚀 Deployment
- **Environment**: Local edge inference (CPU-only, 16GB RAM compatible)
- **API**: FastAPI REST + WebSocket for real-time HUD sync
- **Evaluation**: `scripts/evaluate_final.py` — ensemble uses internal `Rescaling(1./255)` layer

---

### ⚠️ Known Constraints
- **CPU-only**: No GPU on Windows native TF ≥ 2.11 (GPU requires WSL2 or DirectML plugin)
- **Sequential Training**: 16GB RAM requires training one modality at a time
- **FER2013 ceiling**: Inherently noisy dataset — 65-70% is realistic maximum without heavy augmentation

---

### 🔮 Next Steps
1. Confirm emotion v4 val_accuracy after training completes
2. Re-evaluate ensemble `ensemble_backbone.keras` with corrected normalization
3. Integrate trained `emotion_master` and `voice_cloud` models into FastAPI backend inference

---

*Last updated: 2026-04-09 | Phase 2 Recovery*
