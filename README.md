# 💎 ORIEN | Neural Synergy Ecosystem — Phase 2 Recovery

ORIEN is a proactive, deeply empathetic multimodal AI companion that anticipates user needs through real-time behavioral and emotional alignment. It delivers human-centric assistance using adaptive neural perception across Face, Voice, and Behavior modalities.

---

## 🚀 Core Capabilities

### 1. Parallel Multimodal Perception
- Real-time analysis across Face (Emotion + Identity), Voice, Gesture, Behavior, and Gaze
- Designed for sub-200ms local CPU inference
- Streaming disk-based data pipeline — no full-dataset cache required

### 2. Adaptive Behavioral Intelligence
- Detects user states: `FLOW`, `CALM`, `STRESSED`, `DISTRACTED`, `OVERWHELMED`
- Continuous local fine-tuning on user interaction patterns
- Probabilistic temporal smoothing to prevent rapid state oscillation

### 3. Multimodal Fusion Engine
- Each modality produces an emotion/state probability vector
- Bayesian Resolver combines vectors into a final unified state
- WebSocket bridge pushes live state to the HUD frontend

### 4. Futuristic HUD Interface
- 3D Neural Matrix Grid + Nerve Orb (Three.js) synced to user state
- Real-time telemetry charts, voice STT/TTS, behavioral feed
- Chromic state-transition animations

---

## 🧠 Neural Architecture

### Fusion Pipeline
```
FACE EMOTION  ──┐
FACE IDENTITY ──┤
GAZE/EYE      ──┼──► MULTIMODAL FUSION ──► BAYESIAN RESOLVER ──► HUD STATE
GESTURE       ──┤
VOICE         ──┤
BEHAVIOR      ──┘
```

### Model Clusters (Production Trainers)

| Modality | Architecture | Input | Script | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Face Emotion** | VGG-style CNN (4 Conv blocks + GAP) | FER2013 32×32 grayscale | `train_emotion_master.py` | ✅ Ready |
| **Voice Emotion** | 1D-CNN + BiLSTM | RAVDESS MFCC (130×120) | `train_voice_master.py` | ✅ Ready |
| **Face Identity** | MobileNetV2 α=0.35 | LFW 96×96 | `local_trainer.py` | ✅ Saved |
| **Behavior** | Dense MLP (14 features) | Balabit kinematic CSV | `local_trainer.py` | ✅ Saved |
| **Gesture** | MobileNetV2 α=0.35 | HaGRID 96×96 | `local_trainer.py` | ✅ Saved |
| **Ensemble** | Sequential (Rescaling→MobileNetV2→Head) | 96×96 | `ensemble_backbone.keras` | ✅ Healthy |

> **Note on Ensemble Backbone**: Model includes `Rescaling(1./255)` internally. Do **not** manually divide by 255 in evaluation scripts.

---

## 📊 Dataset Ecosystem

| Modality | Dataset | Samples | Location |
| :--- | :--- | :--- | :--- |
| **Face Emotion** | FER2013 | 35,887 (7 classes) | `dataset/face_emotion/fer2013.csv` |
| **Voice Emotion** | RAVDESS | 1,445 WAV → MFCC PNGs | `dataset/voice/Actor_01-24/` |
| **Face Identity** | LFW | 26,499+ JPGs | `dataset/vision_preprocessed/` |
| **Gesture** | HaGRID | — | `dataset/gesture/classes/` |
| **Behavior** | Balabit | 822 sessions | `dataset/behavior/` |
| **Gaze/Eye** | Custom | — | `dataset/eye_monitor/train/` |

---

## ⚙️ Core Optimization Log

| Issue | Fix | Status |
| :--- | :--- | :--- |
| Dummy bias (`class_high`/`class_low`) | Deleted from train/ and val/ | ✅ Done |
| Voice `classes/` empty (Parquet shards) | Ran `scripts/unpack_voice_mfcc.py` — extracted 1,445 MFCC PNGs | ✅ Done |
| Ensemble double-normalisation (`/255` twice) | Removed manual divide from `evaluate_final.py` | ✅ Done |
| MobileNetV2 ImageNet mismatch on FER/MFCC | Replaced with domain-specific CNN and BiLSTM | ✅ Done |
| OOM at batch_size=4 | Sequential training, GAP head, reduced BiLSTM units | ✅ Done |

---

## ⚡ Getting Started

```bash
# 1. Unpack voice features (one-time)
python scripts/unpack_voice_mfcc.py

# 2. Train emotion master model (FER2013 CSV, CPU-safe)
python scripts/train_emotion_master.py

# 3. Train voice master model (RAVDESS WAV, CPU-safe)
python scripts/train_voice_master.py

# 4. Evaluate ensemble (post-training)
python scripts/evaluate_final.py

# 5. Launch HUD + Backend
python backend/main.py
# Open frontend/index.html
```

> ⚠️ **CPU-Only Note**: Run emotion and voice training **sequentially** — simultaneous training causes OOM on 16GB RAM systems.

---

## 🔐 System Integrity
ORIEN is a secure, user-centric system. Internal architectures and model details are abstracted into human-friendly insights. All telemetry is processed locally to ensure privacy and low-latency performance.

---

*Last updated: 2026-04-09 | Unified Neural Synergy Engine Status: OPTIMAL*
