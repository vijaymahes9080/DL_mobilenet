# 💎 ORIEN | Delivery Specification — Phase 2

### Emotion-Aware AI Assistant System [Neural End-To-End]

This document defines the current architectural delivery state for the ORIEN Neural Ecosystem after Phase 2 Recovery.

---

## 🧠 1. Neural Intelligence — Model Architecture

| Component | Architecture | Input | Target Acc | Actual (Phase 2) | Script |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Face Emotion** | VGG CNN (4 blocks + GAP) | FER2013 48×48 gray | 65%+ | 🔄 Training | `train_emotion_v4.py` |
| **Voice Emotion** | 1D-CNN + BiLSTM | RAVDESS MFCC (130×120) | 70%+ | ✅ **73.3%** | `train_voice_v2.py` |
| **Face Identity** | MobileNetV2 α=0.35 | LFW 96×96 | 90%+ | Saved | `local_trainer.py` |
| **Behavior** | Dense MLP (14 features) | Balabit kinematic | 90%+ | Saved | `local_trainer.py` |
| **Gesture** | MobileNetV2 α=0.35 | HaGRID 96×96 | 85%+ | Saved | `local_trainer.py` |
| **Fusion (Ensemble)** | Bayesian Resolver | All modality outputs | 95%+ | Eval-corrected | `evaluate_final.py` |

**Output Files:**
- `models/vmax/emotion_master/emotion_master_optimal.keras`
- `models/vmax/voice_cloud/voice_cloud_optimal.keras`
- `models/vmax/face/face_optimal.keras`
- `models/vmax/behavior/behavior_optimal.keras`
- `models/autonomous/ensemble_backbone.keras`

> ⚠️ **Normalisation Note**: `ensemble_backbone.keras` has `Rescaling(1./255)` as its first internal layer. Do NOT divide input by 255 externally when using this model.

---

## 🔌 2. API Failover Strategy
- **Key Rotation**: Automatic switching on network/rate errors
- **Local First**: Inference runs locally; cloud API used only for LLM response generation
- **Retry Logic**: Exponential backoff implemented in `backend/main.py`

---

## 🎨 3. HUD Frontend (Three.js)
- **Neural Matrix Grid**: 3D background syncs to emotional state (FLOW/CALM/STRESSED)
- **Nerve Orb**: Animated icosahedron reflecting emotional intensity
- **Chromic Glitch Effects**: State-transition visual feedback
- **Multi-Modal Panel**: Real-time charts, STT voice recording, TTS speaker output

---

## ⚙️ 4. Backend System (FastAPI)
- **WebSockets**: Full-duplex bridge for real-time frame transfer and inference results
- **Memory Engine**: ChromaDB vector DB for session-pattern personalization
- **Prediction Endpoint**: `/api/predict` — accepts face image + voice features, returns state JSON
- **Training Bridge**: `/api/training/update` — receives epoch telemetry from trainer callbacks

---

## 🚀 5. Inference Pipeline

```
1. SENSORS    → HUD captures webcam frames + mic audio + mouse telemetry
2. FEATURES   → Face crop → 48×48; Audio → MFCC (130×120); Mouse → 14 kinematic features
3. INFERENCE  → emotion_master.keras + voice_cloud.keras + behavior.keras → probability vectors
4. FUSION     → Bayesian Resolver combines vectors → unified state (FLOW/CALM/STRESSED/...)
5. RESPONSE   → Emotion-aware LLM prompt strategy → API call → human-friendly response
6. RENDER     → HUD updates state orb + color theme + speaks response via TTS
```

---

## 🏁 6. Setup Instructions

```bash
# Step 1: Create and activate venv
python -m venv .venv_training
.venv_training\Scripts\activate

# Step 2: Install dependencies
pip install tensorflow librosa pillow scikit-learn pandas numpy

# Step 3: Unpack voice MFCC (one-time)
python scripts/unpack_voice_mfcc.py

# Step 4: Train emotion model (runs ~2-3 hours on CPU)
python scripts/train_emotion_v4.py

# Step 5: Train voice model (runs ~6 min on CPU)
python scripts/train_voice_v2.py

# Step 6: Launch backend
pip install -r backend/requirements.txt
python backend/main.py

# Step 7: Open HUD
# Open frontend/index.html in browser
```

---

## ⚠️ 7. Constraints & Limitations

| Constraint | Detail |
| :--- | :--- |
| **CPU-only on Windows** | GPU requires WSL2 or TF-DirectML plugin; native Windows TF ≥ 2.11 is CPU-only |
| **Sequential Training** | 16GB RAM — do not run emotion + voice training simultaneously (OOM) |
| **FER2013 ceiling** | Inherently noisy 48×48 grayscale; SOTA ~73% — realistic target 60-70% |
| **RAVDESS size** | Only 1,445 samples — small dataset; augmentation needed to push above 80% |

---

## 🔮 8. Recommended Improvements

1. **WSL2 + GPU**: Move training to WSL2 for native GPU support → 10-50× speedup
2. **Emotion Augmentation**: Add elastic distortions, MixUp to improve FER2013 generalization
3. **Voice Augmentation**: Add time-stretch, pitch-shift, noise injection to RAVDESS
4. **Ensemble Re-training**: Retrain `ensemble_backbone.keras` on correct 7-class FER distribution (not dummy 9-class)
5. **Bio-Sensor Integration**: Wearable HR/HRV for behavioral state ground truth

---

*Last updated: 2026-04-09 | Phase 2 Recovery*
