import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import base64
import cv2
import threading
import asyncio

# Global TF import (Deferred)
tf = None

log = logging.getLogger("ORIEN.Inference")

class NeuralPredictor:
    """
    Optimized Neural Predictor.
    """
    EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    GESTURE_CLASSES = ['call', 'dislike'] # Synced with models/vmax/gesture/classes.json

    def __init__(self):
        self.models = {}
        self.model_shapes = {}
        self.mean_std = {} # For normalization
        self.behavior_history = []
        self._lock = threading.Lock()
        self._gpu_active = False
        
    def _check_tf(self):
        global tf
        if tf is None:
            try:
                import tensorflow as _tf
                tf = _tf
                log.info("🧠 Neural Synapse (TensorFlow) Connected.")
                # GPU Auto-Tuning (Deferred)
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self._gpu_active = True
                    log.info("🚀 Neural Accelerator [GPU] Activated.")
            except ImportError:
                log.warning("⚠️ TensorFlow not detected. Minimal fallbacks active.")
                return False
            except Exception as e:
                log.error(f"❌ Critical Neural Fault: {e}")
                return False
        return True

    def load_model(self, modality: str):
        if not self._check_tf(): return False
        with self._lock:
            if modality in self.models: return True
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if modality == "behavior":
                path = os.path.join(base, "models", "vmax", modality, f"{modality}_optimal.joblib")
                if os.path.exists(path):
                    try:
                        import joblib
                        self.models[modality] = {"type": "KERAS", "core": joblib.load(path)}
                        log.info(f"✅ BEHAVIOR Neural Path Established (Joblib Elite).")
                        return True
                    except Exception as e:
                        log.error(f"Err {modality}: {e}")
                return False

            # Support for both Keras and TFLite optimized cores
            keras_path = os.path.join(base, "models", "vmax", modality, f"{modality}_optimal.keras")
            tflite_path = os.path.join(base, "models", "vmax", modality, f"{modality}_optimal.tflite")
            
            if os.path.exists(tflite_path):
                try:
                    from tensorflow import lite
                    interpreter = lite.Interpreter(model_path=tflite_path)
                    interpreter.allocate_tensors()
                    self.models[modality] = {
                        "type": "TFLITE",
                        "core": interpreter,
                        "input": interpreter.get_input_details(),
                        "output": interpreter.get_output_details()
                    }
                    log.info(f"⚡ {modality.upper()} Neural Path Established (TFLite Optimized).")
                    return True
                except Exception as e:
                    log.error(f"Err TFLite {modality}: {e}")

            if os.path.exists(keras_path):
                try:
                    m = tf.keras.models.load_model(keras_path, compile=False)
                    self.models[modality] = {"type": "KERAS", "core": m}
                    self.model_shapes[modality] = m.input_shape
                    
                    # Load normalization artifacts for voice
                    if modality == "voice_cloud":
                        try:
                            m_path = os.path.join(base, "models", "vmax", modality, "voice_mean.npy")
                            s_path = os.path.join(base, "models", "vmax", modality, "voice_std.npy")
                            if os.path.exists(m_path) and os.path.exists(s_path):
                                self.mean_std["voice"] = (np.load(m_path), np.load(s_path))
                                log.info(f"📊 Voice Normalization Artifacts Loaded.")
                        except: pass

                    log.info(f"✅ {modality.upper()} Neural Path Established.")
                    return True
                except Exception as e:
                    log.error(f"Err {modality}: {e}")
            return False

    def predict_behavior_state(self, b) -> str:
        f = [b.mouse_speed, b.jitter, b.click_count, getattr(b,'wpm',0), getattr(b,'backspaces',0), getattr(b,'window_switches',0)]
        self.behavior_history.append(f)
        if len(self.behavior_history) > 50: self.behavior_history.pop(0)

        if not self.load_model("behavior"):
            return "Erratic Alert" if b.jitter > 35 else "Nominal"

        try:
            m_data = self.models["behavior"]["core"]
            # Check if it's a scikit-learn pipeline/model or our custom dictionary
            if isinstance(m_data, dict):
                model = m_data["model"]
                scaler = m_data.get("scaler")
            else:
                model = m_data
                scaler = None
            
            # Extract features (match extract_features from pipeline)
            # Simplification: Use provided metrics as feature vector
            features = {
                'mouse_speed': b.mouse_speed,
                'jitter': b.jitter,
                'click_count': b.click_count,
                'wpm': getattr(b, 'wpm', 0),
                'backspaces': getattr(b, 'backspaces', 0),
                'window_switches': getattr(b, 'window_switches', 0)
            }
            # Pad with zeros to match the training feature count (which was ~14)
            # Re-read training feature list if possible, or use the saved list
            f_names = m_data.get("features", [])
            f_vector = []
            for name in f_names:
                f_vector.append(features.get(name, 0.0))
            
            X = np.array([f_vector])
            X_scaled = scaler.transform(X)
            
            p = model.predict(X_scaled)[0]
            
            if p == 1: return "Highly Anomalous" # Binary classification match
            if b.jitter > 25: return "Stressed"
            return "Nominal"
        except Exception as e:
            log.warning(f"Behavior prediction fallback: {e}")
            return "Stable"

    def _decode_frame(self, frame_b64: str, size: int = 128):
        try:
            if "," in frame_b64: frame_b64 = frame_b64.split(",")[1]
            img_bytes = base64.b64decode(frame_b64)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None: return None
            
            # High-Fidelity Image Processing (Aligned with Vision-Lite trainer)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            
            # Normalization is now a layer in the model (tf.keras.layers.Rescaling)
            # We only cast to float32 here.
            img = img.astype('float32') # Removed / 255.0
            return np.expand_dims(img, axis=0)
        except Exception as e:
            log.debug(f"Frame decode error: {e}")
            return None

    def _predict_core(self, modality: str, img: np.ndarray):
        m = self.models.get(modality)
        if not m: return None
        if m["type"] == "TFLITE":
            interp = m["core"]
            interp.set_tensor(m["input"][0]['index'], img)
            interp.invoke()
            return interp.get_tensor(m["output"][0]['index'])
        else:
            return m["core"].predict(img, verbose=0)

    def predict_face_emotion(self, frame) -> dict:
        if not self.load_model("face"): return {"emotion":"Neutral", "confidence":0.0}
        img = self._decode_frame(frame, 128) # Signal alignment
        if img is None: return {"emotion":"Neutral", "confidence":0.0}
        p = self._predict_core("face", img)[0]
        i = np.argmax(p)
        return {"emotion": self.EMOTION_CLASSES[i], "confidence": float(p[i])}

    def predict_eye_gaze(self, frame) -> dict:
        if not self.load_model("eye"): return {"gaze":"Center", "confidence":0.0}
        img = self._decode_frame(frame, 128) # Signal alignment
        if img is None: return {"gaze":"Center", "confidence":0.0}
        res = self._predict_core("eye", img)
        p = res[0]
        i = np.argmax(p)
        return {"gaze": ["Center", "Left", "Right"][i], "confidence": float(p[i])}

    def predict_face_orl_identity(self, frame) -> dict:
        if not self.load_model("face_orl"): return {"subject":"Unknown", "confidence":0.0}
        img = self._decode_frame(frame, 128) # Signal alignment
        if img is None: return {"subject":"Unknown", "confidence":0.0}
        res = self._predict_core("face_orl", img)
        p = res[0]
        return {"subject": f"Subject_{np.argmax(p)+1:02d}", "confidence": float(np.max(p))}
    async def _run_model_async(self, modality: str, img):
        """Internal helper to run model in a thread pool."""
        if not self.load_model(modality): return None
        try:
            return await asyncio.to_thread(self.models[modality].predict, img, verbose=0)
        except Exception as e:
            log.warning(f"Async inference error [{modality}]: {e}")
            return None

    def predict_voice_emotion(self, features: np.ndarray) -> dict:
        """Neural Voice Prediction (Optimal BiLSTM)."""
        if not self.load_model("voice_cloud"): 
            return {"emotion": "Neutral", "confidence": 0.0}
        
        try:
            # Expected shape: (130, 120) for BiLSTM
            if features.shape != (130, 120):
                log.warning(f"Voice feature shape mismatch: {features.shape}. Expected (130, 120)")
                # Minimal Padding/Truncation check
                if features.size == 130 * 120:
                    features = features.reshape(130, 120)
                else:
                    return {"emotion": "Neutral", "confidence": 0.0}

            # Normalization
            if "voice" in self.mean_std:
                mean, std = self.mean_std["voice"]
                features = (features - mean) / std

            feat_batch = np.expand_dims(features, axis=0) # (1, 130, 120)
            res = self._predict_core("voice_cloud", feat_batch)
            p = res[0]
            idx = np.argmax(p)
            return {"emotion": self.EMOTION_CLASSES[idx] if idx < len(self.EMOTION_CLASSES) else "Neutral", "confidence": float(p[idx])}
        except Exception as e:
            log.warning(f"Voice prediction error: {e}")
            return {"emotion": "Neutral", "confidence": 0.0}

    def _decode_once(self, frame_b64: str, size: int = 128):
        """Unified decoding to prevent redundant processing."""
        return self._decode_frame(frame_b64, size)

    async def predict_ensemble(self, frame_b64: str) -> dict:
        """
        Parallel Ensemble Inference.
        Decodes once, runs all models concurrently.
        """
        img = self._decode_once(frame_b64, 128)
        if img is None: return {"emotion":"Neutral", "confidence":0.0, "votes": {}}

        # Identifier separation
        # Only 'face_alt' and 'emotion_master' have FER features.
        modalities = ["face_alt", "emotion_master"]
        
        # Trigger all predictions in parallel
        tasks = [self._run_model_async(m, img) for m in modalities]
        results = await asyncio.gather(*tasks)

        rs = []
        votes = {}
        for i, res in enumerate(results):
            if res is not None:
                p = res[0]
                if p.shape[0] >= 7:
                    probs = p[:7]
                    rs.append(probs)
                    votes[modalities[i]] = self.EMOTION_CLASSES[np.argmax(probs)]
        
        if not rs: return {"emotion":"Neutral", "confidence":0.0, "votes": {}}
        
        mean_p = np.mean(rs, axis=0)
        idx = int(np.argmax(mean_p))
        return {
            "type":       "EMOTION_UPDATE",
            "emotion":    self.EMOTION_CLASSES[idx],
            "confidence": float(mean_p[idx]),
            "votes":      votes,
            "scores":     {self.EMOTION_CLASSES[j]: float(mean_p[j]) for j in range(7)}
        }

    def _decode_once(self, frame_b64: str) -> Dict[str, np.ndarray]:
        """
        Optimized Unified Decode.
        Decodes the frame once and produces all required resolution tensors.
        """
        try:
            if "," in frame_b64: frame_b64 = frame_b64.split(",")[1]
            img_bytes = base64.b64decode(frame_b64)
            raw = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if raw is None: return {}

            # Reverting to 128x128 to match baseline.
            r128 = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_CUBIC).astype('float32')
            
            return {
                "128": np.expand_dims(r128, axis=0)
            }
        except Exception as e:
            log.error(f"Critical Frame Decode Failure: {e}")
            return {}

    async def predict_full_suite_parallel(self, frame_b64: str) -> dict:
        """
        Complete Neural Suite Parallelization (Thread-Safe).
        Runs Ensemble + Gaze + Identity in a single non-blocking pass.
        Includes Self-Healing Logic for model failures.
        """
        if not frame_b64: return {"error": "SENSOR_DROPOUT", "mode": "MINIMAL"}

        # 1. Unified Decode
        tensors = await asyncio.to_thread(self._decode_once, frame_b64)
        if not tensors: return {"error": "DECODE_FAILURE", "mode": "MINIMAL"}

        t128 = tensors.get("128")

        # 2. Execution Map
        tasks = {
            "ensemble": self.predict_ensemble_optimized(t128),
            "gaze":     self._run_model_async_direct("eye", t128),
            "id":       self._run_model_async_direct("face_orl", t128),
            "gesture":  self._run_model_async_direct("gesture", t128)
        }

        # 3. Dynamic Concurrent Execution
        keys = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        res_map = dict(zip(keys, results))

        # 4. Self-Healing & Fallback Logic
        ensemble = res_map.get("ensemble", {"emotion": "Neutral", "confidence": 0.0, "votes": {}})
        if isinstance(ensemble, Exception): ensemble = {"emotion": "Neutral", "confidence": 0.0, "votes": {}}

        fused = {
            "type":       "EMOTION_UPDATE",
            "emotion":    ensemble.get("emotion", "Neutral"),
            "confidence": ensemble.get("confidence", 0.0),
            "votes":      ensemble.get("votes", {}),
            "gaze":       "Center",
            "identity":   "Unknown",
            "gesture":    "NONE",
            "status":     "HEALTHY"
        }

        # Safe Extraction for Gaze
        gaze_res = res_map.get("gaze")
        if gaze_res is not None and not isinstance(gaze_res, Exception):
            p = gaze_res[0]
            idx = np.argmax(p)
            fused["gaze"] = ["Center", "Left", "Right"][idx]
            fused["gaze_conf"] = float(p[idx])
        else:
            fused["status"] = "REGRADED" # Sensor/Model dropout

        # Safe Extraction for Identity
        id_res = res_map.get("id")
        if id_res is not None and not isinstance(id_res, Exception):
            p = id_res[0]
            idx = np.argmax(p)
            fused["identity"] = f"Subject_{idx+1:02d}"
            fused["id_conf"] = float(p[idx])

        # Safe Extraction for Gesture
        gest_res = res_map.get("gesture")
        if gest_res is not None and not isinstance(gest_res, Exception):
            p = gest_res[0]
            idx = np.argmax(p)
            if idx < len(self.GESTURE_CLASSES):
                fused["gesture"] = self.GESTURE_CLASSES[idx]
            fused["gest_conf"] = float(p[idx])

        return fused

    async def predict_ensemble_optimized(self, t128) -> dict:
        """Optimized ensemble using correctly scaled inputs for each member."""
        # Identifier separation
        modalities = ["face_alt", "emotion_master"]
        
        results = []
        votes = {}
        
        # ── Member 1: face_alt (128x128x3) ──
        res_alt = await self._run_model_async_direct("face_alt", t128)
        if res_alt is not None:
            p = res_alt[0][:7]
            results.append(p)
            votes["face_alt"] = self.EMOTION_CLASSES[np.argmax(p)]

        # ── Member 2: emotion_master (32x32x1 Gray) ──
        if self.load_model("emotion_master"):
            try:
                # Convert 128x128x3 to 32x32x1
                img32 = cv2.resize(t128[0], (32, 32), interpolation=cv2.INTER_AREA)
                gray32 = cv2.cvtColor(img32.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray32 = np.expand_dims(gray32, axis=(0, -1)).astype('float32') # (1, 32, 32, 1)
                
                res_master = await asyncio.to_thread(self._predict_core, "emotion_master", gray32)
                if res_master is not None:
                    p = res_master[0][:7]
                    results.append(p)
                    votes["emotion_master"] = self.EMOTION_CLASSES[np.argmax(p)]
            except Exception as e:
                log.warning(f"Emotion Master Ensemble error: {e}")

        if not results: return {"emotion": "Neutral", "confidence": 0.0}
        
        mean_p = np.mean(results, axis=0)
        idx = int(np.argmax(mean_p))
        return {
            "emotion":    self.EMOTION_CLASSES[idx],
            "confidence": float(mean_p[idx]),
            "votes":      votes
        }

    async def _run_model_async_direct(self, modality: str, tensor: np.ndarray):
        """Runs model directly on provided tensor in a thread pool."""
        if not self.load_model(modality): return None
        try:
            return await asyncio.to_thread(self._predict_core, modality, tensor)
        except Exception as e:
            log.error(f"Inference Fault [{modality}]: {e}")
            return None

predictor = NeuralPredictor()
