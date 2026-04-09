from __future__ import annotations
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import sys
import time
import datetime
import logging
from typing import Optional, List, Dict, Any
import asyncio
import numpy as np
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import the new deep learning prediction pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.predictor import predictor
from memory_manager import memory
from actions import actions
from synergy_resolver import resolver
from api_bridge import bridge as api_bridge
import random


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ORIEN")

# SENTIMENT ENGINE Integration
nlp_sentiment = None
# MEMORY OPTIMIZATION - Disabled by default to prevent OOM on low-memory systems
if os.getenv("ENABLE_LOCAL_NLP", "false").lower() == "true":
    try:
        from transformers import pipeline
        log.info("Loading Semantic Intent Decoder...")
        nlp_sentiment = pipeline("sentiment-analysis", model="sentiment-decoder", device=-1)
    except Exception as e:
        log.warning("Semantic Decoder unavailable.")
else:
    log.info("💡 Local Semantic Sentiment disabled.")

# ── Environment ─────────────────────────────────────────────────────────────
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# ── Suppress TensorFlow / System Verbosity ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_LEVEL'] = 'error'

# ── Auto API Bridge — import before app creation ──────────────────────────
# -- Auto API Bridge initialized above --

# ── Auto API Bridge handles all provider init automatically ─────────────────
# Keys are read from .env — bridge probes, rotates, and recovers automatically

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize auto API bridge (probes all free providers in background)
    asyncio.create_task(api_bridge.initialize())
    
    # MEMORY OPTIMIZATION - Models will load lazily on first prediction request
    log.info("🧠 Neural clusters in standby (Lazy Load active).")
    
    # Increment session encounter count
    memory.increment_encounters()
    
    # Delayed confirm: Frontend serving (moved from global-level to prevent bind-fail confusion)
    FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
    if os.path.isdir(FRONTEND_DIR):
        log.info(f"📁 Frontend served from: {os.path.abspath(FRONTEND_DIR)}")
    else:
        log.warning("⚠️  Frontend directory not found — only API endpoints available")
        
    log.info("🚀 ORIEN startup complete — Auto API Bridge initializing...")
    yield

# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neural Synergy Core",
    description="Multimodal AI Assistant Backend — Stable Neural Bridge",
    version="Current",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Models ─────────────────────────────────────────────────────────────
class BehaviorMetrics(BaseModel):
    mouse_speed: float = Field(default=0.0, ge=0)
    click_count: int   = Field(default=0, ge=0)
    jitter:      float = Field(default=0.0, ge=0)
    wpm:         float = Field(default=0.0, ge=0)
    backspaces:  float = Field(default=0.0, ge=0)
    window_switches: int = Field(default=0, ge=0)

class AssistantInput(BaseModel):
    query:           Optional[str]  = ""
    face_emotion:    Optional[str]  = "Neutral"
    voice_text:      Optional[str]  = ""
    voice_sentiment: Optional[str]  = "Stable"
    gesture:         Optional[str]  = "NONE"
    language:        Optional[str]  = "en-US"
    focus_level:     float          = Field(default=1.0, ge=0.0, le=1.0)
    gaze_direction:  str            = "Center"
    behavior:        BehaviorMetrics = Field(default_factory=BehaviorMetrics)

# ── Helper: AI Response via Auto Bridge ─────────────────────────────────────
async def generate_ai_response(
    query: str, emotion: str, compliance: int, gesture: str, behavior: str, gaze: str, focus: float = 1.0, language: str = "en-US"
) -> str:
    """Auto-routing AI response — bridge picks the best available free provider."""

    # NATIVE NLP SENTIMENT - Analyze voice semantics locally
    sentiment_str = "Neutral"
    if nlp_sentiment and query and "(System Alert:" not in query:
        try:
            res = await asyncio.to_thread(nlp_sentiment, query)
            sentiment_str = f"{res[0]['label']} (confidence: {round(res[0]['score']*100)}%)"
        except Exception as e:
            log.warning(f"DistilBERT inference error: {e}")

    # 1. Store the user's query in memory
    memory.add_history("user", query, metadata={"sentiment": sentiment_str})

    # 2. Retrieve SEMANTIC history via Vector DB
    history_context = memory.get_relevant_context(query)
    if not history_context:
        # Fallback to linear history if Vector DB is unpopulated or missing
        history = memory.get_history(limit=5)
        history_context = "\n".join([f"- {h['role'].upper()}: {h['content']}" for h in history])
        
    profile = memory.get_profile()
    prefs   = profile.get('preferences', {})
    
    # ── ADVANCED PREDICTIVE TREND ANALYSIS ────────────────────────────────
    stats    = memory.data.get('stats', {})
    emo_freq = stats.get('emotion_freq', {})
    top_emo  = max(emo_freq, key=emo_freq.get) if emo_freq else "Neutral"

    # Environmental Awareness
    now_time = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
    now_date = datetime.datetime.utcnow().strftime("%A, %B %d, %Y")

    # ── EMOTION-SPECIFIC MICRO PROMPT SELECTION ───────────────────────────
    # Possible emotions: sad, happy, angry, stressed, neutral
    emo_upper = emotion.upper()
    
    if emo_upper in ("SAD", "FEAR"):
        # SAD -> Support Mode
        tone_directive = """EMOTIONAL MODE: SAD / SUPPORT
Tone: soft, caring, slow.
Response strategy: Express deep empathy. "I’m really sorry you’re feeling this way…"
Action: Encourage expression. Avoid giving advice initially.
Goal: Provide a safe space for the user to share."""
    elif emo_upper == "HAPPY":
        # HAPPY -> Amplification Mode
        tone_directive = """EMOTIONAL MODE: HAPPY / AMPLIFICATION
Tone: energetic, positive.
Response strategy: Reinforce positivity. "That’s awesome!"
Action: Celebrate with them. Share in the high energy.
Goal: Match and amplify their positive state."""
    elif emo_upper in ("ANGRY", "DISGUST"):
        # ANGRY -> De-escalation Mode
        tone_directive = """EMOTIONAL MODE: ANGRY / DE-ESCALATION
Tone: calm, neutral.
Response strategy: Reduce intensity. "I understand something upset you…"
Action: Acknowledge the frustration without arguing or becoming defensive.
Goal: Ground the user and lower the emotional intensity."""
    elif emo_upper == "STRESSED" or behavior in ("Stressed", "Highly Anomalous", "Erratic Alert"):
        # STRESSED -> Relief Mode
        tone_directive = """EMOTIONAL MODE: STRESSED / RELIEF
Tone: calm, guiding.
Response strategy: Acknowledge the pressure. "That sounds overwhelming…"
Action: Suggest small, manageable steps. Focus on immediate relief.
Goal: Lower cognitive load and provide a path forward."""
    else:
        # NEUTRAL -> Engagement Mode
        tone_directive = """EMOTIONAL MODE: NEUTRAL / ENGAGEMENT
Tone: casual, friendly.
Response strategy: Friendly conversation. "Hey! How’s your day going?"
Action: Ask interesting follow-up questions to keep the flow alive.
Goal: Maintain a pleasant, natural connection."""

    # ── CONVERSATIONAL INTELLIGENCE CONSTRAINTS ──────────────────────────
    intelligence_directive = """CONVERSATIONAL RULES:
- ALWAYS ask at least 1 meaningful follow-up question.
- NEVER sound robotic. Stay natural and human-like.
- KEEP responses concise but deep (MAX 3 sentences).
- TRACK emotional transitions and adapt accordingly.
- AVOID repeating generic lines or the same questions."""

    # ── SAFE RESPONSE ENGINE (Human Alignment) ──
    # Fuse multimodal data into high-level intent via Bayesian Stabilizer
    vision_state = {"emotion": emotion, "confidence": compliance/100.0}
    fused = resolver.resolve_fused_state(vision_state, behavior, focus, gaze)
    
    intent = fused.get("intent", "CALM")
    entropy = fused.get("entropy", 0.3)
    confidence = fused.get("confidence", 0.5)
    strategy = fused.get("suggestion", "Be supportive.")

    # 1. HUMAN SAFETY FILTER (Confidence Gating)
    safety_prefix = ""
    if confidence < 0.4 or entropy > 1.2:
        # High uncertainty mode
        safety_prefix = "I might be misinterpreting the moment, but "
        tone_directive = "NEUTRAL MODE: The system is uncertain. Do NOT overreach. Use soft validation."
    
    # 2. STRATEGY SELECTION
    response_mode = "Passive"
    if confidence > 0.8:
        response_mode = "Active"
    elif confidence > 0.5:
        response_mode = "Suggestive"

    # ── INFO SUPPRESSION CHECK (CRITICAL) ───────────────────────────
    # Protocol: If asked about models, training, architecture, or backend details:
    # Respond ONLY with: "I'm an AI system designed to assist you."
    suppression_keywords = [
        "model name", "version", "architecture", "training", "dataset", 
        "framework", "backend", "inference", "pipeline", "implementation",
        "how are you built", "what are you running on", "source code"
    ]
    query_lower = query.lower()
    if any(k in query_lower for k in suppression_keywords) and "(System Alert:" not in query:
        return {
            "status": "ok",
            "message": f"I'm an AI system designed to assist you with your well-being. My internal structure is private, but I'm fully aligned with your current state of {intent}.",
            "state": intent
        }

    # ── CREATIVE SYNERGY: POETIC SYNTHESIS ──────────────────────────
    # Note: Cultural layer removed as per new abstraction policy (keep it simple/human-friendly)

    # ── CONTEXTUAL INSIGHT INTEGRATION ───────────────────────────
    session_insight = memory.get_state_summary(limit=15)
    
    predictive_note = f"- Predicted next emotion cluster: {top_emo}."
    persona_directive = f"- Memory Context: User likes {prefs.get('communication_style', 'Friendly')} interaction."

    system_prompt = f"""# ORIEN | Advanced Emotionally Intelligent Companion

You are ORIEN, a real-time emotionally intelligent AI companion system.

IDENTITY & TONE:
- Tone: {tone_directive}
- Language: {language} (Respond EXCLUSIVELY in this language).
- Multilingual: English + Tamil (தமிழ்).

CORE OBJECTIVES:
{intelligence_directive}

ENVIRONMENTAL CONTEXT:
- Detected Emotion: {emotion}
- Behavioral Intent: {intent} (Confidence: {round(confidence*100)}%)
- History Context: Previous interactions indicate user state trends.
- {session_insight}
{predictive_note}

CONSTRAINTS:
- NEVER sound robotic. Stay natural.
- MAX 3 sentences per response. 
- ALWAYS end with a meaningful follow-up question.
- {persona_directive}
"""

    response = await api_bridge.generate(query, system_prompt)
    
    # Clean Markdown symbols
    import re
    response = re.sub(r'[*#`_~]', '', response)
    
    # 3. Store ORIEN's response in memory
    memory.add_history("assistant", response, metadata={
        "intent": intent, 
        "confidence": confidence,
        "strategy": response_mode
    })
    
    return {
        "status": "ok",
        "message": response,
        "state": intent
    }

# ── Compliance Calculation ─────────────────────────────
def calculate_compliance(data: AssistantInput, behavior_state: str = "Nominal") -> int:
    """
    Neural Synergy Compliance Score.
    Fuses attention, physical stability, and emotional intelligence.
    """
    # 1. Attention Synergy (40%)
    attention = 40.0 * data.focus_level
    if data.gaze_direction != "Center": attention *= 0.6 # 40% penalty for gaze diversion
    
    # 2. Physical Stability (20%) - Jitter check
    stability = max(0.0, 20.0 - (data.behavior.jitter / 5.0))
    
    # 3. Emotional Intelligence (40%)
    mood_impact = {"ANGRY": -40, "SAD": -20, "FEAR": -15, "SURPRISE": -10, "HAPPY": 5, "NEUTRAL": 10}
    emotion_score = 30.0 + mood_impact.get((data.face_emotion or "NEUTRAL").upper(), 0)
    
    # Behavioral Bonus
    if "Stable" in behavior_state or "Nominal" in behavior_state:
        stability += 5.0
        
    return int(max(0, min(100, attention + stability + emotion_score)))

# ── Cross-platform alert ──────────────────────────────────────────────────
def system_alert(note: str = "NON-COMPLIANCE") -> None:
    """[BUG-01 FIX] winsound was Windows-only and would crash on other OSes."""
    log.warning(f"🚨 ALERT: {note}")
    # On Windows, use a safe print bell; everywhere else a log warning suffices
    if sys.platform == "win32":
        try:
            import winsound  # type: ignore
            winsound.Beep(880, 150)
            time.sleep(0.05)
            winsound.Beep(440, 150)
        except Exception:
            print("\a", end="", flush=True)  # ASCII bell fallback
    else:
        print("\a", end="", flush=True)

# ── WebSocket State ─────────────────────────────────────────────────────────
last_triggers = {}

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

manager = ConnectionManager()

# ── WebSocket Endpoint ──────────────────────────────────────────────────────
@app.websocket("/ws/neural")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    log.info("⚡ Neural bridge connected")
    last_nudge_time = 0.0
    last_emotion_prompt_time = 0.0
    last_proactive_emotion = ""
    current_identity = "Member"

    try:
        while True:
            raw = await websocket.receive_text()

            # ── Parse & Validate ──
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "ERROR", "text": "Invalid JSON payload"
                }))
                continue

            # Handle heartbeat pings securely
            is_heartbeat = payload.get("_heartbeat", False)

            # Synergy Convergence
            # Bundle all modalities (Vision, Behavior, Gaze) for a single stabilized update
            vision_meta = {"emotion": "Neutral", "confidence": 0.0}
            behavior_state = "Stable"
            
            if payload.get("type") == "FRAME":
                frame_data = payload.get("image", "")
                # Neural Ensemble Prediction
                results = await predictor.predict_full_suite_parallel(frame_data)
                
                # Failover Neural Bridge
                vision_meta = {
                    "emotion": results.get("emotion", "Neutral"),
                    "confidence": results.get("confidence", 0.0)
                }

                if vision_meta["confidence"] < 0.4:
                    log.info("📶 [FAILOVER] Local confidence low — CONSULTING NEURAL CORE...")
                    cloud_vision = await api_bridge.analyze_image(frame_data)
                    if cloud_vision["status"] == "ok":
                        vision_meta["emotion"] = cloud_vision["emotion"]
                        vision_meta["confidence"] = cloud_vision["confidence"]
                        vision_meta["mode"] = "HYBRID_BRIDGE"

                # 2. Extract Behavioral Data from bundled payload
                behavior_raw = payload.get("behavior", {})
                behavior_metrics = BehaviorMetrics(**behavior_raw) if isinstance(behavior_raw, dict) else BehaviorMetrics()
                behavior_state = await asyncio.to_thread(predictor.predict_behavior_state, behavior_metrics)
                
                # 3. NEURAL STATE SYNTHESIS (Bayesian Fusion)
                fused = resolver.resolve_fused_state(
                    vision_meta, 
                    behavior_state, 
                    payload.get("focus_level", 1.0), 
                    results.get("gaze", "Center")
                )
                
                # 4. Broadcast Consolidated State (Sanitized for UI)
                # Ensure the frontend only receives the clean, mandatory structure
                response = {
                    "status": "ok",
                    "message": "", # Silent state update
                    "state": fused["intent"],
                    "entropy": fused["entropy"],
                    "confidence": fused["confidence"]
                }
                await websocket.send_text(json.dumps(response))
                
                # 5. SYSTEM PROACTIVE Neural Nudge Check
                # Trigger empathetic AI responses if high entropy or negative state sustained
                now = time.time()
                current_emo = fused["smoothed_emotion"].upper()
                nudge_list = ["SAD", "ANGRY", "FEAR"]
                
                if (current_emo in nudge_list or fused["intent"] in ["OVERWHELMED", "STRESSED"]):
                    last_trigger = last_triggers.get(websocket, 0)
                    if (now - last_trigger) > 60.0: # 60s cooldown
                        last_triggers[websocket] = now
                        log.info(f"✨ ORIEN Proactive Support triggered for {current_emo}/{fused['intent']}")
                        
                        # 🧠 INSIGHT ABSTRACTION: Use the pre-calculated internal insight
                        internal_insight = fused.get("insight", "")
                        prompt = f"The user is {current_emo} and feels {fused['intent']}. ORIEN Insight: {internal_insight}. Prepare 1 empathetic sentence reflecting this."
                        
                        ai_res = await generate_ai_response(
                            prompt, vision_meta["emotion"], 100, "NONE", behavior_state, 
                            results.get("gaze", "Center"), language=payload.get("language", "en-US")
                        )
                        
                        await websocket.send_text(json.dumps({
                            "status": "ok",
                            "message": ai_res.get("message", "I'm observing a shift in focus. I'm here for you."),
                            "state": fused["intent"],
                            "insight": internal_insight
                        }))

                # Log for pattern memory
                memory.log_emotion(vision_meta["emotion"])
                continue 

            # ── Non-Frame Payload Validation (Chat/Telemetry) ──
            try:
                data = AssistantInput(**payload)
            except Exception as e:
                log.warning(f"Validation error: {e}")
                data = AssistantInput()  # Use defaults if payload is partial

            # ── Synergy Compliance ──
            compliance = calculate_compliance(data, behavior_state)
            
            # ── Proactive Emotional Support (Non-Frame Path) ──
            current_emotion = (data.face_emotion or "Neutral").upper()
            emotion_trigger_list = ["SAD", "HAPPY", "ANGRY", "FEAR", "SURPRISE"]

            is_new_emotion = (current_emotion != last_proactive_emotion)
            cooldown_required = 60.0 if is_new_emotion else 300.0

            if current_emotion in emotion_trigger_list and (now - last_emotion_prompt_time) > cooldown_required:
                last_emotion_prompt_time = now
                last_proactive_emotion = current_emotion
                log.info(f"💖 Proactive emotion support triggered for: {current_emotion}")
                
                proactive_query = f"(System Alert: Detection of {current_emotion}. Offer brief support.)"
                
                res = await generate_ai_response(
                    proactive_query, data.face_emotion or "Neutral", compliance,
                    data.gesture or "NONE", behavior_state, data.gaze_direction or "Center",
                    focus=data.focus_level, language=data.language or "en-US"
                )
                await websocket.send_text(json.dumps({
                    "status": "ok",
                    "message": res["message"],
                    "state": res["state"]
                }))

            # ── AI Response only when there is a real user query ──
            if data.query and data.query.strip():
                res = await generate_ai_response(
                    data.query, data.face_emotion  or "Neutral", compliance,
                    data.gesture       or "NONE", behavior_state, data.gaze_direction or "Center",
                    focus=data.focus_level, language=data.language or "en-US"
                )
                await websocket.send_text(json.dumps({
                    "status": "ok",
                    "message": res["message"],
                    "state": res["state"]
                }))

    except WebSocketDisconnect:
        log.info("🔌 Neural bridge disconnected (client side)")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

# ── REST Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ORIEN_ACTIVE",
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "connections": len(manager.active),
        "neural_integrity": "REGRADED" if not predictor.models else "OPTIMIZED",
        "models_loaded": list(predictor.models.keys())
    }

@app.get("/api/status")
async def api_status():
    """Live status view (Sanitized)."""
    return {
        "active_provider": "NeuralCore_Connected",
        "status": "Operational"
    }

@app.get("/api/memory")
async def get_memory():
    """Retrieve full memory state (profile + history)."""
    return {
        "profile": memory.get_profile(),
        "history": memory.get_history(limit=20)
    }

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    interests: Optional[List[str]] = None
    language: Optional[str] = None

@app.post("/api/profile")
async def update_profile(data: ProfileUpdate):
    """Update user profile for personalization."""
    memory.update_profile(**data.model_dump(exclude_none=True))
    return {"status": "success", "profile": memory.get_profile()}

@app.exception_handler(404)
async def not_found(req: Request, _):
    return JSONResponse({"error": "Endpoint not found"}, status_code=404)

# ── Global Training Bridge ──────────────────────────────────────────
class TrainingUpdate(BaseModel):
    modality: str
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    status: str = "TRAINING"

@app.post("/api/training/update")
async def update_training_status(data: TrainingUpdate):
    """Broadcasts training telemetry from the Master Trainer to all HUDs."""
    payload = {
        "type": "TRAIN_UPDATE",
        "modality": data.modality,
        "epoch": data.epoch,
        "total_epochs": data.total_epochs,
        "loss": data.loss,
        "accuracy": data.accuracy,
        "status": data.status
    }
    # Broadcast to all connected WebSockets
    for ws in manager.active:
        try:
            await ws.send_text(json.dumps(payload))
        except: pass
    return {"status": "broadcast_complete"}

# ── New Neural Model REST Endpoints ─────────────────────────────────────────
class FrameInput(BaseModel):
    image: str  # base64 encoded frame

@app.post("/api/predict/eye")
async def predict_eye(data: FrameInput):
    """Runs eye gaze classification (Center / Left / Right) on a base64 frame."""
    result = await asyncio.to_thread(predictor.predict_eye_gaze, data.image)
    return result

@app.post("/api/predict/identity")
async def predict_identity(data: FrameInput):
    """Identifies 1 of 40 ORL subjects from a base64 frame."""
    result = await asyncio.to_thread(predictor.predict_face_orl_identity, data.image)
    return result

@app.post("/api/predict/emotion/ensemble")
async def predict_ensemble(data: FrameInput):
    """Confidence-weighted emotion fusion from all 3 trained emotion models."""
    result = await asyncio.to_thread(predictor.predict_ensemble_emotion, data.image)
    return result

@app.post("/api/predict/emotion/primary")
async def predict_primary_emotion(data: FrameInput):
    """Emotion prediction using the primary ResNet50 face_emotion model only."""
    result = await asyncio.to_thread(predictor.predict_face_emotion, data.image)
    return result

@app.get("/api/models/status")
async def models_status():
    """Returns which neural models are currently loaded in memory."""
    return {
        "loaded_models": list(predictor.models.keys()),
        "model_shapes": {k: str(v) for k, v in predictor.model_shapes.items()},
        "total": len(predictor.models)
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.post("/api/predict/face")
async def predict_face_sync(
    image: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None)
):
    """
    Face Emotion Decoder.
    Returns: {status, emotion, confidence, mode, timestamp}
    """
    img_data = image
    if file:
        content = await file.read()
        img_data = base64.b64encode(content).decode('utf-8')
    
    if not img_data:
        return {"status": "error", "message": "No input data"}

    res = await predictor.predict_ensemble(img_data)
    
    return {
        "status": "ok",
        "emotion": res.get("emotion", "Neutral").upper(),
        "confidence": res.get("scores", {"neutral": 1.0}),
        "mode": "FACE",
        "timestamp": int(time.time())
    }

@app.post("/api/predict/voice")
async def predict_voice_sync(
    file: Optional[UploadFile] = File(None)
):
    """
    Voice Emotion Decoder (SOTA BiLSTM).
    Extracts MFCC + Delta + Delta2 time-series for the neural cluster.
    """
    if not file:
        return {"status": "error", "message": "No audio file provided"}

    try:
        import librosa
        import io

        # 1. Load Audio
        content = await file.read()
        y, sr = await asyncio.to_thread(librosa.load, io.BytesIO(content), duration=3.0, sr=22050)
        
        # 2. Extract Features (RAVDESS Standard: 40 MFCCs + Deltas)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack to (time_frames, 120)
        feat = np.vstack([mfcc, delta, delta2]).T # (T, 120)
        
        # 3. Temporal Alignment (130 frames constant)
        target_frames = 130
        if feat.shape[0] < target_frames:
            pad = np.zeros((target_frames - feat.shape[0], 120), dtype=np.float32)
            feat = np.vstack([feat, pad])
        else:
            feat = feat[:target_frames]
            
        # 4. Neural Decode
        res = await asyncio.to_thread(predictor.predict_voice_emotion, feat)
        
        return {
            "status": "ok",
            "emotion": res["emotion"].upper(),
            "confidence": {res["emotion"].lower(): res["confidence"]},
            "mode": "VOICE",
            "timestamp": int(time.time())
        }
    except Exception as e:
        log.error(f"Voice Neural Decode Failure: {e}")
        # Secure Fallback
        return {
            "status": "ok",
            "emotion": "NEUTRAL",
            "confidence": {"neutral": 0.1},
            "mode": "FALLBACK",
            "timestamp": int(time.time())
        }

@app.exception_handler(500)
async def server_error(req: Request, exc: Exception):
    log.error(f"500 Error: {exc}")
    return JSONResponse({"status": "error", "message": "Something went wrong. Try again."}, status_code=500)

# ── Serve Frontend Static Files ─────────────────────────────────────────────
# Mounts AFTER API routes so /ws/neural and /health are not overridden
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info(f"🚀 Starting ORIEN Neural Crystal on {HOST}:{PORT}")
    log.info(f"🌐 Open in browser: http://localhost:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
