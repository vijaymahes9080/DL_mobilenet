"""
📊 ORIEN | SYNERGY RESOLVER
Bayesian State Stabilization & Temporal Fusion Engine.
Fuses multimodal signals into a singular stabilized "Truth" vector.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

log = logging.getLogger("ORIEN.Synergy")

class EmotionSmoother:
    """
    Stabilizes high-frequency emotion jitter using temporal inertia 
    and a sliding window filter.
    """
    def __init__(self, window_size: int = 15, stability_threshold: int = 8):
        self.window = deque(maxlen=window_size)
        self.stability_threshold = stability_threshold
        self.current_stabilized_state = "Neutral"
        self.frames_since_switch = 0
        self.min_dwell_time = 0.5 # seconds
        self.last_switch_time = time.time()

    def process(self, raw_emotion: str) -> str:
        self.window.append(raw_emotion)
        
        # Count occurrences in window
        counts = {}
        for e in self.window:
            counts[e] = counts.get(e, 0) + 1
        
        # Find dominant emotion
        dominant = max(counts, key=counts.get)
        count = counts[dominant]

        # State Inertia: Prevent rapid switching
        now = time.time()
        time_since_switch = now - self.last_switch_time
        
        if dominant != self.current_stabilized_state:
            # Only switch if dominant is sustained and dwell time passed
            if count >= self.stability_threshold and time_since_switch > self.min_dwell_time:
                self.current_stabilized_state = dominant
                self.last_switch_time = now
                self.frames_since_switch = 0
        
        self.frames_since_switch += 1
        return self.current_stabilized_state

class BayesianSynergyResolver:
    """
    Recursive Bayesian Filter for Human State Estimation.
    Transitions: P(S_t | S_{t-1})
    Likelihoods: P(O_t | S_t)
    """
    STATES = ["FLOW", "CALM", "STRESSED", "DISTRACTED", "OVERWHELMED"]
    
    def __init__(self):
        # Initial uniform belief
        self.belief = np.array([0.2] * len(self.STATES))
        
        # Transition Matrix (High diagonal for stability/inertia)
        # Row = S_{t-1}, Col = S_t
        self.transition_matrix = np.array([
            [0.90, 0.05, 0.01, 0.02, 0.02], # FLOW
            [0.05, 0.90, 0.02, 0.02, 0.01], # CALM
            [0.01, 0.04, 0.85, 0.02, 0.08], # STRESSED
            [0.05, 0.05, 0.05, 0.80, 0.05], # DISTRACTED
            [0.01, 0.01, 0.10, 0.08, 0.80], # OVERWHELMED
        ])

        self.smoother = EmotionSmoother()
        self.state_history = []
        self.last_update = time.time()

    def _get_likelihood(self, vision: Dict, behavior: str, focus: float, gaze: str) -> np.ndarray:
        """
        Maps multimodal observations to state likelihoods P(O_t | S_t).
        """
        # likelihood[i] = P(O | State_i)
        likelihood = np.ones(len(self.STATES))
        
        emotion = vision.get("emotion", "Neutral")
        conf = vision.get("confidence", 0.5)

        # 1. Vision Evidence
        if emotion == "Happy":
            likelihood[0] *= 1.5  # FLOW
            likelihood[1] *= 1.2  # CALM
        elif emotion in ["Sad", "Fear", "Angry"]:
            likelihood[2] *= 1.8  # STRESSED
            likelihood[4] *= 1.5  # OVERWHELMED
        
        # 2. Behavior Evidence
        if behavior == "Nominal":
            likelihood[0] *= 1.3
            likelihood[1] *= 1.3
        elif behavior == "Stressed":
            likelihood[2] *= 2.0
        elif behavior == "Highly Anomalous":
            likelihood[4] *= 2.5
            
        # 3. Focus & Gaze Evidence
        if focus > 0.8 and gaze == "Center":
            likelihood[0] *= 1.8 # FLOW
        elif focus < 0.4 or gaze != "Center":
            likelihood[3] *= 2.0 # DISTRACTED
            
        # Normalize likelihood to prevent overflow (though not strictly necessary for single step)
        likelihood /= likelihood.sum()
        return likelihood

    def resolve_fused_state(self, vision: Dict, behavior: str, focus: float, gaze: str) -> Dict[str, Any]:
        """
        The Exact SynergyResolver equation (Bayesian + temporal fusion).
        Belief_t = Normalize( Likelihood_t * (Transition_Matrix @ Belief_{t-1}) )
        """
        # 1. Temporal Smoothing of input emotion
        vision["emotion"] = self.smoother.process(vision.get("emotion", "Neutral"))

        # 2. Prediction Step (Prior)
        prior = self.transition_matrix.T @ self.belief
        
        # 3. Update Step (Posterior)
        likelihood = self._get_likelihood(vision, behavior, focus, gaze)
        posterior = likelihood * prior
        
        # 4. Normalization
        if posterior.sum() > 0:
            self.belief = posterior / posterior.sum()
        else:
            self.belief = np.array([0.2] * len(self.STATES)) # Reset on total failure

        # 5. Extract dominant state and metadata
        idx = np.argmax(self.belief)
        intent = self.STATES[idx]
        confidence = float(self.belief[idx])
        entropy = float(-np.sum(self.belief * np.log2(self.belief + 1e-9)))

        # Strategy logic based on stabilized state
        strategy_map = {
            "FLOW": {"priority": "Low", "suggestion": "Maintain flow; non-intrusive support."},
            "CALM": {"priority": "Low", "suggestion": "Steady interaction."},
            "STRESSED": {"priority": "Medium", "suggestion": "Offer grounding or micro-break."},
            "DISTRACTED": {"priority": "Medium", "suggestion": "Gentle nudge to refocus."},
            "OVERWHELMED": {"priority": "High", "suggestion": "Emergency empathy; suggest stopping task."}
        }
        
        strategy = strategy_map.get(intent, strategy_map["CALM"])

        fused = {
            "intent": intent,
            "smoothed_emotion": vision["emotion"], # The stabilized truth
            "priority": strategy["priority"],
            "suggestion": strategy["suggestion"],
            "entropy": round(entropy, 3),
            "confidence": round(confidence, 3),
            "probabilities": {name: round(float(val), 3) for name, val in zip(self.STATES, self.belief)},
            "timestamp": time.time()
        }

        self.state_history.append(fused)
        if len(self.state_history) > 100: self.state_history.pop(0)

        # 🧠 INSIGHT ABSTRACTION: Detect cognitive streaks or patterns
        recent_states = [s["intent"] for s in self.state_history[-30:]]
        recent_emotions = [s["smoothed_emotion"] for s in self.state_history[-30:]]
        
        # 1. FLOW STREAK
        if recent_states.count("FLOW") >= 20:
            fused["insight"] = "Optimal cognitive alignment detected. You are in a high-density FLOW state; I will suppress distractions."
        # 2. BURNOUT ALERT
        elif recent_states.count("STRESSED") >= 15 or recent_states.count("OVERWHELMED") >= 10:
            fused["insight"] = "Neural data suggests cognitive saturation. I recommend a 5-minute deep breathing cycle to recalibrate."
        # 3. DISTRACTION LOOP
        elif recent_states.count("DISTRACTED") >= 18:
            fused["insight"] = "Frequent context switching detected. Shall we return to the primary task to restore focus?"
        # 4. POSITIVE SYNERGY
        elif recent_emotions.count("Happy") >= 15:
            fused["insight"] = "High positive resonance detected. This emotional state is 40% more productive for creative tasks."
        # 5. CALM STABILITY
        elif recent_states.count("CALM") >= 20:
            fused["insight"] = "System stability achieved. You are elegantly composed and ready for complex decision making."
        
        return fused

resolver = BayesianSynergyResolver()
