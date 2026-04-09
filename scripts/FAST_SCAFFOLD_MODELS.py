import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# 🧬 ORIEN | NEURAL SKELETON Deployment [Optimized RECOVERY]
# ===============================================
# Generates "Skeleton" models for ORIEN to allow immediate logic-testing 
# when datasets or pre-trained weights are missing.
#
# These are valid architecture-compliant models with random initial weights.
# ───────────────────────────────────────────────

def generate_skeleton():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
    except ImportError:
        print("❌ Error: TensorFlow is required for skeleton genesis.")
        return

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT, 'models', 'vmax')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # config: { modality_name: (input_shape, output_classes) }
    SKELETONS = {
        "face":           ((128, 128, 3), 7),
        "face_alt":       ((128, 128, 3), 7),
        "emotion_master": ((128, 128, 3), 7),
        "eye":            ((128, 128, 3), 3),
        "face_orl":       ((128, 128, 3), 40),
        "gesture":        ((128, 128, 3), 2),
        "behavior":       ((14,), 3), # 3 states: Nominal, Stressed, Highly Anomalous
        "voice":          ((128,), 7), # 7 standard emotions
    }

    print("\n" + "💎"*30)
    print("  ORIEN: System NEURAL Deployment")
    print("  Status: Building Neural Skeletons [Compliant]")
    print("💎"*30 + "\n")

    V0_DIR = os.path.join(ROOT, 'models', 'v0')

    for mod, (ishape, ncls) in SKELETONS.items():
        out_file = os.path.join(MODELS_DIR, mod, f"{mod}_optimal.keras")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        print(f"[*] Deploying: {mod.upper()} Hub...")
        
        # Check for V0 seeds
        seed_path = None
        if mod == "gesture": seed_path = os.path.join(V0_DIR, "gesture_base_v0.h5")
        elif mod == "face_emotion" or mod == "emotion_master": seed_path = os.path.join(V0_DIR, "face_emotion_base_v0.h5")
        elif mod == "face": seed_path = os.path.join(V0_DIR, "face_base_v0.h5")

        if seed_path and os.path.exists(seed_path):
            print(f"    [SEED] Loading high-fidelity weights from v0...")
            try:
                m = tf.keras.models.load_model(seed_path)
            except Exception as e:
                print(f"    [WARN] Seed load failed: {e}. Falling back to skeleton.")
                m = None
        else:
            m = None

        if m is None:
            # Build a minimal, compliant architecture
            inp = layers.Input(shape=ishape)
            if len(ishape) > 1:
                x = layers.Conv2D(8, (3,3), activation='relu')(inp)
                x = layers.GlobalAveragePooling2D()(x)
            else:
                x = layers.Dense(32, activation='relu')(inp)
            x = layers.Dense(16, activation='relu')(x)
            act = 'softmax' if ncls > 1 else 'sigmoid'
            out = layers.Dense(ncls, activation=act)(x)
            m = Model(inp, out)
            m.compile(optimizer='adam', loss='categorical_crossentropy')
        
        m.save(out_file)
        
        # Save dummy classes.json if needed
        classes = []
        if mod == "eye": classes = ["Center", "Left", "Right"]
        elif "face" in mod or mod == "emotion_master": classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        elif mod == "gesture": classes = ["call", "dislike"]
        elif mod == "behavior": classes = ["Nominal", "Stressed", "Highly Anomalous"]
        elif mod == "voice": classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        else: classes = [f"Class_{i}" for i in range(ncls)]
        
        with open(os.path.join(os.path.dirname(out_file), "classes.json"), 'w') as f:
            json.dump(classes, f)

    print("\n✅ NEURAL SKELETONS INITIALIZED.")
    print(f"   Root: {MODELS_DIR}")
    print("   [INFO] ORIEN is now REGRADED but operational.")
    print("   Run 'scripts/train_all_modalities.py' to refine these weights with real data.\n")

if __name__ == "__main__":
    generate_skeleton()
