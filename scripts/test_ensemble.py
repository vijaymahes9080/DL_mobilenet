import os, sys, time, numpy as np
import tensorflow as tf

# [SYNERGY] UTF-8 Fixed for Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Handle case where stdout doesn't support reconfigure

# 💠 ORIEN: NEURAL SYNERGY TESTER (ALL 8 MODALITIES)
# Verifies model integrity and inference latency.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "vmax")

def test_synergy():
    print("\n" + "="*60 + "\n 💠 ORIEN: NEURAL SYNERGY INTEGRITY TEST\n" + "="*60)
    
    # All 8 Target Modalities
    MODALITIES = ["face", "gesture", "voice", "behavior", "eye", "face_alt", "face_orl", "emotion_master"]
    
    active_mods = []
    failed_mods = []
    
    for mod in MODALITIES:
        # Try .tflite then .keras then .h5
        path = os.path.join(MODELS_DIR, mod, f"{mod}_optimal.tflite")
        if not os.path.exists(path):
            path = os.path.join(MODELS_DIR, mod, f"{mod}_optimal.keras")
        if not os.path.exists(path):
            path = os.path.join(MODELS_DIR, mod, f"{mod}_optimal.h5")
            
        print(f"[*] Testing [{mod.upper()}] path: {mod}/{os.path.basename(path)}")
        
        if not os.path.exists(path):
            print(f"  ❌ Missing: {mod}")
            failed_mods.append(mod)
            continue
            
        try:
            # Quick load check
            m = tf.keras.models.load_model(path, compile=False)
            
            # Dummy inference to test performance
            shape = m.input_shape
            # If shape is [None, sz, sz, 3]
            sz = shape[1] if shape[1] else 96
            
            t0 = time.time()
            if len(shape) == 4:
                dummy = np.random.randn(1, sz, sz, 3).astype('float32')
            else:
                # Handle Non-Vision (Behavior/Voice)
                dummy = np.random.randn(1, *shape[1:]).astype('float32')
                
            model_out = m.predict(dummy, verbose=0)
            latency = (time.time() - t0) * 1000
            
            print(f"  ✅ ACTIVE Latency: {latency:.2f}ms | Load: SUCCESS")
            active_mods.append(mod)
        except Exception as e:
            print(f"  ❌ CRITICAL ERROR IN NEURAL CORE: {e}")
            failed_mods.append(mod)

    print("\n" + "="*60)
    print(f" 📊 Final Neural Report")
    print(f"   Active Nodes: {len(active_mods)}/8")
    print(f"   Offline/Missing: {len(failed_mods)}")
    print("="*60)
    
    if len(active_mods) == 8:
        print("\n🏆 NEURAL SYNERGY ESTABLISHED. ORIEN IS FULLY COMPLIANT.")
    else:
        print("\n⚠️  Synergy Incomplete. Run 'MASTER_EXECUTE.bat' to rebuild missing nodes.")

if __name__ == "__main__":
    test_synergy()
