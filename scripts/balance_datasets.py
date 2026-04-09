import os, shutil, random, sys
from pathlib import Path
from collections import Counter

# ⚖️ ORIEN: Neural Data Synergy Balancer
# Resolves BIAS by performing balancing across all splits.
# This ensures structural integrity regardless of how many subfolders exist.

if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except: pass

ROOT = Path(__file__).parent.parent.absolute()
DATASET_ROOT = ROOT / "dataset"

MODALITIES = {
    "face":           "face_core",
    "gesture":        "gesture_hub",
    "voice":          "voice_cloud",
    "behavior":       "behavior_node",
    "eye":            "eye_monitor",
    "face_alt":       "face_alt_(lfw)",
    "face_orl":       "face_orl",
    "emotion_master": "face_emotion"
}

BALANCE_CONFIG = {
    "face":           {"min_total": 40,  "cap_total": 120}, # Global Sync
    "face_alt":       {"min_total": 40,  "cap_total": 120},
    "face_orl":       {"min_total": 20,  "cap_total": 60},
    "gesture":        {"min_total": 400, "cap_total": 1200}, # 3.0 ratio
    "emotion_master": {"min_total": 400, "cap_total": 1200},
    "voice":          {"min_total": 60,  "cap_total": 180},
    "behavior":       {"min_total": 30,  "cap_total": 90},
    "eye":            {"min_total": 150, "cap_total": 450}
}

def balance_modality_global(name, folder):
    data_path = DATASET_ROOT / folder
    if not data_path.exists(): return
    cfg = BALANCE_CONFIG.get(name, {"min_total": 2, "cap_total": 1000})

    exts = ('.jpg', '.jpeg', '.png', '.pgm', '.parquet', '.wav', '.bmp')
    
    # 1. Map all class files globally
    class_files = {}
    for root, _, files in os.walk(data_path):
        valid = [os.path.join(root, f) for f in files if f.lower().endswith(exts)]
        if valid:
            cls_name = os.path.basename(root)
            if cls_name not in class_files: class_files[cls_name] = []
            class_files[cls_name].extend(valid)
    
    if not class_files: return

    print(f"[*] Globally Balancing: {name.upper()}")
    
    total_pruned = 0
    total_capped = 0
    
    for cls, files in class_files.items():
        count = len(files)
        
        # PRUNE: If aggregate count is too low
        if count < cfg["min_total"]:
            # Find and remove all directories associated with this class name
            for f in files:
                try: 
                    os.remove(f)
                    total_pruned += 1
                except: pass
            continue
            
        # CAP: If aggregate count is too high
        if count > cfg["cap_total"]:
            random.shuffle(files)
            to_remove = files[cfg["cap_total"]:]
            for f in to_remove:
                try:
                    os.remove(f)
                    total_capped += 1
                except: pass

    print(f"  [OK] Aggregate Pruned: {total_pruned} | Aggregate Capped: {total_capped}")

def run_balancing():
    print("\n" + "💎"*30)
    print("  ORIEN: NEURAL DATA BALANCER")
    print("  Goal: Eradicate Bias & Stabilize Entropy")
    print("💎"*30 + "\n")
    
    for name, folder in MODALITIES.items():
        balance_modality_global(name, folder)
        
    print("\n[FINISH] Global Synergy Balancing complete.")

if __name__ == "__main__":
    run_balancing()
