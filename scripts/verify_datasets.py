import os, shutil, sys
from pathlib import Path

# Fix UTF-8 output and ensure scripts work on CP1252 (Windows)
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# 🔍 ORIEN Neural Data Verifier & Scaffolder [Optimized Advanced]
ROOT = Path(__file__).parent.parent.absolute()
DATASET_ROOT = ROOT / "dataset"

# mod_name: folder_name
MODALITIES = {
    "face":           "face/faces",
    "gesture":        "gesture/classes",
    "voice":          "voice_cloud",
    "behavior":       "behavior_node",
    "eye":            "eye_monitor",
    "face_alt":       "face_alt",
    "face_orl":       "face_orl",
    "emotion_master": "face_emotion/train"
}

def create_scaffold(mod_name, path):
    """Creates a minimal directory structure with dummy data to prevent training crashes."""
    print(f"[*] Setting up scaffold for: {mod_name} at {path}")
    path.mkdir(parents=True, exist_ok=True)
    
    if mod_name in ["behavior", "voice"]:
        (path / "readme.txt").write_text(f"Scaffold for {mod_name}")
        return

    # Image-based modalities
    for subset in ["train", "val"]:
        for cls in ["class_low", "class_high"]:
            cls_path = path / subset / cls
            cls_path.mkdir(parents=True, exist_ok=True)
            (cls_path / ".gitkeep").touch()

def check_all():
    print("\n" + "💎"*30)
    print("  ORIEN: NEXT-LEVEL NEURAL DATA AUDIT")
    print("  Status: Checking for Imbalance & Deep Integrity")
    print("💎"*30 + "\n")
    
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    
    report = []
    for mod_name, folder in MODALITIES.items():
        path = DATASET_ROOT / folder
        status = "[READY]"
        count = 0
        imbalance = "STABLE"
        
        if not path.exists():
            status = "[SCFLD] MISSING"
            create_scaffold(mod_name, path)
        else:
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if not subdirs and mod_name not in ["behavior", "voice"]:
                train_path = path / "train"
                if not train_path.exists() or not [d for d in train_path.iterdir() if d.is_dir()]:
                    status = "[SCFLD] EMPTY"
                    create_scaffold(mod_name, path)
            
            # Count items for all modalities with suitable extensions
            exts = ('.jpg', '.jpeg', '.png', '.parquet', '.pgm', '.bmp', '.wav', '.mp3')
            # Aggregate counts by CLASS NAME across all splits (train/val/test)
            class_map = {}
            for root, dirs, files in os.walk(path):
                valid_files = [f for f in files if f.lower().endswith(exts)]
                if valid_files:
                    cls_name = os.path.basename(root)
                    class_map[cls_name] = class_map.get(cls_name, 0) + len(valid_files)
            
            counts = list(class_map.values())
            count = sum(counts)
            
            if counts:
                mx, mn = max(counts), min(counts)
                if mx > mn * 3: imbalance = "⚠️ HIGH BIAS"
                elif mx > mn * 1.5: imbalance = "MODERATE"

        report.append(f"| {mod_name.upper():<16} | {count:<10} | {imbalance:<12} | {status:<15} |")

    header = f"| {'Modality':<16} | {'Items':<10} | {'Imbalance':<12} | {'Status':<15} |"
    print(header)
    print("|" + "-"*18 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|")
    for r in report:
        print(r)
    print("\n[OK] Deep Synergy Audit Complete.")

if __name__ == "__main__":
    check_all()
