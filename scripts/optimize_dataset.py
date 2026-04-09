import os, sys, shutil, concurrent.futures
from pathlib import Path
from PIL import Image
from threading import Lock

# 🛠️ ORIEN: Neural Data Optimizer
ROOT = Path(__file__).parent.parent.absolute()
DATASET_ROOT = ROOT / "dataset"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".pgm")

# Stats lock
stats_lock = Lock()
count_valid = 0
count_dead = 0

def process_file(fp_str):
    global count_valid, count_dead
    f = os.path.basename(fp_str)
    # Process if it has an allowed extension or NO extension
    is_valid_ext = f.lower().endswith(EXTS)
    has_no_ext = "." not in f
    
    if is_valid_ext or (has_no_ext and os.path.isfile(fp_str)):
        try:
            do_remove = False
            fp_abs = os.path.abspath(fp_str)
            img = Image.open(fp_abs)
            try:
                # [FIX] Force re-save to fix TF decode_image errors (header issues)
                if has_no_ext:
                    new_fp = fp_abs + '.jpg'
                elif f.lower().endswith('.pgm') or f.lower().endswith('.bmp'):
                    new_fp = os.path.splitext(fp_abs)[0] + '.jpg'
                else:
                    new_fp = fp_abs
                    
                img_rgb = img.convert('RGB')
                img_rgb.save(new_fp, 'JPEG', quality=95)
                
                if new_fp != fp_abs:
                    do_remove = True
            finally:
                img.close()
            
            if do_remove:
                try:
                    os.remove(fp_abs)
                except OSError:
                    pass # Still locked? Skip for now.
            
            with stats_lock:
                count_valid += 1
        except Exception as e:
            # Only print if not too many errors, otherwise it floods
            # print(f"  [DEAD] {f} - Purging. ({e})")
            try: 
                os.remove(os.path.abspath(fp_str))
            except: pass
            with stats_lock:
                count_dead += 1

def clean_modality(name, path):
    global count_valid, count_dead
    print(f"[*] Optimizing: {name.upper()} at {path}")
    if not path.exists():
        print(f"  [SKIP] Dir not found: {path}")
        return

    count_valid = 0
    count_dead = 0
    
    file_list = []
    for root, _, files in os.walk(str(path)):
        for f in files:
            file_list.append(os.path.join(root, f))
    
    # Use ThreadPoolExecutor for heavy IO re-encoding
    # Lowered thread count to prevent system-wide OOM crashes
    max_workers = min(8, (os.cpu_count() or 4) * 1)
    print(f"  [SYNC] Launching {max_workers} neural worker threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, file_list)

    print(f"  [OK] Valid & Re-encoded: {count_valid} | [FAIL] Purged: {count_dead}")

    # Remove empty directories
    for root, dirs, _ in os.walk(path, topdown=False):
        for d in dirs:
            dp = Path(root) / d
            try:
                # If directory is empty, remove it
                if not any(dp.iterdir()):
                    os.rmdir(dp)
            except: pass

def run_optimization():
    print("\n" + "="*60)
    print(" ORIEN: Neural Data Synergy Optimization")
    print("="*60)
    # Using the standardized modality-to-folder mapping from local_trainer
    paths = { 
        "face": "face_core", 
        "gesture": "gesture_hub", 
        "voice": "voice_cloud", 
        "behavior": "behavior_node", 
        "eye": "eye_monitor", 
        "face_alt": "face_alt_(lfw)", 
        "face_orl": "face_orl", 
        "emotion_master": "emotion_master"
    }
    
    for name, folder in paths.items():
        # Optimization: also clean 'face_emotion' and 'voice' if they exist separately
        clean_modality(name, DATASET_ROOT / folder)
        if name == "emotion_master": clean_modality("face_emotion", DATASET_ROOT / "face_emotion")
        if name == "voice": clean_modality("voice_legacy", DATASET_ROOT / "voice")
    
    print("\n[FINISH] Dataset optimization complete.")

if __name__ == "__main__":
    run_optimization()
