import os
import cv2
from pathlib import Path
from tqdm import tqdm

def sanitize_dataset(path):
    print(f"[*] SCRUBBING DATASET: {path}")
    path = Path(path)
    if not path.exists():
        print(f"⚠️  Path not found: {path}")
        return
        
    corrupt_count = 0
    valid_count = 0
    
    # Supported image formats
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    files = [f for f in path.rglob("*") if f.suffix.lower() in exts]
    print(f"[*] Found {len(files)} potential images. Validating integrity...")
    
    for f in tqdm(files):
        try:
            img = cv2.imread(str(f))
            if img is None:
                raise ValueError("Could not decode image")
            valid_count += 1
        except Exception:
            # print(f"  [X] Removing corrupt: {f.name}")
            try:
                os.remove(f)
                corrupt_count += 1
            except: pass
            
    print(f"\n[SUMMARY] Scrubbing Complete.")
    print(f" [OK] Valid Images: {valid_count}")
    print(f" [FAIL] Corrupt Removed: {corrupt_count}")

if __name__ == "__main__":
    # Sanitize all target modalities
    targets = [
        "d:/current project/DL/dataset/gesture",
        "d:/current project/DL/dataset/face",
        "d:/current project/DL/dataset/face_emotion",
        "d:/current project/DL/dataset/eye_monitor"
    ]
    for target in targets:
        sanitize_dataset(target)
