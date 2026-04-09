import cv2
import os
import time
from pathlib import Path

# ── ORIEN | REAL-TIME DATASET COLLECTOR ─────────────────────────────
# Allows the user to capture real-time classes via webcam.
# Controls:
#  - [0-9]: Select Class Index
#  - [SPACE]: Capture Image
#  - [Q]: Quit
# ───────────────────────────────────────────────────────────────────

def collect_data(base_path="dataset/face_emotion"):
    base_path = Path(base_path)
    classes = ["class_low", "class_high", "class_neutral", "class_surprise"]
    
    # Ensure directories exist
    for cls in classes:
        (base_path / "train" / cls).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    current_cls_idx = 0
    
    print("\n" + "="*50)
    print("  ORIEN: INTERACTIVE CAMERA COLLECTOR")
    print("="*50)
    print(f"  Target: {base_path}")
    print(f"  Classes: {classes}")
    print("  Controls:")
    print("    [0-3] : Switch Class")
    print("    [SPACE]: Capture Frame")
    print("    [Q]    : Finish Collection")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        cv2.putText(display, f"ACTIVE CLASS: {classes[current_cls_idx]}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("ORIEN Collector", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('3'):
            current_cls_idx = int(chr(key))
            print(f"[*] Switched to: {classes[current_cls_idx]}")
        elif key == ord(' '):
            cls_name = classes[current_cls_idx]
            timestamp = int(time.time() * 1000)
            fname = f"capture_{timestamp}.png"
            fpath = base_path / "train" / cls_name / fname
            cv2.imwrite(str(fpath), frame)
            print(f"[+] Saved: {fpath.name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
