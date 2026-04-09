"""
ORIEN — Voice MFCC PNG Unpacker
================================
Reads RAVDESS .wav files from dataset/voice/Actor_*/
Extracts MFCC spectrograms and saves as PNG into dataset/voice_cloud/classes/{label}/
Labels 0-7 map to: neutral, calm, happy, sad, angry, fearful, disgust, surprised
(RAVDESS emotion code in filename segment [2], 1-indexed)
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.absolute()
VOICE_DIR = ROOT / "dataset" / "voice"
OUT_DIR   = ROOT / "dataset" / "voice_cloud" / "classes"

EMOTION_LABELS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

MFCC_N = 40
DURATION = 3
SR = 22050
IMG_W = 128
IMG_H = 128

def extract_mfcc_image(wav_path):
    """Extract MFCC and convert to RGB image array (128x128x3)."""
    import librosa
    from PIL import Image
    y, sr = librosa.load(str(wav_path), duration=DURATION, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N)
    # Normalize to 0-255
    mfcc_min, mfcc_max = mfcc.min(), mfcc.max()
    if mfcc_max - mfcc_min < 1e-6:
        mfcc_norm = np.zeros_like(mfcc, dtype=np.uint8)
    else:
        mfcc_norm = ((mfcc - mfcc_min) / (mfcc_max - mfcc_min) * 255).astype(np.uint8)
    # Resize to 128x128
    img = Image.fromarray(mfcc_norm).resize((IMG_W, IMG_H), resample=2)  # 2=BICUBIC
    img_rgb = img.convert("RGB")
    return img_rgb

def unpack():
    if not VOICE_DIR.exists():
        print(f"[ERROR] Voice source not found: {VOICE_DIR}")
        return

    actors = sorted([d for d in VOICE_DIR.iterdir() if d.is_dir() and d.name.startswith("Actor_")])
    if not actors:
        print("[ERROR] No Actor_ folders found in dataset/voice/")
        return

    print(f"[INFO] Found {len(actors)} actor folders.")

    # Pre-create class output dirs (0-indexed labels = emotion_code - 1)
    for i in range(8):
        (OUT_DIR / str(i).zfill(2)).mkdir(parents=True, exist_ok=True)

    total = 0
    errors = 0
    skipped = 0

    for actor in actors:
        wavs = sorted(actor.glob("*.wav"))
        for wav in wavs:
            parts = wav.stem.split("-")
            if len(parts) < 3:
                skipped += 1
                continue
            try:
                emo_code = int(parts[2])  # 1-indexed emotion code
            except ValueError:
                skipped += 1
                continue

            label_idx = emo_code - 1  # 0-indexed
            if label_idx < 0 or label_idx > 7:
                skipped += 1
                continue

            out_class_dir = OUT_DIR / str(label_idx).zfill(2)
            out_img_path = out_class_dir / f"{wav.stem}.png"

            if out_img_path.exists():
                skipped += 1
                continue

            try:
                img = extract_mfcc_image(wav)
                img.save(str(out_img_path))
                total += 1
            except Exception as e:
                errors += 1
                print(f"  [WARN] {wav.name}: {e}")

        print(f"  [OK] {actor.name} processed.")

    print(f"\n[DONE] Unpacked {total} MFCC PNGs | Skipped {skipped} | Errors {errors}")
    print(f"  Output: {OUT_DIR}")

    # Summary: count per class
    print("\n  Class Distribution:")
    for i in range(8):
        d = OUT_DIR / str(i).zfill(2)
        count = len(list(d.glob("*.png")))
        label = EMOTION_LABELS.get(i + 1, "unknown")
        print(f"    Class {i:02d} ({label:10s}): {count} images")

if __name__ == "__main__":
    print("=" * 60)
    print("  ORIEN Voice MFCC Unpacker")
    print("=" * 60)
    try:
        import librosa
        from PIL import Image
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("  Install with: pip install librosa Pillow")
        sys.exit(1)
    unpack()
