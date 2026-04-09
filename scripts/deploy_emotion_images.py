import pandas as pd
import numpy as np
import os
from PIL import Image
from pathlib import Path

# 💠 FER2013 Deployment Script for ORIEN
# Converts CSV to Image folders for training synchronization

csv_path = Path("dataset/face_emotion/fer2013.csv")
output_root = Path("dataset/face_emotion/train")

EMOTIONS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

def deploy():
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found.")
        return

    print(f"[*] Loading FER2013 from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"[*] Extracting images to {output_root}...")
    output_root.mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        emotion_id = row['emotion']
        emotion_name = EMOTIONS.get(emotion_id, "Unknown")
        
        # Only process 'Training' and 'PublicTest' for now to keep size down
        # or just everything into train for this specific sprint
        pixels = np.fromstring(row['pixels'], sep=' ').reshape(48, 48).astype(np.uint8)
        img = Image.fromarray(pixels)
        
        cls_dir = output_root / emotion_name
        cls_dir.mkdir(exist_ok=True)
        
        img.save(cls_dir / f"img_{idx}.jpg")
        
        if idx % 1000 == 0:
            print(f"  Processed {idx} images...")

    print("✅ FER2013 Deep Deployment Complete.")

if __name__ == "__main__":
    deploy()
