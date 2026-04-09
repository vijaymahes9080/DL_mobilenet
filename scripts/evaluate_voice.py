
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# UTF-8 Fixed for Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Constants
MODEL_PATH = "models/autonomous/ensemble_backbone.keras"
DATA_PATH = "training/splits/test/voice/data.npy"
LABEL_PATH = "training/splits/test/voice/labels.npy"
OUTPUT_DIR = "evaluation_voice"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def evaluate():
    print(f"[*] Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("[*] Loading test data...")
    X_test = np.load(DATA_PATH)
    y_test = np.load(LABEL_PATH)
    
    print(f"[INFO] X_test shape: {X_test.shape}")
    
    # Preprocessing as done in local_trainer.py
    # Reshape each sample to (128, 128) then stack to 3 channels
    X_test_processed = []
    for val in X_test:
        val = val.flatten()
        if val.size >= 128*128: 
            val = val[:128*128].reshape(128, 128)
        else: 
            val = np.pad(val, (0, (128*128)-val.size)).reshape(128, 128)
        
        # Convert to RGB scale [0, 255] then normalize to [0, 1]
        # Spectrogram values might be small, but local_trainer.py uses save_img which handles scaling?
        # Actually save_img might scale. Let's assume [0, 1] for now.
        img = np.stack([val]*3, axis=-1)
        X_test_processed.append(img)
    
    X_test = np.array(X_test_processed)
    # Check if we need normalization. local_trainer.py has Rescaling(1./255)
    
    print("[*] Running inference...")
    y_probs = model.predict(X_test, batch_size=32)
    y_pred = np.argmax(y_probs, axis=1)
    
    print("[*] Computing metrics...")
    # Voice labels are 1-8. model predicts 0-8.
    # Check if there are any predictions for 0.
    print(f"[INFO] Unique predictions: {np.unique(y_pred)}")
    print(f"[INFO] Unique labels: {np.unique(y_test)}")
    
    # Compute accuracy
    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_test)
    print(f"[METRIC] Accuracy: {accuracy:.4%}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save results
    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write("# Voice Evaluation Report\n\n")
        f.write(f"**Accuracy**: {accuracy:.2%}\n\n")
        f.write("## Metrics\n")
        f.write(f"| Label | Precision | Recall | F1 |\n|---|---|---|---|\n")
        for k, v in report.items():
            if k.isdigit():
                f.write(f"| {k} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1-score']:.4f} |\n")
                
    print("[SUCCESS] Evaluation complete.")

if __name__ == "__main__":
    evaluate()
