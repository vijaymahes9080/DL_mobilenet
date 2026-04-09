
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
MODEL_PATH = "models/vmax/emotion_master/emotion_master_optimal.keras"
DATA_PATH = "training/splits/test/emotion/data.npy"
LABEL_PATH = "training/splits/test/emotion/labels.npy"
OUTPUT_DIR = "evaluation_emotion"
IMG_SIZE = 96

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def evaluate():
    print(f"[*] Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("[*] Loading test data...")
    X_test = np.load(DATA_PATH)
    y_test = np.load(LABEL_PATH)
    
    if len(X_test.shape) == 3:
        X_test = np.expand_dims(X_test, axis=-1)
    
    print("[*] Resizing data to 96x96...")
    X_test_processed = []
    for img in X_test:
        img_resized = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
        if img_resized.shape[-1] == 1:
            img_resized = np.concatenate([img_resized]*3, axis=-1)
        X_test_processed.append(img_resized)
    
    X_test = np.array(X_test_processed) / 255.0
    
    print("[*] Running inference...")
    y_probs = model.predict(X_test, batch_size=32)
    y_pred = np.argmax(y_probs, axis=1)
    
    print("[*] Computing metrics...")
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    
    # Check accuracy on only the known classes
    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_test)
    print(f"Accuracy: {accuracy:.4%}")
    
    report = classification_report(y_test, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0)
    print(report)
    
    # Save report
    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write("# Emotion Master Evaluation\n\n")
        f.write(f"**Accuracy**: {accuracy:.2%}\n\n")
        f.write("## Metrics\n```\n" + str(report) + "\n```\n")
        
if __name__ == "__main__":
    evaluate()
