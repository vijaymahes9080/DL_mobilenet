
import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Fix for Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Constants
MODEL_PATH = "models/vmax/face/face_optimal.keras"
CLASSES_PATH = "models/vmax/face/classes.json"
TEST_DATA_PATH = "training/splits/test/vision"
OUTPUT_DIR = "evaluation_vision"
IMG_SIZE = 96

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def evaluate():
    print(f"[*] Loading vision model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(CLASSES_PATH, 'r') as f:
        target_classes = json.load(f)
    print(f"[INFO] Model expects {len(target_classes)} classes.")
    
    # Filter test directories to only those in target_classes
    test_dirs = [d for d in os.listdir(TEST_DATA_PATH) if d in target_classes]
    print(f"[INFO] Found {len(test_dirs)} matching target classes in test directory.")
    
    if not test_dirs:
        print("[ERROR] No matching classes found in test data.")
        return

    # Load images
    X_test = []
    y_test = []
    
    class_to_idx = {name: i for i, name in enumerate(target_classes)}
    
    print("[*] Loading and preprocessing matching images...")
    for class_name in test_dirs:
        class_path = os.path.join(TEST_DATA_PATH, class_name)
        label = class_to_idx[class_name]
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.utils.img_to_array(img)
                # Note: face_optimal.keras summary might reveal if rescaling is internal.
                # Assuming no internal rescaling for now (safe bet for MobilenetV2 if standard).
                # Actually local_trainer.py has Rescaling layer.
                X_test.append(img_array)
                y_test.append(label)
            except:
                continue
                
    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)
    
    print(f"[INFO] Final test set: {X_test.shape}")
    
    print("[*] Running inference...")
    y_probs = model.predict(X_test, batch_size=32)
    y_pred = np.argmax(y_probs, axis=1)
    
    accuracy = np.mean(y_pred == y_test)
    print(f"[METRIC] Accuracy on matched classes: {accuracy:.4%}")
    
    report = classification_report(y_test, y_pred, target_names=[target_classes[i] for i in np.unique(y_test)], zero_division=0)
    print(report)
    
    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write("# Vision Model Filtered Evaluation\n\n")
        f.write(f"**Target Accuracy hit**: {accuracy:.2%}\n")
        f.write(f"**Test Samples**: {len(X_test)}\n\n")
        f.write("## Detailed Metrics\n```\n" + report + "\n```\n")

if __name__ == "__main__":
    evaluate()
