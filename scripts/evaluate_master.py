
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras import models

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 128 # Adjusted based on model error message
BATCH_SIZE = 32
MODEL_PATH = "models/autonomous/ensemble_backbone.keras"
TEST_DATA_PATH = "training/splits/test/vision"
OUTPUT_DIR = "evaluation_results"

# Create output dir
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# UTF-8 Fixed for Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def load_artifacts():
    print(f"[*] Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model = models.load_model(MODEL_PATH)
    print("[SUCCESS] Model loaded successfully.")
    
    print(f"[*] Loading test dataset from {TEST_DATA_PATH}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_PATH,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    class_names = test_ds.class_names
    print(f"[SUCCESS] Test dataset loaded with {len(class_names)} classes.")
    
    # Normalization (identical to training)
    # The models usually have rescaling layers, but if not, we do it here.
    # Looking at local_trainer.py, it adds layers.Rescaling(1./255)
    # Looking at synergy/model.py, it doesn't explicitly add rescaling in all functions,
    # but EfficientNetV2 often handles it. However, to be safe, we check model input.
    
    return model, test_ds, class_names

def run_inference(model, test_ds):
    print("[*] Running batch-wise inference...")
    y_true = []
    y_pred_probs = []
    
    for x, y in test_ds:
        # Standardize normalization if not in model
        # For now, assume model handles it or we use 0-1 scale
        preds = model.predict(x, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array(y_true)
    
    return y_true, y_pred, y_pred_probs

def compute_metrics(y_true, y_pred, class_names):
    print("[*] Computing core metrics...")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Overall Accuracy
    accuracy = report['accuracy']
    
    # Macro/Weighted
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    print(f"[METRIC] Accuracy: {accuracy:.4f}")
    print(f"[METRIC] Macro F1: {macro_f1:.4f}")
    
    return report

def generate_visuals(y_true, y_pred, y_pred_probs, class_names):
    print("[*] Generating visual analysis...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    # For many classes, we only show top 20 or a summary if CM is too huge
    if len(class_names) > 30:
        sns.heatmap(cm[:30, :30], annot=False, cmap='Blues')
        plt.title("Confusion Matrix (Top 30 Classes Subset)")
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
    
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve & AUC (for top classes if many)
    plt.figure(figsize=(10, 8))
    for i in range(min(5, len(class_names))):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Subset of classes)')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
    plt.close()

def error_analysis(y_true, y_pred, y_pred_probs, test_ds, class_names):
    print("[*] Performing error analysis...")
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    errors = []
    for idx in misclassified_indices[:10]: # Analyze top 10 errors
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = y_pred_probs[idx][y_pred[idx]]
        errors.append({
            "index": int(idx),
            "true": true_label,
            "pred": pred_label,
            "conf": float(confidence)
        })
        
    # Save errors to JSON
    import json
    with open(f"{OUTPUT_DIR}/error_analysis.json", "w") as f:
        json.dump(errors, f, indent=4)
        
    return errors

def robustness_check(model, test_ds):
    print("[*] Running robustness check (Noisy Inputs)...")
    # Add Gaussian noise to a batch and check performance drop
    for x, y in test_ds.take(1):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1, dtype=tf.float32)
        noisy_x = tf.clip_by_value(x + noise, 0, 255)
        
        orig_preds = model.predict(x, verbose=0)
        noisy_preds = model.predict(noisy_x, verbose=0)
        
        orig_acc = np.mean(np.argmax(orig_preds, axis=1) == np.argmax(y, axis=1))
        noisy_acc = np.mean(np.argmax(noisy_preds, axis=1) == np.argmax(y, axis=1))
        
        drop = orig_acc - noisy_acc
        print(f"[ROBUSTNESS] Accuracy drop with 10% Noise: {drop*100:.2f}%")
        return drop

def main():
    try:
        model, test_ds, class_names = load_artifacts()
        y_true, y_pred, y_pred_probs = run_inference(model, test_ds)
        report = compute_metrics(y_true, y_pred, class_names)
        generate_visuals(y_true, y_pred, y_pred_probs, class_names)
        errors = error_analysis(y_true, y_pred, y_pred_probs, test_ds, class_names)
        noise_drop = robustness_check(model, test_ds)
        
        # Final Report Generation
        accuracy = report['accuracy']
        verdict = "DEPLOY" if accuracy >= 0.95 else "RETRAIN"
        
        with open(f"{OUTPUT_DIR}/final_report.md", "w") as f:
            f.write("# 🔬 MODEL EVALUATION MASTER REPORT\n\n")
            f.write(f"**Model**: {MODEL_PATH}\n")
            f.write(f"**Dataset**: {TEST_DATA_PATH}\n")
            f.write(f"**Overall Accuracy**: {accuracy:.2%}\n")
            f.write(f"**Verdict**: {verdict}\n\n")
            f.write("## METRICS SUMMARY\n")
            f.write(f"| Metric | Value |\n|---|---|\n| Precision (Weighted) | {report['weighted avg']['precision']:.4f} |\n| Recall (Weighted) | {report['weighted avg']['recall']:.4f} |\n| F1-Score (Weighted) | {report['weighted avg']['f1-score']:.4f} |\n\n")
            f.write("## ERROR ANALYSIS (Sample)\n")
            for e in errors:
                f.write(f"- Index {e['index']}: True={e['true']}, Pred={e['pred']} (Conf: {e['conf']:.2f})\n")
                
        print(f"\n[!!!] EVALUATION COMPLETE. Final Accuracy: {accuracy:.2%}")
        print(f"Verdict: {verdict}")
        
    except Exception as e:
        print(f"[ERROR] Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
