import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef, log_loss
)
from sklearn.preprocessing import LabelBinarizer
import logging

def optimize_hardware():
    """
    10x Speed Multiplier: Configures hardware acceleration for CPU and GPU.
    Enables XLA, oneDNN, and optimized memory growth.
    """
    # 1. Environment-level CPU Boosts
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
    
    # 2. TensorFlow Optimizations
    try:
        import tensorflow as tf
        # Enable XLA (Accelerated Linear Algebra)
        tf.config.optimizer.set_jit(True)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Mixed Precision for GPU
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logging.info("Mixed precision policy set to mixed_float16")
            except:
                pass
            return "GPU_ACCELERATED"
        else:
            # Optimized Threading for CPU (Ryzen 7 7730U has 16 threads)
            tf.config.threading.set_intra_op_parallelism_threads(16)
            tf.config.threading.set_inter_op_parallelism_threads(4)
            # Force XLA for CPU
            tf.config.optimizer.set_jit(True)
            return "CPU_XLA_MAXIMIZED"
    except Exception:
        pass
    
    # 3. PyTorch Optimizations (YOLO)
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
        
    return "STABLE_DEFAULT"

def setup_gpu():
    """Legacy wrapper for compatibility."""
    return optimize_hardware()

def calculate_mastery_score(metrics):
    """
    Calculates a unified Mastery Score (0-100) based on weighted metrics.
    Weights: Accuracy (40%), F1-Macro (30%), AUC-ROC (20%), MCC (10%).
    """
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_macro', 0)
    auc = metrics.get('auc_roc', 0.5) # Default to 0.5 for random if missing
    mcc = (metrics.get('mcc', 0) + 1) / 2 # Scale MCC from [-1, 1] to [0, 1]
    
    mastery = (acc * 0.4 + f1 * 0.3 + auc * 0.2 + mcc * 0.1) * 100
    return mastery

def compute_all_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """
    Computes a comprehensive suite of metrics for multi-class classification.
    """
    metrics = {}
    
    # Core Metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Specificity calculation (Macro-averaged)
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    metrics['specificity_macro'] = np.mean(specificities)
    
    # Metrics requiring probabilities
    if y_probs is not None:
        try:
            # Multi-class AUC ROC
            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            
            # If binary classification, lb returns (n, 1), we need (n, 2) for roc_auc_score with ovr
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                
            metrics['auc_roc'] = roc_auc_score(y_true_bin, y_probs, multi_class='ovr', average='macro')
            metrics['log_loss'] = log_loss(y_true_bin, y_probs)
        except Exception as e:
            logging.warning(f"Could not compute probability-based metrics: {e}")
            metrics['auc_roc'] = 0.5
            metrics['log_loss'] = 1.0
            
    metrics['mastery_score'] = calculate_mastery_score(metrics)
    return metrics

def plot_visuals(y_true, y_pred, y_probs, class_names, output_dir):
    """
    Generates and saves visual reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. Classification Report (CSV)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    logging.info(f"Visuals saved to {output_dir}")

def verify_paths(paths):
    """
    Checks if all provided paths exist. Returns a list of missing paths or None if all exist.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    return missing if missing else None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Small test on dummy data
    test_y_true = [0, 1, 2, 0, 1, 2]
    test_y_pred = [0, 2, 1, 0, 1, 2]
    test_y_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    classes = ['Class A', 'Class B', 'Class C']
    
    res = compute_all_metrics(test_y_true, test_y_pred, test_y_probs, classes)
    print("Test Metrics:", res)
