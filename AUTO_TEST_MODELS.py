import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time
import sys
import metric_utils

def load_val_data(folder, img_size=(224, 224), preprocess_mode='none'):
    dataset_path = os.path.join(".", "dataset")
    if os.path.exists(os.path.join(dataset_path, "val")):
        dataset_path = os.path.join(dataset_path, "val")
        ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path, image_size=img_size, batch_size=32, label_mode='categorical', shuffle=False
        )
    else:
        ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path, validation_split=0.2, subset="validation", seed=42,
            image_size=img_size, batch_size=32, label_mode='categorical', shuffle=False
        )
    
    class_names = ds.class_names
    y_true = []
    images = []
    
    for img_batch, label_batch in ds:
        y_true.extend(np.argmax(label_batch.numpy(), axis=1))
        images.append(img_batch.numpy())
    
    images = np.concatenate(images, axis=0)
    y_true = np.array(y_true)
    
    # Preprocessing
    if preprocess_mode == 'mobilenet':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        images = preprocess_input(images)
    elif preprocess_mode == 'resnet':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        images = preprocess_input(images)
    elif preprocess_mode == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        images = preprocess_input(images)
        
    return images, y_true, class_names

def test_model(name, folder, script_path, model_type='tflite', benchmark=False):
    print(f"\n[AUDIT] {name} - Deep Evaluation...")
    
    # 1. Pre-flight Checks
    metric_utils.setup_gpu()
    
    model_path = os.path.join('.', 'models', 'champion_model.pt') if model_type == 'pt' else \
                 os.path.join('.', 'models', 'optimized', 'champion_model.tflite')
    
    missing = metric_utils.verify_paths([model_path, os.path.join('.', 'dataset')])
    if missing:
        print(f"  [SKIP] Required assets missing: {missing}")
        return False
    
    # Set Preprocess Mode
    p_mode = 'none'
    if 'mobilenet' in name.lower(): p_mode = 'mobilenet'
    elif 'resnet' in name.lower(): p_mode = 'resnet'
    elif 'efficient' in name.lower(): p_mode = 'efficientnet'
    
    # Load Data
    images, y_true, class_names = load_val_data(folder, preprocess_mode=p_mode)
    if images is None:
        print(f"  [ERROR] Could not load dataset from {folder}")
        return False

    try:
        y_probs = []
        start_time = time.time()
        
        if model_type == 'pt':
            model = YOLO(model_path)
            # YOLO predict batch (requires uint8 for [0, 255] numpy arrays)
            images_yolo = images.astype(np.uint8) if images.dtype != np.uint8 else images
            results = model.predict(images_yolo, verbose=False)
            for r in results:
                y_probs.append(r.probs.data.cpu().numpy().flatten())
            y_probs = np.array(y_probs)
            
        elif model_type == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            in_det = interpreter.get_input_details()
            out_det = interpreter.get_output_details()
            
            for img in images:
                in_data = np.expand_dims(img, axis=0)
                interpreter.set_tensor(in_det[0]['index'], in_data)
                interpreter.invoke()
                y_probs.append(interpreter.get_tensor(out_det[0]['index'])[0])
            y_probs = np.array(y_probs)
            
        load_time = time.time() - start_time
        fps = len(images) / load_time
        y_pred = np.argmax(y_probs, axis=1)
        
        # Compute Metrics
        metrics = metric_utils.compute_all_metrics(y_true, y_pred, y_probs, class_names)
        
        print(f"  [PASS] Mastery Score: {metrics['mastery_score']:.2f}%")
        print(f"  [PASS] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | AUC: {metrics['auc_roc']:.4f}")
        print(f"  [INFO] Latency: {1000/fps:.2f}ms/img ({fps:.1f} FPS) | MCC: {metrics['mcc']:.4f}")
        
        if benchmark:
            metric_utils.plot_visuals(y_true, y_pred, y_probs, class_names, os.path.join('outputs', name))
            
        return True
    except Exception as e:
        print(f"  [FAIL] Error during inference: {e}")
        return False

test_plan = [
    ("YOLOv8", "DL -YOLO", "inference_hud.py", "pt"),
    ("ResNet50", "DL - imagenet", "inference_hud.py", "tflite"),
    ("EfficientNet", "DL - efficientnet b0", "inference_hud.py", "tflite"),
    ("MobileNetV2", "DL - mobilenet", "inference_hud.py", "tflite")
]

# Mode: Test or Benchmark
mode = "Benchmark" if "--evaluate" in sys.argv else "Integrity"
target_idx = None

for i, arg in enumerate(sys.argv):
    if arg == "--model" and i+1 < len(sys.argv):
        target_idx = int(sys.argv[i+1]) - 1

print(f"==========================================")
print(f"  NEURAL SYNERGY - {mode.upper()} MODE")
print(f"==========================================")

# Identify the local model based on the current directory
current_dir = os.path.basename(os.getcwd())
local_idx = None
for i, (name, folder, script, mtype) in enumerate(test_plan):
    if folder == current_dir:
        local_idx = i
        break

if target_idx is not None:
    if 0 <= target_idx < len(test_plan):
        test_model(*test_plan[target_idx], benchmark=(mode=="Benchmark"))
    else:
        print("[ERROR] Invalid Model Index")
elif local_idx is not None:
    print(f"[INFO] Auto-detected local model: {test_plan[local_idx][0]}")
    test_model(*test_plan[local_idx], benchmark=(mode=="Benchmark"))
else:
    print("[INFO] No local model detected for this directory. Testing all known models...")
    for name, folder, script, mtype in test_plan:
        test_model(name, folder, script, mtype, benchmark=(mode=="Benchmark"))

print("==========================================")


