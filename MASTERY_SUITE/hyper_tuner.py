import os
import sys
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import metric_utils
import itertools
import time

# --- SHARED CONFIG ---
IMG_SIZE = (224, 224)
TUNING_EPOCHS = 1  # Minimum epochs for fast verification
DEFAULT_RESULTS_FILE = "hyper_tuning_results.csv"

def tune_keras_model(model_name, folder, build_fn, preprocess_fn):
    print(f"\n[TUNER] Tuning {model_name} in {folder}...")
    dataset_path = os.path.join(folder, 'dataset')
    
    if not os.path.exists(dataset_path):
        print(f"  [SKIP] Dataset missing: {dataset_path}")
        return []

    # Load Data
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        dataset_path, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=16, label_mode='categorical'
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        dataset_path, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=16, label_mode='categorical'
    )
    
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)
    
    train_ds = train_ds_raw.take(20).repeat() # Use ~320 images for tuning, repeat for epochs
    val_ds = val_ds_raw.take(10).repeat() # Use ~160 images for tuning
    
    # Preprocess
    train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y))

    # Search Space
    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [8, 16, 32]
    
    results = []

    for lr, bs in itertools.product(lrs, batch_sizes):
        print(f"  [TRIAL] LR={lr}, BS={bs}...")
        
        # Adjust batch size
        train_ds_tuned = train_ds.unbatch().batch(bs).prefetch(tf.data.AUTOTUNE)
        val_ds_tuned = val_ds.unbatch().batch(bs).prefetch(tf.data.AUTOTUNE)
        
        model, _ = build_fn(num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        history = model.fit(train_ds_tuned, epochs=TUNING_EPOCHS, validation_data=val_ds_tuned, verbose=0)
        duration = time.time() - start_time
        
        # Evaluate with metric_utils
        y_true = []
        y_probs = []
        for x, y in val_ds_tuned:
            y_true.extend(np.argmax(y.numpy(), axis=1))
            y_probs.extend(model.predict(x, verbose=0))
        
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = np.argmax(y_probs, axis=1)
        
        metrics = metric_utils.compute_all_metrics(y_true, y_pred, y_probs)
        metrics.update({
            'model': model_name,
            'lr': lr,
            'batch_size': bs,
            'duration': duration,
            'val_acc': history.history['val_accuracy'][-1]
        })
        results.append(metrics)
        
    return results

def tune_yolo_model(folder):
    print(f"\n[TUNER] Tuning YOLOv8 in {folder}...")
    dataset_path = os.path.join(folder, 'dataset')
    
    if not os.path.exists(dataset_path):
        print(f"  [SKIP] Dataset missing: {dataset_path}")
        return []

    lrs = [0.01, 0.001] # YOLO defaults are different
    batch_sizes = [8, 16]
    
    results = []
    
    for lr, bs in itertools.product(lrs, batch_sizes):
        print(f"  [TRIAL] LR={lr}, BS={bs}...")
        model = YOLO('yolov8n-cls.pt')
        
        start_time = time.time()
        res = model.train(
            data=dataset_path,
            epochs=TUNING_EPOCHS,
            lr0=lr,
            batch=bs,
            imgsz=224,
            project='tuning_runs',
            name=f'yolo_lr{lr}_bs{bs}',
            verbose=False,
            exist_ok=True
        )
        duration = time.time() - start_time
        
        # Get metrics from YOLO results
        # YOLO top1 accuracy is usually in res.results_dict
        acc = res.results_dict.get('metrics/accuracy_top1', 0)
        
        results.append({
            'model': 'YOLOv8',
            'lr': lr,
            'batch_size': bs,
            'accuracy': acc,
            'duration': duration
        })
        
    return results

def main():
    # Helper Imports (Mental mapping of build functions)
    # ResNet50
    sys.path.append(os.path.join(os.getcwd(), 'DL - imagenet'))
    import train_local as resnet_train
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    
    # MobileNet
    sys.path.append(os.path.join(os.getcwd(), 'DL - mobilenet'))
    import train_local as mobilenet_train
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre
    
    # EfficientNet
    sys.path.append(os.path.join(os.getcwd(), 'DL - efficientnet b0'))
    import train_local as effnet_train
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre

    all_results = []
    
    # 1. ResNet50
    # all_results.extend(tune_keras_model("ResNet50", "DL - imagenet", resnet_train.build_mastery_model, resnet_pre))
    
    # 2. MobileNetV2
    # all_results.extend(tune_keras_model("MobileNetV2", "DL - mobilenet", mobilenet_train.build_mastery_model, mobilenet_pre))
    
    # 3. EfficientNetB0
    # all_results.extend(tune_keras_model("EfficientNetB0", "DL - efficientnet b0", effnet_train.build_mastery_model, effnet_pre))
    
    # 4. YOLOv8
    # all_results.extend(tune_yolo_model("DL -YOLO"))

    # Since I cannot run them all at once (time limit), I'll make this script callable with arguments
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if 'resnet' in target:
            all_results.extend(tune_keras_model("ResNet50", "DL - imagenet", resnet_train.build_mastery_model, resnet_pre))
        elif 'mobile' in target:
            all_results.extend(tune_keras_model("MobileNetV2", "DL - mobilenet", mobilenet_train.build_mastery_model, mobilenet_pre))
        elif 'efficient' in target:
            all_results.extend(tune_keras_model("EfficientNetB0", "DL - efficientnet b0", effnet_train.build_mastery_model, effnet_pre))
        elif 'yolo' in target:
            all_results.extend(tune_yolo_model("DL -YOLO"))
    
    if all_results:
        df = pd.DataFrame(all_results)
        if os.path.exists(DEFAULT_RESULTS_FILE):
            df.to_csv(DEFAULT_RESULTS_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(DEFAULT_RESULTS_FILE, index=False)
        print(f"\n[SUCCESS] Tuning results appended to {DEFAULT_RESULTS_FILE}")

if __name__ == "__main__":
    main()
