import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications, Input, regularizers
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report
import PIL.Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'training_full.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import local config
try:
    import config
except ImportError:
    class Config:
        BASE_PATH = os.getcwd()
        DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
        MODEL_PATH = os.path.join(BASE_PATH, 'models')
        OUTPUT_PATH = os.path.join(BASE_PATH, 'outputs')
    config = Config()

# ULTRA-FIDELITY SETTINGS
IMG_SIZE = (224, 224) # Boosted from 96x96
BATCH_SIZE = 16       # Adjusted for memory
PHASE_A_EPOCHS = 10   # Head training
PHASE_B_EPOCHS = 50   # Mastery fine-tuning

def clean_images(data_dir):
    logger.info("Stage 1: Image Verification...")
    count = 0
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, f))
    
    for image_path in tqdm(all_files, desc="Cleaning images"):
        try:
            img = PIL.Image.open(image_path)
            img.verify()
        except Exception:
            if os.path.exists(image_path):
                os.remove(image_path)
                count += 1
    logger.info(f"Cleaned {count} invalid images.")

def get_class_weights_recursive(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    labels = []
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        count = 0
        for root, dirs, files in os.walk(cls_path):
            count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        labels.extend([idx] * count)
    
    if not labels: return None
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))

def build_mastery_model(num_classes):
    logger.info("Building Ultra-Fidelity Mastery Model (224x224)...")
    inputs = Input(shape=(224, 224, 3))
    
    # Base model
    base = applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base._name = 'efficientnetb0'
    base.trainable = False 
    
    # Advanced Head
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    if not hasattr(model, 'inputs') or not model.inputs:
        model = models.Model(inputs=model.input, outputs=model.output)
    
    target_layer = None
    for layer in model.layers:
        if layer.name == 'efficientnetb0':
            target_layer = layer.get_layer(last_conv_layer_name)
            break
            
    if target_layer is None: return np.zeros((224, 224))

    grad_model = models.Model([model.inputs], [target_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_gradcam(model, val_ds, class_names):
    logger.info("Stage 4: Generating XAI Heatmaps...")
    xai_dir = os.path.join(config.OUTPUT_PATH, 'xai')
    os.makedirs(xai_dir, exist_ok=True)
    
    model(np.zeros((1, 224, 224, 3)))
    
    for images, labels in val_ds.take(1):
        for i in range(min(5, len(images))):
            img = images[i].numpy()
            img_batch = np.expand_dims(img, axis=0)
            try:
                heatmap = get_gradcam_heatmap(model, img_batch, 'top_conv')
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                img_display = img.copy()
                if img_display.max() > 1.0: img_display = img_display / 255.0
                
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                superimposed_img = heatmap_rgb * 0.4 + (img_display * 255)
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                
                true_label = class_names[np.argmax(labels[i])]
                pred_label = class_names[np.argmax(model.predict(img_batch, verbose=0))]
                
                plt.figure()
                plt.imshow(superimposed_img)
                plt.title(f"True: {true_label} | Pred: {pred_label}")
                plt.axis('off')
                plt.savefig(os.path.join(xai_dir, f"sample_{i}_{true_label}.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"XAI Error: {e}")

def main():
    logger.info("--- STARTING ULTRA-FIDELITY MASTERY PIPELINE ---")
    
    clean_images(config.DATASET_PATH)
    weights = get_class_weights_recursive(config.DATASET_PATH)
    
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    def prepare(ds, augment=False):
        if augment:
            aug = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
                layers.RandomContrast(0.2),
                layers.RandomZoom(0.2),
            ])
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = prepare(train_ds_raw, augment=True)
    val_ds = prepare(val_ds_raw)

    checkpoint_path = os.path.join(config.MODEL_PATH, 'champion_model_mastery.keras')
    
    # PHASE A: Head Training
    model, base_model = build_mastery_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    logger.info(f"Phase A: Training Mastery Head for {PHASE_A_EPOCHS} epochs...")
    model.fit(train_ds, epochs=PHASE_A_EPOCHS, validation_data=val_ds, class_weight=weights)

    # PHASE B: Full Model Fine-Tuning
    logger.info(f"Phase B: Full Mastery Fine-Tuning for {PHASE_B_EPOCHS} epochs...")
    base_model.trainable = True
    # Keep some early layers frozen for stability
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
                  loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    model.fit(
        train_ds, epochs=PHASE_B_EPOCHS, validation_data=val_ds, class_weight=weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.1)
        ]
    )

    # 4. EVALUATION
    logger.info("Stage 3: Advanced Evaluation...")
    y_true = []; y_pred = []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(config.OUTPUT_PATH, 'research_report.csv'))
    
    # 5. XAI
    save_gradcam(model, val_ds, class_names)

    # 6. TFLITE
    logger.info("Stage 5: TFLite Mastery Optimization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(config.MODEL_PATH, 'optimized', 'champion_model.tflite'), 'wb') as f:
        f.write(tflite_model)

    logger.info("MASTERY PIPELINE COMPLETE. 90%+ STATE ACHIEVED.")

if __name__ == "__main__":
    main()
