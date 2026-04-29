import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Use the same config if possible, or fallback
try:
    import config
except ImportError:
    class Config:
        BASE_PATH = os.getcwd()
        DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
        MODEL_PATH = os.path.join(BASE_PATH, 'models')
        LOG_PATH = os.path.join(BASE_PATH, 'logs')
    config = Config()

def run_ablation_scenario(name, disable_aug=False):
    logger.info(f"--- Running Ablation Scenario: {name} ---")
    
    IMG_SIZE = (224, 224); BATCH_SIZE = 16
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Preprocessing
    if not disable_aug:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
        ])
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))
    
    # Model (MobileNetV2 with Preprocessing)
    model = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)),
        applications.MobileNetV2(include_top=False, weights='imagenet'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.layers[1].trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fast training for ablation proof
    history = model.fit(train_ds, epochs=3, validation_data=val_ds, verbose=1)
    
    val_acc = history.history['val_accuracy'][-1]
    logger.info(f"Scenario {name} completed with Val Accuracy: {val_acc:.4f}")
    return val_acc

def main():
    results = []
    
    # Reference (Full Pipeline - estimated from previous champion run or quick run)
    # To keep it fast, we just run 2 scenarios: Base vs No-Aug
    results.append({'scenario': 'Standard Pipeline', 'accuracy': run_ablation_scenario('Standard', disable_aug=False)})
    results.append({'scenario': 'No Augmentation', 'accuracy': run_ablation_scenario('No Augmentation', disable_aug=True)})
    
    df = pd.DataFrame(results)
    df['performance_drop'] = df['accuracy'].iloc[0] - df['accuracy']
    
    output_path = os.path.join(config.LOG_PATH, 'ablation_results.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Ablation results saved to {output_path}")

if __name__ == "__main__":
    main()
