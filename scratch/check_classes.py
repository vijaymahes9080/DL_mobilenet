import tensorflow as tf
import os

DATASET_PATH = r'D:\current project\DL\dataset'
IMG_SIZE = (224, 224)

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=32
)
print(f"Dataset Class Names: {train_ds_raw.class_names}")
