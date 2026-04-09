import os
import shutil
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

log = logging.getLogger("SynergyData")

class SynergyDataPipeline:
    def __init__(self, splits_path="training/splits", img_size=(224, 224)):
        self.splits_path = Path(splits_path)
        self.img_size = img_size
        self.class_names = []
        
    def download_or_generate(self, kaggle_slug=None):
        """
        In this context, we assume data is already preprocessed and split.
        If not, we would normally download, but here we verify the directory.
        """
        if not self.splits_path.exists():
            log.warning(f"⚠️ Splits path {self.splits_path} not found. Using root vision directories if available.")
            # Fallback for synthetic generation if needed
            if not Path("dataset/synergy").exists():
                self._generate_synthetic_data()
            self.splits_path = Path("dataset/synergy")

    def _generate_synthetic_data(self):
        classes = ["Class_A", "Class_B", "Class_C"]
        path = Path("dataset/synergy")
        path.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            cls_dir = path / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(100):
                img_data = np.random.randint(0, 255, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                img = Image.fromarray(img_data)
                img.save(cls_dir / f"synthetic_{i}.jpg")
        log.info(f"✅ Synthetic data generated in {path}")

    def validate_and_clean(self):
        log.info("🧹 Validating dataset integrity...")
        count = 0
        for img_path in self.splits_path.glob("**/*"):
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    img_path.unlink()
                    count += 1
        log.info(f"✅ Cleanup complete. Removed {count} corrupted files.")

    def get_dataset(self, modality="vision", batch_size=32):
        """
        Loads the specified modality split. Defaulting to vision if not specified.
        """
        train_path = self.splits_path / "train" / modality
        val_path = self.splits_path / "val" / modality
        
        if not train_path.exists():
            # Fallback to general dataset if splits don't match
            log.info(f"Path {train_path} not found, using generic loader.")
            return self._get_generic_dataset(batch_size)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        self.class_names = train_ds.class_names
        
        # Prefetching
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return train_ds, val_ds

    def _get_generic_dataset(self, batch_size):
        # Generic loader for unsplit data
        dataset_path = Path("dataset/synergy")
        if not dataset_path.exists():
             self._generate_synthetic_data()
             
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="both",
            seed=42,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        self.class_names = train_ds.class_names
        return train_ds, val_ds

    def get_augmentation_layer(self):
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ], name="adaptive_augmentation")
