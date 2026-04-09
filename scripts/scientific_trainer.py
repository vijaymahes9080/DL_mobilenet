import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU Only for stability unless override

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, applications, optimizers
from sklearn.metrics import classification_report, confusion_matrix

# Scientific Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ScientificTrainer")

@dataclass
class Hyperparams:
    learning_rate: float = 1e-3
    batch_size: int = 16
    dropout_rate: float = 0.3
    aug_factor: float = 0.2
    optimizer_name: str = "adam"
    warmup_epochs: int = 5
    partial_epochs: int = 10
    full_epochs: int = 15
    top_n_unfreeze: int = 30
    
class ScientificTrainer:
    def __init__(self, dataset_path, output_dir="models/scientific"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "scientific_training_history.json"
        self.history = self._load_history()
        self.current_cycle = len(self.history) + 1
        self.max_cycles = 20
        self.img_size = 224
        
    def _load_history(self):
        if self.log_path.exists():
            return json.loads(self.log_path.read_text())
        return []

    def _save_history(self, record):
        self.history.append(record)
        self.log_path.write_text(json.dumps(self.history, indent=4))

    def prepare_data(self, batch_size):
        log.info(f"🧬 Preparing Dataset: {self.dataset_path}")
        
        if str(self.dataset_path).lower() == "cifar10":
            log.info("📦 Loading CIFAR-10 via Keras Datasets")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.num_classes = 10
            
            # Convert to float and resize (Note: this is RAM intensive if done upfront)
            # Use a generator or tf.data for scale
            def gen(x, y):
                for img, label in zip(x, y):
                    img = tf.cast(img, tf.float32)
                    img = tf.image.resize(img, (self.img_size, self.img_size))
                    yield img, label
            
            # Split train into train/val
            split = int(len(x_train) * 0.8)
            x_val, y_val = x_train[split:], y_train[split:]
            x_train, y_train = x_train[:split], y_train[:split]
            
            self.train_ds = tf.data.Dataset.from_generator(lambda: gen(x_train, y_train), output_signature=(tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32), tf.TensorSpec(shape=(1,), dtype=tf.int64))).batch(batch_size)
            self.val_ds = tf.data.Dataset.from_generator(lambda: gen(x_val, y_val), output_signature=(tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32), tf.TensorSpec(shape=(1,), dtype=tf.int64))).batch(batch_size)
            self.test_ds = tf.data.Dataset.from_generator(lambda: gen(x_test, y_test), output_signature=(tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32), tf.TensorSpec(shape=(1,), dtype=tf.int64))).batch(batch_size)
            
        else:
            # Validate dataset
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset path {self.dataset_path} not found.")
                
            # Data Generator with Augmentation
        # Initial Augmentation Strength set by hyperparams
        self.train_ds, self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2, # We split 80/20 initially, then 70/15/15 is handled via further split
            subset="both",
            seed=42,
            image_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            label_mode='int'
        )
        
        # Split val into val/test
        val_batches = tf.data.experimental.cardinality(self.val_ds)
        self.test_ds = self.val_ds.take(val_batches // 2)
        self.val_ds = self.val_ds.skip(val_batches // 2)
        
        # Class names
        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)
        
        # Caching/Prefetching
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def build_model(self, h: Hyperparams):
        log.info(f"🏗️ Building EfficientNetV2-B0 Backbone")
        base = applications.EfficientNetV2B0(
            include_top=False, 
            weights='imagenet', 
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Data Augmentation Layer
        aug = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(h.aug_factor),
            layers.RandomZoom(h.aug_factor),
        ], name="augmentation")
        
        model = models.Sequential([
            layers.Input(shape=(self.img_size, self.img_size, 3)),
            aug,
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(h.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ], name="ScientificNet")
        
        return model, base

    async def run_cycle(self, h: Hyperparams):
        log.info(f"🔄 Starting Scientific Cycle {self.current_cycle} / {self.max_cycles}")
        log.info(f"⚙️ Params: {h}")
        
        self.prepare_data(h.batch_size)
        model, base = self.build_model(h)
        
        # --- PHASE 1: Warm-up ---
        log.info("🔹 Phase 1: Warming up Classifier Head...")
        base.trainable = False
        model.compile(
            optimizer=optimizers.Adam(learning_rate=h.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        hist1 = model.fit(self.train_ds, validation_data=self.val_ds, epochs=h.warmup_epochs, verbose=1)
        
        # --- PHASE 2: Partial Fine-Tuning ---
        log.info(f"🔹 Phase 2: Partial Fine-Tuning (Top {h.top_n_unfreeze} layers)...")
        base.trainable = True
        # Freeze all but top N
        for layer in base.layers[:-h.top_n_unfreeze]:
            layer.trainable = False
            
        model.compile(
            optimizer=optimizers.Adam(learning_rate=h.learning_rate / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        hist2 = model.fit(self.train_ds, validation_data=self.val_ds, epochs=h.partial_epochs, verbose=1)
        
        # --- PHASE 3: Full Fine-Tuning ---
        log.info("🔹 Phase 3: Full Model Optimization...")
        base.trainable = True
        model.compile(
            optimizer=optimizers.Adam(learning_rate=h.learning_rate / 100),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # Early Stopping for Phase 3
        cb = [callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')]
        hist3 = model.fit(self.train_ds, validation_data=self.val_ds, epochs=h.full_epochs, verbose=1, callbacks=cb)
        
        # Final Evaluation
        test_loss, test_acc = model.evaluate(self.test_ds, verbose=0)
        log.info(f"🏁 Cycle {self.current_cycle} Test Accuracy: {test_acc:.4f}")
        
        # Advanced Metrics
        y_true = []
        y_pred = []
        for img, label in self.test_ds:
            y_true.extend(label.numpy())
            y_pred.extend(np.argmax(model.predict(img, verbose=0), axis=1))
        
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Check Overfitting
        train_acc = hist3.history['accuracy'][-1]
        val_acc = hist3.history['val_accuracy'][-1]
        gap = train_acc - val_acc
        
        record = {
            "cycle": self.current_cycle,
            "hyperparams": asdict(h),
            "metrics": {
                "test_accuracy": test_acc,
                "val_accuracy": val_acc,
                "train_accuracy": train_acc,
                "gap": gap,
                "report": report
            },
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        self._save_history(record)
        
        # Save model if top accuracy
        best_prev = max([r["metrics"]["val_accuracy"] for r in self.history[:-1]]) if len(self.history) > 1 else 0
        if val_acc > best_prev:
            model.save(self.output_dir / "scientific_best_model.keras")
            log.info("💎 New Elite High-Score Model Saved!")
            
        return record

    def adjust_hyperparams(self, last_record):
        m = last_record["metrics"]
        h_dict = last_record["hyperparams"]
        h = Hyperparams(**h_dict)
        
        # Trigger Condition: Gap too high
        if m["gap"] > 0.05:
            log.info("⚠️ High Overfitting Detected (> 5%). Improving Regularization.")
            # Adjust ONE: Dropout or Augmentation
            if h.dropout_rate < 0.5:
                h.dropout_rate += 0.05
            else:
                h.aug_factor += 0.05
            return h
            
        # Trigger Condition: Accuracy too low
        if m["val_accuracy"] < 0.95:
            log.info("📈 Accuracy below 95%. Optimizing Learning Rate Cycle.")
            # Reduce LR slightly if we reached a plateau
            h.learning_rate *= 0.8
            return h
            
        return h

# --- MAIN CONTROLLER ---
async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--partial_epochs", type=int, default=10)
    parser.add_argument("--full_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="cifar10")
    args = parser.parse_args()

    trainer = ScientificTrainer(args.dataset)
    h = Hyperparams(
        warmup_epochs=args.warmup_epochs,
        partial_epochs=args.partial_epochs,
        full_epochs=args.full_epochs,
        batch_size=args.batch_size
    )
    
    # Check if we should resume
    if trainer.history:
        log.info("Resuming from last cycle...")
        last = trainer.history[-1]
        if last["metrics"]["val_accuracy"] >= 0.95 and last["metrics"]["gap"] <= 0.05:
            log.info("✅ Target already achieved in previous run.")
            return
        h = trainer.adjust_hyperparams(last)
        
    for cycle in range(trainer.current_cycle, 21):
        record = await trainer.run_cycle(h)
        
        # Stop Conditions
        if record["metrics"]["val_accuracy"] >= 0.95 and record["metrics"]["gap"] <= 0.05:
            log.info("🎯 TARGET ACHIEVED. Terminating Optimization Protocol.")
            break
            
        # check for 5-cycle no improvement
        if len(trainer.history) >= 5:
            recent_accs = [r["metrics"]["val_accuracy"] for r in trainer.history[-5:]]
            if max(recent_accs) - min(recent_accs) < 0.001:
                log.info("🛑 Plateau Detected. No improvement for 5 consecutive cycles. Stopping.")
                break
                
        h = trainer.adjust_hyperparams(record)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
