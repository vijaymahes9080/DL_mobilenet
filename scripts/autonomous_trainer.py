import os
import sys
import json
import time
import datetime
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import asyncio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, applications, optimizers
import matplotlib.pyplot as plt

# Force CPU if no GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("--- Optimized NEURAL TRAINER BOOTING ---", flush=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("AutonomousSystem")
print("--- Log initialized ---", flush=True)

@dataclass
class Hyperparams:
    learning_rate: float = 1e-3
    batch_size: int = 16
    dropout_rate: float = 0.3
    aug_factor: float = 0.2
    optimizer_name: str = "adam"
    warmup_epochs: int = 2
    partial_epochs: int = 4
    full_epochs: int = 8
    top_n_unfreeze: int = 20
    last_modified: str = "init"

class AutonomousSystem:
    def __init__(self, dataset_path, output_dir="models/autonomous"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "training_history.json"
        self.history = self._load()
        self.img_size = 128
        
    def _load(self):
        return json.loads(self.log_path.read_text()) if self.log_path.exists() else []

    def _save(self, rec):
        self.history.append(rec)
        self.log_path.write_text(json.dumps(self.history, indent=4))

    def prepare_data(self, bs):
        log.info(f"🧬 Preparing Data from: {self.dataset_path}")
        ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path, 
            label_mode='int', 
            image_size=(self.img_size, self.img_size), 
            batch_size=bs, 
            shuffle=True, 
            seed=42
        )
        self.class_names = ds.class_names
        self.num_classes = len(self.class_names)
        
        # Split: 70/15/15
        total = tf.data.experimental.cardinality(ds).numpy()
        tr = int(0.7 * total)
        vl = int(0.15 * total)
        
        self.train_ds = ds.take(tr).cache().prefetch(tf.data.AUTOTUNE)
        rem = ds.skip(tr)
        self.val_ds = rem.take(vl).cache().prefetch(tf.data.AUTOTUNE)
        self.test_ds = rem.skip(vl).cache().prefetch(tf.data.AUTOTUNE)
        log.info(f"Batches: Train={tr}, Val={vl}, Test={total-tr-vl}")

    def build(self, h: Hyperparams):
        log.info(f"🏗️ Building Ultra-Light Neural Core")
        # Using a very light base for this environment
        base = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.GlobalAveragePooling2D(),
        ])
        
        aug = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ])
        
        model = models.Sequential([
            layers.Input(shape=(128,128,3)),
            aug,
            base,
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model, base

    async def run_cycle(self, h: Hyperparams, cycle):
        log.info(f"🔄 CYCLE {cycle} BEGINS")
        self.prepare_data(h.batch_size)
        model, base = self.build(h)
        
        # Focused Single Phase for Speed
        model.compile(optimizer=optimizers.Adam(h.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(self.train_ds, validation_data=self.val_ds, epochs=10, verbose=1)
        
        # Eval
        _, test_acc = model.evaluate(self.test_ds, verbose=0)
        val_acc = hist.history['val_accuracy'][-1]
        gap = hist.history['accuracy'][-1] - val_acc
        
        log.info(f"🏁 CYCLE {cycle} RESULTS: Val_Acc={val_acc:.4f}, Gap={gap:.4f}")
        
        rec = {
            "cycle": cycle,
            "params": asdict(h),
            "metrics": {"val_acc": val_acc, "test_acc": test_acc, "gap": gap},
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self._save(rec)
        
        # Optimize & Stop
        if val_acc >= 0.95 and gap <= 0.05:
            log.info("🎯 TARGET ACHIEVED. Finalizing Production Core...")
            self.convert(model)
            self.audit(model)
            
            # Finalize Deployment to vmax
            vmax_dir = Path("models/vmax/emotion_master")
            vmax_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.output_dir / "optimized_model_fp16.tflite", vmax_dir / "emotion_master_optimal.tflite")
            log.info(f"✅ Deployment Finalized: {vmax_dir / 'emotion_master_optimal.tflite'}")
            return True
        return False

    def convert(self, model, quant_type="FP16"):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quant_type == "INT8":
            def representative_dataset():
                for data, _ in self.train_ds.take(100):
                    yield [data]
            converter.representative_dataset = representative_dataset
            # Ensure full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tag = "int8"
        else:
            # FP16 (Default)
            converter.target_spec.supported_types = [tf.float16]
            tag = "fp16"
            
        tflite_model = converter.convert()
        model_name = f"optimized_model_{tag}.tflite"
        (self.output_dir / model_name).write_bytes(tflite_model)
        log.info(f"⚡ TFLite [{tag.upper()}] Transformation Complete.")
        return model_name

    def evolve(self, rec):
        h = Hyperparams(**rec["params"])
        m = rec["metrics"]
        if m["gap"] > 0.05:
            h.dropout_rate += 0.1; h.last_modified = "dropout"
        elif m["val_acc"] < 0.95:
            h.learning_rate *= 0.5; h.last_modified = "lr"
        return h

    def audit(self, model):
        log.info("🧪 Launching F1-Score Audit...")
        y_true, y_pred = [], []
        # Need to re-collect test data if batching was weird
        for x, y in self.test_ds:
            p = model.predict(x, verbose=0)
            y_true.extend(y.numpy())
            y_pred.extend(np.argmax(p, axis=1))
        
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        log.info(f"\n📊 CLASSIFICATION REPORT:\n{report}")
        (self.output_dir / "classification_report.txt").write_text(report)

async def main():
    # Correct path to training subfolders to ensure classes are picked up correctly
    trainer = AutonomousSystem("dataset/face_emotion/train")
    h = Hyperparams()
    for c in range(1, 4): # Focused 1-3 Cycle range as per request
        done = await trainer.run_cycle(h, c)
        if done: 
            # Auditing the successful model
            # We don't have the model object here easily unless we modify run_cycle to return it.
            # Let's modify run_cycle to return the model as well.
            break
        h = trainer.evolve(trainer.history[-1])

if __name__ == "__main__":
    asyncio.run(main())

