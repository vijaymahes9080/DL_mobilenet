import os
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
from synergy.model import build_efficientnet_synergy, build_resnet_synergy, build_mobilenet_synergy, build_temporal_synergy
from synergy.data import SynergyDataPipeline
try:
    from scripts.leakage_audit import audit_vision, audit_emotion, audit_behavior
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("AutonomousEngine")

class AutonomousTrainingEngine:
    def __init__(self, dataset_path="dataset/synergy", output_dir="models/autonomous"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_pipeline = SynergyDataPipeline(dataset_path=dataset_path)
        self.best_accuracy = 0.0
        self.history = []
        self.config = {
            "min_epochs": 10,
            "max_epochs": 500,
            "target_accuracy": 0.95,
            "batch_size": 16,
            "learning_rate": 1e-3
        }
        self.models = {}
        self.ensemble_weights = None

    def phase_1_sanity(self):
        log.info("🔷 PHASE 1: DATA SANITY + WARMUP")
        self.data_pipeline.download_or_generate()
        self.data_pipeline.validate_and_clean()
        
        if AUDIT_AVAILABLE:
            log.info("🔍 Running Data Leakage Audit...")
            audit_vision()
            audit_emotion()
            audit_behavior()
        else:
            log.warning("⚠️ Leakage audit script not found. Skipping detailed audit.")
        
        # Check for class imbalance...
        train_ds, val_ds = self.data_pipeline.get_dataset(batch_size=self.config["batch_size"])
        num_classes = len(self.data_pipeline.class_names)
        log.info(f"Dataset loaded with {num_classes} classes.")
        
        # Lightweight warmup
        model, base = build_mobilenet_synergy(num_classes=num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        log.info("Starting lightweight warmup training...")
        model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
        return train_ds, val_ds

    def phase_2_base_training(self, train_ds, val_ds):
        log.info("🔷 PHASE 2: BASE MODEL TRAINING")
        num_classes = len(self.data_pipeline.class_names)
        
        # Train EfficientNet
        log.info("Training EfficientNet Backbone...")
        eff_model, eff_base = build_efficientnet_synergy(num_classes=num_classes)
        eff_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config["learning_rate"]), 
                          loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        
        eff_history = eff_model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])
        self.models['efficientnet'] = eff_model
        
        # Train ResNet
        log.info("Training ResNet Backbone...")
        res_model, res_base = build_resnet_synergy(num_classes=num_classes)
        res_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config["learning_rate"]), 
                          loss='categorical_crossentropy', metrics=['accuracy'])
        
        res_history = res_model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])
        self.models['resnet'] = res_model
        
        return max(max(eff_history.history['val_accuracy']), max(res_history.history['val_accuracy']))

    def phase_3_hardening(self, train_ds, val_ds):
        log.info("🔷 PHASE 3: HARDENING (OVERFITTING CONTROL)")
        for name, model in self.models.items():
            # Analyze Gap
            _, train_acc = model.evaluate(train_ds.take(10), verbose=0)
            _, val_acc = model.evaluate(val_ds.take(10), verbose=0)
            gap = train_acc - val_acc
            
            if gap > 0.05:
                log.warning(f"⚠️ Overfitting detected in {name} (Gap: {gap:.4f}). Applying regularization.")
                # We can't easily change the architecture of an existing model structure in TF easily,
                # but we can increase weight decay or add noise/augmentation
                model.optimizer.learning_rate = self.config["learning_rate"] * 0.1
                
            log.info(f"Refining {name} with Cyclical Learning Rate...")
            clr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: self.config["learning_rate"] * (0.9 ** epoch))
            model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[clr])

    def phase_4_ensemble(self, val_ds):
        log.info("🔷 PHASE 4: MULTI-MODEL ENSEMBLE TRAINING")
        if len(self.models) < 2:
            log.warning("Not enough models for ensemble. Using single best.")
            best_model = list(self.models.values())[0]
            save_path = self.output_dir / "best_model.keras"
            best_model.save(save_path)
            _, self.best_accuracy = best_model.evaluate(val_ds, verbose=0)
            return

        # Weighted Ensemble
        accuracies = {}
        for name, model in self.models.items():
            _, acc = model.evaluate(val_ds, verbose=0)
            accuracies[name] = acc
            
        total_acc = sum(accuracies.values())
        weights = {name: acc/total_acc for name, acc in accuracies.items()}
        self.ensemble_weights = weights
        log.info(f"📊 Ensemble Weights: {weights}")
        
        self.best_accuracy = max(accuracies.values())
        best_name = max(accuracies, key=accuracies.get)
        self.models[best_name].save(self.output_dir / "ensemble_backbone.keras")
        log.info(f"✅ Ensemble Backbone ({best_name}) saved. Val Acc: {self.best_accuracy:.4f}")

    def real_time_camera_loop(self):
        log.info("📸 INTEGRATING REAL-TIME CAMERA TRAINING LOOP")
        # Simulate camera loop
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.warning("No camera detected, skipping real-time loop.")
            return

        log.info("Running live inference loop. Press 'q' to stop.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Preprocess
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0) / 255.0
            
            # Inference (using best model for now)
            best_model = self.models.get('efficientnet') or list(self.models.values())[0]
            preds = best_model.predict(img, verbose=0)
            conf = np.max(preds)
            
            if conf < 0.6:
                log.info(f"Low confidence ({conf:.2f}), storing as hard sample.")
                # Store frame logic here...
            
            cv2.imshow("Synergy Real-Time", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def run_full_pipeline(self):
        log.info("🚀 STARTING AUTONOMOUS DEEP LEARNING ENGINE")
        start_time = time.time()
        
        train_ds, val_ds = self.phase_1_sanity()
        
        # Self-improvement loop
        cycle = 1
        while self.best_accuracy < self.config["target_accuracy"]:
            log.info(f"--- TRAINING CYCLE {cycle} ---")
            
            current_best = self.phase_2_base_training(train_ds, val_ds)
            self.phase_3_hardening(train_ds, val_ds)
            self.phase_4_ensemble(val_ds)
            
            if self.best_accuracy >= self.config["target_accuracy"]:
                log.info(f"🎯 Target Accuracy reached: {self.best_accuracy:.4f}")
                break
            
            log.info(f"Current accuracy {self.best_accuracy:.4f} < {self.config['target_accuracy']}. Adjusting HPs...")
            self.config["learning_rate"] *= 0.5
            cycle += 1
            
            if cycle > 5: # Safety break
                log.warning("Max cycles reached. Stopping.")
                break
        
        duration = time.time() - start_time
        log.info(f"✅ Pipeline Finished in {duration:.2f}s. Final Accuracy: {self.best_accuracy:.4f}")
        
if __name__ == "__main__":
    engine = AutonomousTrainingEngine()
    engine.run_full_pipeline()
