import tensorflow as tf
import logging
import datetime
from pathlib import Path

log = logging.getLogger("SynergyTrainer")

def focal_loss(gamma=2., alpha=4.):
    """
    Multi-class focal loss.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Calculate Weight
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

class SynergyTrainer:
    def __init__(self, model, base_model, train_ds, val_ds, output_dir="models/synergy"):
        self.model = model
        self.base_model = base_model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_phased_training(self):
        # Phase 1: Warmup
        log.info("🔥 Phase 1: Warmup (Backbone Frozen)")
        self.base_model.trainable = False
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=10)
        
        # Phase 2: Fine-Tuning
        log.info("🔥 Phase 2: Fine-Tuning (Partial Unfreeze)")
        self.base_model.trainable = True
        # Unfreeze top 30% of layers
        total_layers = len(self.base_model.layers)
        for layer in self.base_model.layers[:int(total_layers * 0.7)]:
            layer.trainable = False
            
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=10)
        
        # Phase 3: Deep Optimization
        log.info("🔥 Phase 3: Deep Optimization (Full Unfreeze + LR Scheduler)")
        for layer in self.base_model.layers:
            layer.trainable = True
            
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # Auto-switch to AdamW if requested (mimicking AdamW with Adam + decay)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=20, callbacks=[lr_scheduler])
        
        # Phase 4: Stability
        log.info("🔥 Phase 4: Stability (Early Stopping)")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )
        
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=10, callbacks=[early_stopping])
        
        # Save Final Model
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = self.output_dir / f"synergy_model_{timestamp}.keras"
        self.model.save(save_path)
        log.info(f"✅ Training Complete. Model saved to {save_path}")
        return save_path
