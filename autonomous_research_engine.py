import os
import json
import logging
import numpy as np

class AutonomousResearchEngine:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.history_file = os.path.join('logs', 'experiment_history.json')
        self.history = self._load_history()
        self.logger = logging.getLogger(__name__)

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def log_experiment(self, change, reason, result):
        entry = {
            "cycle": len(self.history) + 1,
            "change": change,
            "reason": reason,
            "result": result
        }
        self.history.append(entry)
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
        self.logger.info(f"EXPERIMENT LOGGED: {change} -> {reason}")

    def diagnose(self, metrics):
        """
        Detects: Overfitting, Underfitting, Class Bias, Instability.
        """
        diagnosis = []
        
        train_acc = metrics.get('train_accuracy', 0)
        val_acc = metrics.get('val_accuracy', 0)
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        
        # 1. Overfitting (Train high, Val low or Val loss increasing)
        if train_acc > val_acc + 0.08:
            diagnosis.append("OVERFITTING")
        
        # 2. Underfitting (Both low)
        if train_acc < 0.70:
            diagnosis.append("UNDERFITTING")
            
        # 3. Class Bias (Variance in F1 scores across classes)
        # Assuming metrics contains 'f1_per_class'
        f1_per_class = metrics.get('f1_per_class', [])
        if len(f1_per_class) > 0 and np.std(f1_per_class) > 0.15:
            diagnosis.append("CLASS_BIAS")
            
        # 4. Instability (Loss curve fluctuations)
        loss_history = metrics.get('val_loss_history', [])
        if len(loss_history) > 5:
            recent_std = np.std(loss_history[-5:])
            if recent_std > 0.1:
                diagnosis.append("INSTABILITY")
                
        return diagnosis

    def self_correct(self, diagnosis):
        """
        Suggests hyperparameter modifications based on diagnosis.
        RULE: Change ONLY ONE variable per cycle.
        """
        if not diagnosis:
            return self.config, []

        # Prioritize the first issue detected
        issue = diagnosis[0]
        new_config = self.config.copy()
        action = ""

        if issue == "OVERFITTING":
            # Priority: Dropout -> Augmentation -> L2
            if new_config.get('dropout', 0.4) < 0.6:
                new_config['dropout'] += 0.05
                action = "Increased Dropout (+0.05)"
            elif new_config.get('augment_level', 1) < 3:
                new_config['augment_level'] += 1
                action = "Increased Augmentation Level"
            else:
                new_config['l2'] *= 2
                action = "Doubled L2 Regularization"
                
        elif issue == "UNDERFITTING":
            # Priority: Layers -> LR -> Epochs
            if new_config.get('unfreeze_layers', 30) < 100:
                new_config['unfreeze_layers'] += 10
                action = "Unfroze more layers (+10)"
            elif new_config.get('learning_rate', 1e-4) > 1e-6:
                new_config['learning_rate'] *= 0.5
                action = "Reduced Learning Rate (0.5x)"
            else:
                new_config['epochs'] += 5
                action = "Increased Epochs (+5)"
                
        elif issue == "CLASS_BIAS":
            if not new_config.get('use_class_weights', False):
                new_config['use_class_weights'] = True
                action = "Enabled Class Weights"
            else:
                new_config['targeted_sampling'] = True
                action = "Enabled Targeted Sampling"
                
        elif issue == "INSTABILITY":
            if new_config.get('batch_size', 16) > 4:
                new_config['batch_size'] //= 2
                action = "Halved Batch Size"
            else:
                new_config['learning_rate'] *= 0.8
                action = "Reduced Learning Rate (0.8x) for Stability"
                
        return new_config, [action] if action else []

    def is_better(self, current_metrics, best_metrics):
        """
        Compares current metrics with previous best.
        Accept ONLY if statistically better (>0.5% improvement).
        """
        if not best_metrics:
            return True
            
        current_score = current_metrics.get('mastery_score', 0)
        best_score = best_metrics.get('mastery_score', 0)
        
        # Mastery score improvement of 0.5%
        if current_score > best_score + 0.5:
            return True
            
        # Accuracy improvement if scores are equal
        if current_metrics.get('val_accuracy', 0) > best_metrics.get('val_accuracy', 0) + 0.005:
            return True
            
        return False

    def should_stop(self, metrics):
        """
        Hard conditions for stopping training.
        """
        val_acc = metrics.get('val_accuracy', 0)
        mastery_score = metrics.get('mastery_score', 0)
        
        # Stop if performance is high and stable (Mandate: 93% for MobileNet)
        if val_acc >= 0.93 and mastery_score >= 95:
            return True, "Mastery achieved (>93% Accuracy, >95 Mastery Score)"
            
        # Stop if no improvement for 3 cycles
        if len(self.history) >= 3:
            recent_results = [h['result']['val_accuracy'] for h in self.history[-3:]]
            if val_acc >= 0.95 and max(recent_results) - min(recent_results) < 0.005:
                return True, "Performance plateaued across 3 cycles"
                
        return False, "Continuing evolution..."
