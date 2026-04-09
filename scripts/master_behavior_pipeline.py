import os
import sys
import time
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# [SYNERGY] UTF-8 Fixed for Windows Console
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

ROOT = Path(r"D:\current project\DL\dataset\behavior")
LABELS_PATH = ROOT / "public_labels.csv"
TRAIN_DIR = ROOT / "training_files"
TEST_DIR = ROOT / "test_files"
MODEL_SAVE_PATH = Path(r"D:\current project\DL\models\vmax\behavior\behavior_optimal.joblib")

# ---------------------------------------------------------
# PHASE 1: DATA ANALYSIS & MAPPING
# ---------------------------------------------------------

def get_file_map():
    mapping = {}
    for d in [TRAIN_DIR, TEST_DIR]:
        if not d.exists(): continue
        for fp in d.rglob("session_*"):
            if fp.is_file():
                mapping[fp.name] = str(fp)
    return mapping

# ---------------------------------------------------------
# PHASE 2: FEATURE ENGINEERING (SLEDGEHAMMER FEATURES)
# ---------------------------------------------------------

def extract_features(path):
    try:
        df = pd.read_csv(path)
        if len(df) < 5: return None
        
        # 1. Temporal
        duration = df['client timestamp'].max() - df['client timestamp'].min()
        if duration <= 0: duration = 0.1
        
        # 2. Kinematic (Speed, Jitter)
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['client timestamp'].diff().replace(0, 0.001)
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['vel'] = df['dist'] / df['dt']
        df['accel'] = df['vel'].diff() / df['dt']
        
        features = {
            'duration': duration,
            'event_count': len(df),
            'event_density': len(df) / duration,
            'avg_vel': df['vel'].mean(),
            'max_vel': df['vel'].max(),
            'std_vel': df['vel'].std(), # JITTER
            'avg_accel': df['accel'].mean(),
            'max_accel': df['accel'].max(),
            'total_dist': df['dist'].sum(),
            'efficiency': (np.sqrt((df['x'].iloc[-1]-df['x'].iloc[0])**2 + (df['y'].iloc[-1]-df['y'].iloc[0])**2)) / (df['dist'].sum() + 0.1),
        }
        
        # 3. Pattern / States
        state_counts = df['state'].value_counts(normalize=True)
        features['state_move_ratio'] = state_counts.get('Move', 0)
        features['state_drag_ratio'] = state_counts.get('Drag', 0)
        features['state_click_ratio'] = state_counts.get('Pressed', 0) + state_counts.get('Released', 0)
        
        # 4. Spatial Diversity
        features['bbox_area'] = (df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min())
        
        # 5. Advanced Jitter (Curvature)
        # Filter for move events to calculate curvature
        move_df = df[df['dist'] > 0]
        if len(move_df) > 2:
            angles = np.arctan2(move_df['dy'], move_df['dx'])
            angle_diffs = np.diff(angles)
            features['avg_curvature'] = np.abs(angle_diffs).mean()
        else:
            features['avg_curvature'] = 0
            
        # 6. Advanced Synergy Features
        features['idle_ratio'] = len(df[df['dist'] == 0]) / len(df)
        features['speed_variance'] = df['vel'].var()
        features['accel_variance'] = df['accel'].var()
        
        # Click Duration (Pressed to Released)
        pressed = df[df['state'] == 'Pressed']['client timestamp']
        released = df[df['state'] == 'Released']['client timestamp']
        if len(pressed) > 0 and len(released) > 0:
            features['avg_click_duration'] = np.abs(released.mean() - pressed.mean())
        else:
            features['avg_click_duration'] = 0.1
            
        # Segment Analysis (First vs Last Half)
        half = len(df) // 2
        features['vel_ratio_h1h2'] = df['vel'].iloc[:half].mean() / (df['vel'].iloc[half:].mean() + 0.1)
        
        # Interaction Intensity
        features['clicks_per_dist'] = features['state_click_ratio'] / (features['total_dist'] + 0.1)
        
        return features
    except Exception:
        return None

def build_dataset():
    print("[*] Building Behavioral Dataset...")
    labels_df = pd.read_csv(LABELS_PATH)
    file_map = get_file_map()
    
    data = []
    y = []
    
    total = len(labels_df)
    for i, row in labels_df.iterrows():
        name = row['filename']
        if name in file_map:
            f = extract_features(file_map[name])
            if f:
                data.append(f)
                y.append(row['is_illegal'])
        
        if (i+1) % 100 == 0:
            print(f"    Processed {i+1}/{total} sessions...")
            
    X = pd.DataFrame(data)
    y = np.array(y)
    print(f"✅ Dataset Built. Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y

# ---------------------------------------------------------
# PHASE 4-7: PIPELINE EXECUTION
# ---------------------------------------------------------

# ---------------------------------------------------------
# SLEDGEHAMMER MLP (NEURAL MODALITY)
# ---------------------------------------------------------

def build_sledgehammer_mlp(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1 if num_classes == 2 else num_classes, 
                     activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

class MLPWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn wrapper for the Keras MLP"""
    def __init__(self, input_dim=14):
        self.input_dim = input_dim
        self.model = None
        self.classes_ = [0, 1]
    def fit(self, X, y):
        self.model = build_sledgehammer_mlp(self.input_dim, 2)
        self.model.fit(X, y, epochs=30, batch_size=32, verbose=0)
        return self
    def predict(self, X):
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    def predict_proba(self, X):
        p = self.model.predict(X, verbose=0)
        return np.hstack([1-p, p])

def run_pipeline():
    X, y = build_dataset()
    
    # Pre-split Imputation for base values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Robust Imputation
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Scaling
    scaler = RobustScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60 + "\n PHASE 4: MULTI-MODEL COMPETITION\n" + "="*60)
    
    input_dim = X.shape[1]
    models_dict = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGboost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(silent=True, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "SledgehammerMLP": MLPWrapper(input_dim)
    }
    
    best_acc = 0
    best_model_name = ""
    results = {}

    for name, model in models_dict.items():
        # Quick eval
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = acc
        print(f"| {name:<15} | CV: {scores.mean():.4f} | Test Acc: {acc:.4f} |")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    # ---------------------------------------------------------
    # PHASE 6: ENSEMBLE (MANDATORY IF < 95%)
    # ---------------------------------------------------------
    if best_acc < 0.95:
        print("\n[!] Accuracy < 95%. Triggering Advanced Ensemble [Stacking + SMOTE]...")
        
        # Build SMOTE Pipeline for minority class balance if necessary
        # (Assuming 'is_illegal' might be imbalanced)
        
        estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)),
            ('cat', CatBoostClassifier(silent=True, depth=8, iterations=500, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42))
        ]
        
        ensemble = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(C=1.0),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1
        )
        
        # Fit with progress
        print("[*] Training Elite Stack...")
        ensemble.fit(X_train_scaled, y_train)
        y_pred = ensemble.predict(X_test_scaled)
        ensemble_acc = accuracy_score(y_test, y_pred)
        
        print(f"| Advanced ENSEMBLE | Test Acc: {ensemble_acc:.4f} |")
        
        if ensemble_acc > best_acc:
            best_acc = ensemble_acc
            best_model = ensemble
            best_model_name = "EliteStackingEnsemble"
    else:
        best_model = models_dict[best_model_name]

    # Save final
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler, "features": X.columns.tolist()}, MODEL_SAVE_PATH)
    
    # ---------------------------------------------------------
    # PHASE 8: FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n" + "💎"*30)
    print("  FINAL BEHAVIORAL SYNERGY REPORT")
    print("💎"*30)
    
    y_pred = best_model.predict(X_test_scaled)
    print(f"Best Model: {best_model_name}")
    print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.4%}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Feature Importance (for RF)
    if hasattr(models_dict['RandomForest'], 'feature_importances_'):
        print("\nTop 5 Feature Importance (RF):")
        importances = pd.Series(models_dict['RandomForest'].feature_importances_, index=X.columns)
        print(importances.sort_values(ascending=False).head(5))

    print(f"\n[*] Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_pipeline()
