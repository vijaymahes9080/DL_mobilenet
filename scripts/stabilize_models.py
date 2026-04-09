
import tensorflow as tf
import joblib
import numpy as np
import os
from pathlib import Path

def create_dummy_keras(modality, num_classes=7, input_shape=(120, 120, 3)):
    print(f"Creating dummy model for {modality}...")
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    
    path = Path(f"d:/current project/DL/models/vmax/{modality}")
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / f"{modality}_optimal.keras")
    print(f"Saved to {path / f'{modality}_optimal.keras'}")

def create_dummy_joblib(modality, input_dim=14, num_classes=3):
    print(f"Creating dummy joblib for {modality}...")
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1)
    # Fit on dummy data to initialize
    X = np.random.rand(10, input_dim)
    y = np.random.randint(0, num_classes, 10)
    model.fit(X, y)
    
    path = Path(f"d:/current project/DL/models/vmax/{modality}")
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / f"{modality}_optimal.joblib")
    print(f"Saved to {path / f'{modality}_optimal.joblib'}")

# Create mandatory models for architecture stabilization
create_dummy_keras("emotion_master")
create_dummy_keras("face_alt")
create_dummy_keras("face")
create_dummy_keras("gesture", num_classes=2)
create_dummy_joblib("behavior")
create_dummy_joblib("eye", input_dim=4, num_classes=1)

print("All dummy models generated for system stabilization.")
