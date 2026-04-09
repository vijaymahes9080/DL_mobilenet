
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "models/autonomous/ensemble_backbone.keras"
VOICE_DATA = "training/splits/test/voice/data.npy"
VOICE_LABELS = "training/splits/test/voice/labels.npy"

model = tf.keras.models.load_model(MODEL_PATH)
X = np.load(VOICE_DATA)[:20]
y = np.load(VOICE_LABELS)[:20]

# Preprocess voice (MFCCs are already 128x128 in npy? Let's check)
print(f"Original shape: {X.shape}")
X_p = []
for img in X:
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_resized = tf.image.resize(img, (96, 96)).numpy() # Ensemble expects 96x96
    img_rgb = np.concatenate([img_resized]*3, axis=-1)
    X_p.append(img_rgb)

X_p = np.array(X_p) / 255.0
y_probs = model.predict(X_p)
y_pred = np.argmax(y_probs, axis=1)

print(f"True Labels: {y}")
print(f"Pred Labels: {y_pred}")
print(f"Max Probs: {np.max(y_probs, axis=1)}")
