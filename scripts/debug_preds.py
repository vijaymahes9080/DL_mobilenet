
import numpy as np
import tensorflow as tf
MODEL_PATH = "models/vmax/emotion_master/emotion_master_optimal.keras"
DATA_PATH = "training/splits/test/emotion/data.npy"
LABEL_PATH = "training/splits/test/emotion/labels.npy"

model = tf.keras.models.load_model(MODEL_PATH)
X_test = np.load(DATA_PATH)[:10]
y_test = np.load(LABEL_PATH)[:10]

X_test_p = []
for img in X_test:
    img = np.expand_dims(img, axis=-1)
    img_resized = tf.image.resize(img, (96, 96)).numpy()
    img_rgb = np.concatenate([img_resized]*3, axis=-1)
    X_test_p.append(img_rgb)

X_test_p = np.array(X_test_p) / 255.0
y_probs = model.predict(X_test_p)
y_pred = np.argmax(y_probs, axis=1)

print(f"True Labels: {y_test}")
print(f"Pred Labels: {y_pred}")
print(f"Max Probs: {np.max(y_probs, axis=1)}")
