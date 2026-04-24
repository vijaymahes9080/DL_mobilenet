import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DATASET_PATH = 'dataset'

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

y_true = []
y_pred = []

for idx, emotion in enumerate(CLASS_NAMES):
    path = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(path): continue
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Sample 50 images per class for speed
    for f in files[:50]:
        img = cv2.imread(os.path.join(path, f))
        if img is None: continue
        
        # Preprocessing
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_input = np.expand_dims(face_resized, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], face_input)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        y_true.append(idx)
        y_pred.append(np.argmax(preds))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Check for off-diagonal dominance
for i in range(len(CLASS_NAMES)):
    print(f"Class {CLASS_NAMES[i]}: True Positives={cm[i,i]}, Most common pred={CLASS_NAMES[np.argmax(cm[i])]}")
