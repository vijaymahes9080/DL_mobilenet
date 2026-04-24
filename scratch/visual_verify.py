import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DATASET_PATH = 'dataset'

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def save_verification_image(emotion):
    path = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(path): return
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files: return
    
    img_path = os.path.join(path, files[0])
    img = cv2.imread(img_path)
    
    # Preprocessing
    face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_input = np.expand_dims(face_resized, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    pred_idx = np.argmax(preds)
    pred_label = CLASS_NAMES[pred_idx]
    
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"True: {emotion} | Pred: {pred_label}")
    plt.axis('off')
    plt.savefig(f"scratch/verify_{emotion}.png")
    plt.close()

for emotion in CLASS_NAMES:
    save_verification_image(emotion)
