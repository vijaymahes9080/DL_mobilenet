import os
import cv2
import numpy as np
import tensorflow as tf

TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DATASET_PATH = 'dataset'

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def verify_emotion(emotion):
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
    
    print(f"True: {emotion:10} | Pred: {CLASS_NAMES[np.argmax(preds)]:10} | Confidence: {np.max(preds):.4f}")

for emotion in CLASS_NAMES:
    verify_emotion(emotion)
