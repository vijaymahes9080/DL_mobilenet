import os
import cv2
import numpy as np
import tensorflow as tf

TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def verify_emotion(emotion):
    dataset_path = os.path.join('dataset', emotion)
    if not os.path.exists(dataset_path): return
    files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files: return
    
    img_path = os.path.join(dataset_path, files[0])
    img = cv2.imread(img_path)
    face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    
    # Test [0, 255]
    face_input_255 = np.expand_dims(face_resized, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], face_input_255)
    interpreter.invoke()
    preds_255 = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Test [-1, 1]
    face_input_neg1 = (face_input_255 / 127.5) - 1.0
    interpreter.set_tensor(input_details[0]['index'], face_input_neg1)
    interpreter.invoke()
    preds_neg1 = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print(f"--- Emotion: {emotion} ---")
    print(f"[0, 255] Pred: {CLASS_NAMES[np.argmax(preds_255)]} ({np.max(preds_255):.2f})")
    print(f"[-1, 1]  Pred: {CLASS_NAMES[np.argmax(preds_neg1)]} ({np.max(preds_neg1):.2f})")

for emotion in CLASS_NAMES:
    verify_emotion(emotion)
