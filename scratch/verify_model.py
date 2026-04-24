import os
import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def verify_emotion(emotion):
    dataset_path = os.path.join('dataset', emotion)
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} not found.")
        return
    
    files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"No images in {dataset_path}.")
        return
    
    # Pick first 5 images
    for i in range(min(5, len(files))):
        img_path = os.path.join(dataset_path, files[i])
        img = cv2.imread(img_path)
        
        # Preprocessing (same as in inference_hud.py)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_input = np.expand_dims(face_resized, axis=0).astype(np.float32)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], face_input)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        pred_idx = np.argmax(predictions)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = predictions[pred_idx]
        
        print(f"True: {emotion} | Pred: {pred_label} ({confidence:.2f}) | {files[i]}")

for emotion in CLASS_NAMES:
    verify_emotion(emotion)
