import os
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Configuration & Model Loading
TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

print("--- Initializing 10x Neural Synergy HUD (MobileNet) ---")

# Load TFLite model with Multi-Threading
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=8)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

COLORS = {
    'Angry': (0, 0, 255),    'Disgust': (0, 255, 0),  'Fear': (255, 0, 255),
    'Happy': (0, 255, 255),  'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
    'Surprise': (255, 165, 0)
}

class FastVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    def update(self):
        while True:
            if self.stopped: return
            self.ret, self.frame = self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    fvs = FastVideoStream(0).start()
    time.sleep(1.0)
    
    print("HUD ACTIVE (10X MODE). Press 'Q' to terminate.")
    
    prev_time = 0
    pred_buffer = [] 
    last_preds = np.zeros(len(CLASS_NAMES))
    frame_count = 0
    
    while True:
        frame = fvs.read()
        if frame is None: break
        frame_count += 1
        
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                margin = int(w * 0.10)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(w_frame - x, w + 2 * margin)
                h = min(h_frame - y, h + 2 * margin)
                
                if w > 0 and h > 0:
                    face_crop = frame[y:y+h, x:x+w]
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    face_norm = clahe.apply(face_gray)
                    face_resized = cv2.resize(face_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
                    
                    # 10x Async Inference: Every 2 frames
                    if frame_count % 2 == 0:
                        face_input = np.expand_dims(face_rgb, axis=0).astype(np.float32)
                        face_input = preprocess_input(face_input)
                        interpreter.set_tensor(input_details[0]['index'], face_input)
                        interpreter.invoke()
                        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                        
                        boosts = {'Disgust': 1.5, 'Surprise': 1.2, 'Fear': 1.1}
                        for i, name in enumerate(CLASS_NAMES):
                            if name in boosts: predictions[i] *= boosts[name]
                        
                        happy_idx = CLASS_NAMES.index('Happy') if 'Happy' in CLASS_NAMES else -1
                        sad_idx = CLASS_NAMES.index('Sad') if 'Sad' in CLASS_NAMES else -1
                        if happy_idx != -1 and sad_idx != -1:
                            if predictions[happy_idx] > 0.15 and predictions[sad_idx] > predictions[happy_idx]:
                                predictions[happy_idx] += 0.10
                        
                        predictions = np.clip(predictions, 0, 1)
                        predictions /= (np.sum(predictions) + 1e-6)
                        last_preds = predictions
                    else:
                        predictions = last_preds

                    pred_buffer.append(predictions)
                    if len(pred_buffer) > 15: pred_buffer.pop(0)
                    predictions = np.mean(pred_buffer, axis=0)
                    
                    sidebar_x = frame.shape[1] - 200
                    cv2.rectangle(frame, (sidebar_x - 10, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)

                    for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                        y_pos = 50 + i * 40
                        color = COLORS.get(name, (255, 255, 255))
                        bar_width = int(np.sqrt(prob) * 150)
                        cv2.rectangle(frame, (sidebar_x, y_pos), (sidebar_x + bar_width, y_pos + 10), color, -1)
                        cv2.putText(frame, f"{name}: {prob*100:.1f}%", (sidebar_x, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                    dom_idx = np.argmax(predictions)
                    confidence = predictions[dom_idx]
                    dom_label = CLASS_NAMES[dom_idx]
                    
                    if confidence > 0.15:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        cv2.putText(frame, f"{dom_label.upper()} ({confidence*100:.1f}%)", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[dom_label], 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"10X_SPEED | FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Neural Synergy - MobileNet HUD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    fvs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
