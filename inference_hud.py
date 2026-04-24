import os
import cv2
import numpy as np
import tensorflow as tf
import time

# 1. Configuration & Model Loading
TFLITE_PATH = os.path.join('models', 'optimized', 'champion_model.tflite')
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

print("--- Initializing Neural Synergy HUD ---")

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Setup Face Detection (Using OpenCV Haar Cascades as fallback for Python 3.13)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. HUD Design Constants
COLORS = {
    'Angry': (0, 0, 255),    # Red
    'Disgust': (0, 255, 0),  # Green
    'Fear': (255, 0, 255),   # Magenta
    'Happy': (0, 255, 255),  # Yellow
    'Neutral': (255, 255, 255), # White
    'Sad': (255, 0, 0),      # Blue
    'Surprise': (255, 165, 0) # Orange
}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("HUD ACTIVE. Press 'Q' to terminate mission.")
    
    prev_time = 0
    pred_buffer = [] # Buffer for smoothing
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        
        # Detect Faces (OpenCV Haar Cascade)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Add 30% margin for better context
                margin = int(w * 0.3)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(w_frame - x, w + 2 * margin)
                h = min(h_frame - y, h + 2 * margin)
                
                if w > 0 and h > 0:
                    # --- MASTERY PREPROCESSING PIPELINE ---
                    face_crop = frame[y:y+h, x:x+w]
                    
                    # 1. Grayscale & CLAHE
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    face_norm = clahe.apply(face_gray)
                    
                    # 2. Resize directly to 224x224 (removed 48x48 bottleneck)
                    face_resized = cv2.resize(face_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
                    
                    # 3. Match 3-channel input format
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
                    
                    # 4. Final input prep (Use [0, 255] range as verified)
                    face_input = np.expand_dims(face_rgb, axis=0).astype(np.float32)
                    
                    # Inference
                    interpreter.set_tensor(input_details[0]['index'], face_input)
                    interpreter.invoke()
                    raw_predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # Smoothing
                    pred_buffer.append(raw_predictions)
                    if len(pred_buffer) > 5:
                        pred_buffer.pop(0)
                    predictions = np.mean(pred_buffer, axis=0)
                    
                    # Draw HUD (Display ALL emotions)
                    sidebar_x = frame.shape[1] - 200
                    cv2.rectangle(frame, (sidebar_x - 10, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
                    
                    for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                        y_pos = 50 + i * 40
                        color = COLORS.get(name, (255, 255, 255))
                        
                        # Bar chart
                        bar_width = int(prob * 150)
                        cv2.rectangle(frame, (sidebar_x, y_pos), (sidebar_x + bar_width, y_pos + 10), color, -1)
                        
                        # Text
                        text = f"{name}: {prob*100:.1f}%"
                        cv2.putText(frame, text, (sidebar_x, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                    # Dominant Emotion
                    dom_idx = np.argmax(predictions)
                    confidence = predictions[dom_idx]
                    dom_label = CLASS_NAMES[dom_idx]
                    
                    if confidence > 0.15:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        cv2.putText(frame, f"{dom_label.upper()} ({confidence*100:.1f}%)", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[dom_label], 2)
                    else:
                        cv2.putText(frame, "ANALYZING...", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Neural Synergy - Real-time Emotion HUD', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Mission Terminated ---")

if __name__ == "__main__":
    main()
