import cv2
import time
import numpy as np
import tensorflow as tf
import logging

log = logging.getLogger("SynergyInference")

class SynergyRealTime:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Auto-detect input shape
        try:
             self.img_size = self.model.input_shape[1:3]
             log.info(f"✅ Detected input shape: {self.img_size}")
        except:
             self.img_size = (224, 224)
        
    def start_camera(self):
        """
        Starts real-time inference loop using OpenCV.
        """
        cap = cv2.VideoCapture(0)
        log.info("📸 Camera started. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess
            processed_frame = cv2.resize(frame, self.img_size)
            processed_frame = np.expand_dims(processed_frame, axis=0) / 255.0
            
            # Inference
            start_time = time.time()
            prediction = self.model.predict(processed_frame, verbose=0)
            latency = (time.time() - start_time) * 1000
            
            # Label
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Display
            label = f"Class: {class_idx} ({confidence:.2f}) - Latency: {latency:.1f}ms"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Synergy Real-Time Hybrid", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def convert_to_tflite(self, export_path="models/synergy/synergy_optimized.tflite"):
        """
        Optimize model for low latency.
        """
        log.info("⚡ Converting to TFLite for optimization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(export_path, "wb") as f:
            f.write(tflite_model)
        log.info(f"✅ Optimized model saved to {export_path}")
        return export_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synergy Real-Time Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.keras)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    rt = SynergyRealTime(args.model)
    rt.start_camera()
