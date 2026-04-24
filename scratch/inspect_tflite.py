import tensorflow as tf
import os

model_path = os.path.join('models', 'optimized', 'champion_model.tflite')
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
else:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
