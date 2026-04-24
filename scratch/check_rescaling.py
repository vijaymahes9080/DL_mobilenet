import tensorflow as tf
import os

model_path = os.path.join('models', 'champion_model_mastery.keras')
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    # Check for Rescaling layer
    for layer in model.layers:
        print(f"Layer: {layer.name}")
        if 'rescaling' in layer.name.lower():
            print("FOUND RESCALING LAYER")
else:
    print("Model not found.")
