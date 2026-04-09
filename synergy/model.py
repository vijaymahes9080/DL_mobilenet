import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_efficientnet_synergy(num_classes, sensor_dim=None, backbone_v="B0", input_shape=(224, 224, 3)):
    """
    Builds the EfficientNet-Synergy Hybrid Model.
    - Backbone: EfficientNet (B0-B7)
    - Inputs: Image + Optional Sensor Vector
    - Fusion: Concatenation
    - Activation: Swish
    """
    # 1. Image Backbone
    # Map backbone string to class
    backbone_map = {
        "B0": applications.EfficientNetV2B0,
        "B1": applications.EfficientNetV2B1,
        "B2": applications.EfficientNetV2B2,
        "B3": applications.EfficientNetV2B3,
        "S":  applications.EfficientNetV2S, # Fixed typo
        "M":  applications.EfficientNetV2M,
        "L":  applications.EfficientNetV2L,
    }
    
    # Auto-select B0 for stability on CPU if not specified
    backbone_cls = backbone_map.get(backbone_v, applications.EfficientNetV2B0)
    
    base_model = backbone_cls(include_top=False, weights='imagenet', input_shape=input_shape)
    
    image_input = layers.Input(shape=input_shape, name="image_input")
    # Preprocessing is often built into EfficientNetV2, but we ensure normalization
    x = base_model(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    
    # 2. Sensor Input (Optional)
    if sensor_dim and sensor_dim > 0:
        sensor_input = layers.Input(shape=(sensor_dim,), name="sensor_input")
        # Ensure sensor data is processed before fusion
        s = layers.Dense(64, activation=tf.nn.swish)(sensor_input)
        x = layers.Concatenate()([x, s])
        inputs = [image_input, sensor_input]
    else:
        inputs = image_input
        
    # 3. Dense Head
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256)(x)
    x = layers.Activation(tf.nn.swish)(x)
    
    # 4. Output Layer
    # Logic: if num_classes is 1, use sigmoid, else softmax
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNet_Synergy_Hybrid")
    
    return model, base_model

def build_resnet_synergy(num_classes, sensor_dim=None, input_shape=(224, 224, 3)):
    """
    Builds the ResNet-Synergy Hybrid Model.
    """
    base_model = applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    image_input = layers.Input(shape=input_shape, name="image_input")
    x = base_model(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    
    if sensor_dim and sensor_dim > 0:
        sensor_input = layers.Input(shape=(sensor_dim,), name="sensor_input")
        s = layers.Dense(64, activation='relu')(sensor_input)
        x = layers.Concatenate()([x, s])
        inputs = [image_input, sensor_input]
    else:
        inputs = image_input

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_Synergy")
    return model, base_model

def build_mobilenet_synergy(num_classes, sensor_dim=None, input_shape=(224, 224, 3)):
    """
    Builds the MobileNetV3-Synergy Hybrid Model (Real-time focus).
    """
    base_model = applications.MobileNetV3Small(include_top=False, weights='imagenet', input_shape=input_shape)
    image_input = layers.Input(shape=input_shape, name="image_input")
    x = base_model(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    
    if sensor_dim and sensor_dim > 0:
        sensor_input = layers.Input(shape=(sensor_dim,), name="sensor_input")
        s = layers.Dense(32, activation='relu')(sensor_input)
        x = layers.Concatenate()([x, s])
        inputs = [image_input, sensor_input]
    else:
        inputs = image_input

    x = layers.Dense(256, activation='relu')(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNet_Synergy")
    return model, base_model

def build_temporal_synergy(num_classes, input_shape=(None, 64)):
    """
    Builds an LSTM-based Temporal Synergy Model for sequential data.
    """
    inputs = layers.Input(shape=input_shape, name="temporal_input")
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dense(256, activation='relu')(x)
    
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    model = models.Model(inputs=inputs, outputs=outputs, name="Temporal_Synergy")
    return model

if __name__ == "__main__":
    # Test build
    model, base = build_efficientnet_synergy(num_classes=7, sensor_dim=10)
    model.summary()
    print("Model build successful.")
