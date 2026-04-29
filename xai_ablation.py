import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, applications
import metric_utils

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """
    Computes Grad-CAM heatmap for a given image and Keras model.
    """
    try:
        # Check if the model is a functional model or sequential
        if isinstance(model.layers[0], models.Model): # Nested backbone
            backbone = model.layers[0]
            grad_model = models.Model([backbone.inputs], [backbone.get_layer(last_conv_layer_name).output, model.output])
        else:
            grad_model = models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, tf.argmax(preds[0])]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"[XAI ERROR] Grad-CAM failed: {e}")
        return np.zeros((7, 7))

def occlusion_sensitivity(model, img, label, patch_size=32):
    """
    Occlude parts of the image and see the drop in confidence.
    """
    img_size = img.shape[0]
    heatmap = np.zeros((img_size, img_size))
    
    try:
        # Baseline prediction
        baseline_pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][label]
        
        for i in range(0, img_size, patch_size):
            for j in range(0, img_size, patch_size):
                occluded_img = img.copy()
                occluded_img[i:i+patch_size, j:j+patch_size, :] = 0
                
                new_pred = model.predict(np.expand_dims(occluded_img, axis=0), verbose=0)[0][label]
                heatmap[i:i+patch_size, j:j+patch_size] = baseline_pred - new_pred
                
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
    except Exception as e:
        print(f"[XAI ERROR] Occlusion failed: {e}")
        
    return heatmap

def run_xai_report(model, img, label, class_names, output_path, last_conv_layer):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Grad-CAM
    heatmap_gc = get_gradcam_heatmap(model, np.expand_dims(img, axis=0), last_conv_layer)
    heatmap_gc_res = cv2.resize(heatmap_gc, (img.shape[1], img.shape[0]))
    
    # 2. Occlusion
    heatmap_occ = occlusion_sensitivity(model, img, label)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Raw Image
    img_display = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[0].imshow(img_display)
    axes[0].set_title(f"Original ({class_names[label]})")
    
    # Grad-CAM
    axes[1].imshow(img_display)
    axes[1].imshow(heatmap_gc_res, cmap='jet', alpha=0.5)
    axes[1].set_title("Grad-CAM Focus")
    
    # Occlusion
    axes[2].imshow(img_display)
    axes[2].imshow(heatmap_occ, cmap='hot', alpha=0.5)
    axes[2].set_title("Occlusion Sensitivity")
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[XAI] Report saved to {output_path}")

def main():
    import sys
    metric_utils.setup_gpu()
    
    model_configs = {
        'resnet': ('DL - imagenet', 'resnet50', 'champion_model_mastery.keras'),
        'mobile': ('DL - mobilenet', 'mobilenetv2', 'champion_model_mastery.keras'),
        'efficient': ('DL - efficientnet b0', 'efficientnetb0', 'champion_model_mastery.keras'),
        'yolo': ('DL -YOLO', 'n/a', 'champion_model.pt')
    }
    
    target = sys.argv[1] if len(sys.argv) > 1 else 'mobile'
    
    if target in model_configs:
        folder, last_conv, model_file = model_configs[target]
        model_path = os.path.join('.', 'models', model_file)
        
        if not os.path.exists(model_path):
             print(f"[ERROR] Model file missing: {model_path}")
             return

        print(f"[XAI] Loading {target} from {model_path}...")
        
        if target == 'yolo':
            from ultralytics import YOLO
            model = YOLO(model_path)
            dataset_path = os.path.join('.', 'dataset')
            img_path = None
            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(root, f)
                        break
                if img_path: break
            
            if img_path:
                model.predict(img_path, visualize=True, project='outputs', name='yolo_xai', exist_ok=True)
                print(f"[XAI] YOLOv8 Visualization saved to outputs/yolo_xai")
            else:
                print("[ERROR] No images found for XAI")
            return

        try:
            model = tf.keras.models.load_model(model_path)
        except:
            sys.path.append(os.path.join(os.path.dirname(os.getcwd()), folder))
            try:
                import train_local
                model, _ = train_local.build_mastery_model(7)
                model.load_weights(model_path)
            except Exception as e:
                print(f"[ERROR] Could not build/load model: {e}")
                return
        
        dataset_path = os.path.join('.', 'dataset')
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset missing: {dataset_path}")
            return

        ds = tf.keras.utils.image_dataset_from_directory(dataset_path, image_size=(224, 224), batch_size=1)
        img, label = next(iter(ds))
        img = img.numpy()[0]
        label = np.argmax(label.numpy()[0])
        class_names = ds.class_names
        
        run_xai_report(model, img, label, class_names, f"outputs/{target}/xai_report.png", last_conv)
    else:
        print(f"XAI Utilities Ready. Usage: python xai_ablation.py [resnet|mobile|efficient|yolo]")

if __name__ == "__main__":
    main()
