import os
import requests
from tqdm import tqdm
import sys

# Ensure UTF-8 output for Windows Console to prevent charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Handle case where stdout doesn't support reconfigure

# 🧬 ORIEN: PRE-TRAINED NEURAL SEED DOWNLOADER
# ==========================================
# Downloads the "Seeds" of Optimized Neural Models (MobileNetV2, ResNet50V2)
# used as high-performance foundations for ORIEN assistants.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'models', 'v0')
os.makedirs(MODELS_DIR, exist_ok=True)

SEEDS = {
    'face_base_v0.h5': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5',
    'face_emotion_base_v0.h5': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'gesture_base_v0.h5': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5'
}

def download_file(url, filename):
    local_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(local_path):
        print(f"✅ Found existing seed: {filename}")
        return

    print(f"⬇️ Downloading Neural Seed: {filename}...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(local_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        print(f"✅ Saved Optimized Seed to: {local_path}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  ORIEN NEURAL MASTER: SEED DOWNLOADER")
    print("  Booting up Optimized brain foundations...")
    print("" + "="*50 + "\n")
    
    for filename, url in SEEDS.items():
        download_file(url, filename)
        
    print(f"\n🌱 Pre-Training Seeds (v0) saved to: {MODELS_DIR}")
    print("Ready to enrich these with your personal data in Stage 1!")
