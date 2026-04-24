import os
import sys

def get_base_path():
    """
    Returns the base path for the project.
    Detects if running in Google Colab or locally.
    """
    try:
        import google.colab
        # If in Colab, mount drive and use drive path
        BASE_PATH = '/content/drive/MyDrive/DL/'
        return BASE_PATH
    except ImportError:
        # If local, use current working directory or absolute path to DL folder
        # We'll use the directory where this file resides
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
LOG_PATH = os.path.join(BASE_PATH, 'logs')
MODEL_PATH = os.path.join(BASE_PATH, 'models')
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
PROCESSED_PATH = os.path.join(BASE_PATH, 'processed_data')
OUTPUT_PATH = os.path.join(BASE_PATH, 'outputs')

# Ensure directories exist
for p in [LOG_PATH, MODEL_PATH, DATASET_PATH, PROCESSED_PATH, OUTPUT_PATH]:
    if not os.path.exists(p):
        os.makedirs(p)

# Ensure research subdirectories exist
os.makedirs(os.path.join(MODEL_PATH, 'optimized'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'xai'), exist_ok=True)

print(f"Environment set to: {'COLAB' if 'google.colab' in sys.modules else 'LOCAL'}")
print(f"Base Path: {BASE_PATH}")
