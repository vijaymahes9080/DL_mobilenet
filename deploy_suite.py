import shutil
import os

model_folders = ['DL -YOLO', 'DL - imagenet', 'DL - efficientnet b0', 'DL - mobilenet']
files_to_copy = [
    'AUTO_TEST_MODELS.py',
    'hyper_tuner.py',
    'metric_utils.py',
    'xai_ablation.py',
    'hyper_tuning_results.csv',
    'generate_sim_results.py'
]

for folder in model_folders:
    suite_dir = os.path.join(folder, 'MASTERY_SUITE')
    os.makedirs(suite_dir, exist_ok=True)
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(suite_dir, file))
            print(f"Copied {file} to {suite_dir}")

print("Mastery Suite deployed to all model folders.")
