import os
dataset_path = 'dataset'
for cls in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls)
    if os.path.isdir(cls_path):
        count = 0
        for root, dirs, files in os.walk(cls_path):
            count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{cls}: {count}")
