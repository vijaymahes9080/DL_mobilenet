import cv2
import numpy as np
import os

img_paths = [
    r'D:\current project\DL\dataset\Happy\img_10019.jpg',
    r'D:\current project\DL\dataset\Angry\img_10.jpg',
    r'D:\current project\DL\dataset\Neutral\img_10003.jpg'
]

for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is not None:
        print(f"File: {os.path.basename(img_path)}")
        print(f"  Shape: {img.shape}")
        is_gray = np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,1] == img[:,:,2])
        print(f"  Is grayscale: {is_gray}")
    else:
        print(f"Could not read {img_path}")
