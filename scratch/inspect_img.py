import cv2
import numpy as np

img_path = r'D:\current project\DL\dataset\Angry\img_10.jpg'
img = cv2.imread(img_path)
if img is not None:
    print(f"Shape: {img.shape}")
    print(f"Mean values (BGR): {np.mean(img, axis=(0,1))}")
    # Check if it's likely grayscale (B==G==R)
    is_gray = np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,1] == img[:,:,2])
    print(f"Is grayscale: {is_gray}")
else:
    print("Could not read image.")
