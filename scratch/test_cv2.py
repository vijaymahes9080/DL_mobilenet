import cv2
import os

cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
print(f"Cascade path: {cascade_path}")
if os.path.exists(cascade_path):
    print("Cascade file found!")
else:
    print("Cascade file NOT found.")
