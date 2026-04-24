import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")
try:
    print(f"Solutions: {mp.solutions}")
    print("Success: mediapipe.solutions is available.")
except AttributeError as e:
    print(f"Error: {e}")
    print("Listing mediapipe attributes:")
    print(dir(mp))
