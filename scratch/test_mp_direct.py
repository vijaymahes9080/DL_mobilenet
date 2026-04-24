try:
    import mediapipe.solutions.face_detection as mp_face_detection
    print("Success: import mediapipe.solutions.face_detection worked!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
