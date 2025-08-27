import cv2
import numpy as np

def preprocess_frame(frame):
    """Konversi frame ke grayscale untuk deteksi."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def eye_aspect_ratio(eye):
    """Hitung Eye Aspect Ratio (EAR) dari landmark mata."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear