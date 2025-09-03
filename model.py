import os
from typing import Tuple

import cv2
import numpy as np

# Global variable to cache the cascade classifier
_FACE_CASCADE = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Get or create the Haar cascade classifier for face detection"""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        # Try to find the cascade file in common locations
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                _FACE_CASCADE = cv2.CascadeClassifier(path)
                break
        
        if _FACE_CASCADE is None:
            # Fallback: try to load from OpenCV data directory
            try:
                _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except:
                raise RuntimeError("Could not load Haar cascade classifier. Make sure OpenCV is properly installed.")
    
    return _FACE_CASCADE


def detect_faces(image_path: str) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    Detect faces in an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of face rectangles (x, y, width, height)
    """
    if not os.path.exists(image_path):
        return tuple()
    
    image = cv2.imread(image_path)
    if image is None:
        return tuple()
    
    return detect_faces_from_image(image)


def detect_faces_from_image(image: np.ndarray) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    Detect faces in an image array.
    
    Args:
        image: BGR image array
        
    Returns:
        Tuple of face rectangles (x, y, width, height)
    """
    if image is None or image.size == 0:
        return tuple()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the cascade classifier
    face_cascade = _get_face_cascade()
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return tuple(tuple(map(int, f)) for f in faces)