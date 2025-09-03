import os
from typing import Dict, Tuple
import cv2
import numpy as np
from model import detect_faces_from_image


class FaceRecognizerService:
    def __init__(self, dataset_dir: str, model_path: str) -> None:
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Try to use LBPH recognizer, fallback to basic face detection
        self.use_advanced_recognition = False
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.use_advanced_recognition = True
            print("Using advanced LBPH face recognition")
        except AttributeError:
            print("LBPH not available, using basic face detection mode")
            self.recognizer = None
        
        # Lower distances are better. Anything above this is considered unknown.
        try:
            self.unknown_distance_threshold: float = float(os.environ.get('RECOG_UNKNOWN_DISTANCE', '65'))
        except ValueError:
            self.unknown_distance_threshold = 65.0
        
        # Try to load existing model, if it fails, we'll retrain from dataset
        self.model_loaded = False
        if self.use_advanced_recognition and os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                self.model_loaded = True
                print(f"Successfully loaded existing model from {self.model_path}")
            except cv2.error as e:
                print(f"Failed to load existing model: {e}. Will retrain from dataset.")
                self.model_loaded = False
        
        # If no model was loaded, try to train from existing dataset
        if not self.model_loaded and self.use_advanced_recognition:
            self.train_from_dataset()

    def train_from_dataset(self) -> None:
        if not self.use_advanced_recognition:
            print("Advanced recognition not available, skipping training")
            return
            
        images: list[np.ndarray] = []
        labels: list[int] = []

        print(f"Training recognizer from dataset in {self.dataset_dir}")
        
        for entry in os.listdir(self.dataset_dir):
            if not entry.startswith('person_'):
                continue
            person_id_str = entry.split('_', 1)[1]
            try:
                person_id = int(person_id_str)
            except ValueError:
                continue
            person_dir = os.path.join(self.dataset_dir, entry)
            if not os.path.isdir(person_dir):
                continue
                
            person_image_count = 0
            for file in os.listdir(person_dir):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(person_id)
                person_image_count += 1
            
            if person_image_count > 0:
                print(f"Loaded {person_image_count} images for person_{person_id}")

        if not images:
            print("No training images found in dataset")
            return

        print(f"Training model with {len(images)} images from {len(set(labels))} people")
        try:
            self.recognizer.train(images, np.array(labels))
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.recognizer.write(self.model_path)
            self.model_loaded = True
            print(f"Model trained and saved to {self.model_path}")
        except Exception as e:
            print(f"Error training model: {e}")
            self.model_loaded = False

    def predict(self, image_bgr: np.ndarray) -> Dict:
        faces = detect_faces_from_image(image_bgr)
        if not faces:
            return {'ok': False, 'error': 'No face detected'}

        # Use first face only
        x, y, w, h = faces[0]
        face_img = image_bgr[y:y+h, x:x+w]
        if face_img.size == 0:
            return {'ok': False, 'error': 'Face crop failed'}
        
        # Normalize input: grayscale, equalize, resize
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)

        if not self.use_advanced_recognition or not self.model_loaded:
            # Fallback: return a generic response for face detection
            return {'ok': False, 'error': 'Face recognition not available - face detected but cannot identify person'}

        try:
            label, distance = self.recognizer.predict(gray)
        except cv2.error as e:
            return {'ok': False, 'error': f'Model prediction error: {e}'}

        # For LBPH, lower distance is better. Reject if distance above threshold.
        if float(distance) > self.unknown_distance_threshold:
            return {'ok': False, 'error': 'Unknown face', 'distance': float(distance)}

        # Map to a 0-100 score (higher is better) for display only
        score = max(0.0, 100.0 - float(distance))
        return {'ok': True, 'personId': int(label), 'confidence': score, 'distance': float(distance)}

    def get_status(self) -> Dict:
        """Get the current status of the recognizer"""
        return {
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'dataset_dir': self.dataset_dir,
            'threshold': self.unknown_distance_threshold,
            'advanced_recognition': self.use_advanced_recognition
        }

    def force_retrain(self) -> bool:
        """Force retrain the model from the dataset"""
        try:
            self.train_from_dataset()
            return self.model_loaded
        except Exception as e:
            print(f"Error during forced retrain: {e}")
            return False
