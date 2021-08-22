"""LBPH (Local Binary Patterns Histograms) face recognition plugin"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import os
import pickle

from core.base import IRecognizer, RecognitionResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('lbph')


class LBPHRecognizer(IRecognizer):
    """LBPH face recognition implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.radius = self.config.get('radius', 3)
        self.neighbors = self.config.get('neighbors', 8)
        self.grid_x = self.config.get('grid_x', 8)
        self.grid_y = self.config.get('grid_y', 8)
        self.threshold = self.config.get('threshold', 80.0)
        self.labels = {}
        self.reverse_labels = {}
        self.trained = False
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='lbph',
            version='1.0.0',
            description='LBPH face recognition for lightweight systems',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the LBPH recognizer"""
        try:
            self.config.update(config)
            
            # Create LBPH recognizer with custom parameters
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=self.radius,
                neighbors=self.neighbors,
                grid_x=self.grid_x,
            grid_y=self.grid_y
            )
            
            # Try to load existing model
            model_path = self.config.get('model_path', 'lbph_model.yml')
            if os.path.exists(model_path):
                self.recognizer.read(model_path)
                self._load_labels()
                self.trained = True
                logger.info(f"LBPH model loaded from {model_path}")
            else:
                logger.info("No existing model found, training required")
            
            logger.info("LBPH recognizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LBPH recognizer: {e}")
            return False
    
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize face using LBPH"""
        if not self.trained:
            logger.error("LBPH recognizer not trained")
            return RecognitionResult(user_id=None, confidence=0.0)
        
        try:
            # Ensure the image is grayscale for LBPH
            if len(face_image.shape) == 3:
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_image
            
            # Detect face (simplified - assumes already cropped)
            detected_face = self._preprocess_face(face_gray)
            
            if detected_face is None:
                logger.warning("No face detected in the input image")
                return RecognitionResult(user_id=None, confidence=0.0)
            
            # Recognize
            label, confidence = self.recognizer.predict(detected_face)
            
            # Convert label to user ID
            user_id = self.reverse_labels.get(label, None)
            
            # Apply threshold
            if confidence > self.threshold:
                logger.warning(f"Recognition confidence {confidence} above threshold {self.threshold}")
                return RecognitionResult(user_id=None, confidence=0.0)
            
            logger.info(f"LBPH recognized user {user_id} with confidence {confidence:.2f}")
            return RecognitionResult(user_id=user_id, confidence=1.0 - (confidence / 100.0))
            
        except Exception as e:
            logger.error(f"LBPH recognition failed: {e}")
            return RecognitionResult(user_id=None, confidence=0.0)
    
    def register_user(self, user_id: str, face_image: np.ndarray) -> bool:
        """Register a new user with their face image"""
        try:
            # Ensure the image is grayscale
            if len(face_image.shape) == 3:
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_image
            
            # Detect and preprocess face
            detected_face = self._preprocess_face(face_gray)
            
            if detected_face is None:
                logger.error("No face detected in registration image")
                return False
            
            # Assign label to user
            if user_id not in self.labels:
                label = len(self.labels)
                self.labels[user_id] = label
                self.reverse_labels[label] = user_id
            
            # Add to training set
            if not hasattr(self, 'training_images'):
                self.training_images = []
                self.training_labels = []
            
            self.training_images.append(detected_face)
            self.training_labels.append(self.labels[user_id])
            
            logger.info(f"User {user_id} added to training set")
            return True
            
        except Exception as e:
            logger.error(f"Error registering user {user_id}: {e}")
            return False
    
    def train(self) -> bool:
        """Train the LBPH recognizer with accumulated training data"""
        if not hasattr(self, 'training_images') or len(self.training_images) == 0:
            logger.warning("No training data available")
            return False
        
        try:
            # Train the recognizer
            self.recognizer.train(self.training_images, np.array(self.training_labels))
            self.trained = True
            
            # Save model
            model_path = self.config.get('model_path', 'lbph_model.yml')
            self.recognizer.save(model_path)
            self._save_labels()
            
            logger.info(f"LBPH trained successfully with {len(self.training_images)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training LBPH: {e}")
            return False
    
    def _preprocess_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face image for LBPH"""
        try:
            # Resize to standard size (required by LBPH)
            size = (100, 100)  # Standard size for LBPH
            resized = cv2.resize(face_image, size)
            
            # Apply histogram equalization for better contrast
            equalized = cv2.equalizeHist(resized)
            
            return equalized
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None
    
    def _save_labels(self):
        """Save labels to file"""
        try:
            labels_path = self.config.get('labels_path', 'lbph_labels.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        except Exception as e:
            logger.error(f"Error saving labels: {e}")
    
    def _load_labels(self):
        """Load labels from file"""
        try:
            labels_path = self.config.get('labels_path', 'lbph_labels.pkl')
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as f:
                    self.labels = pickle.load(f)
                    self.reverse_labels = {v: k for k, v in self.labels.items()}
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
    
    def get_supported_modes(self) -> List[str]:
        """Return supported recognition modes"""
        return ['standard', 'incremental_learning', 'fast_matching']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['radius', 'neighbors', 'grid_x', 'grid_y', 'threshold', 'model_path']