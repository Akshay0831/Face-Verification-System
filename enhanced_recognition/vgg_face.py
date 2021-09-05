"""VGG-Face recognition plugin"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
import os

from core.base import IRecognizer, RecognitionResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('vgg_face')


class VGGFaceRecognizer(IRecognizer):
    """VGG-Face recognition implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.model_path = self.config.get('model_path', None)
        self.target_size = self.config.get('target_size', (224, 224))
        self.preprocess_input = tf.keras.applications.vgg16.preprocess_input
        self.threshold = self.config.get('threshold', 0.45)
        self.embeddings = {}
        self.users = {}
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='vgg_face',
            version='1.0.0',
            description='VGG-Face recognition with TensorFlow',
            author='Face Verification System Team',
            dependencies=['tensorflow', 'opencv-python', 'numpy'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the VGG-Face recognizer"""
        try:
            self.config.update(config)
            
            # Load pre-trained VGG-Face model
            if self.model_path and os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info("VGG-Face model loaded from file")
            else:
                # Create VGG-Face model using VGGFace library
                try:
                    # Try to use VGGFace library if available
                    from keras_vggface.vggface import VGGFace
                    self.model = VGGFace(model='resnet50', weights='vggface', include_top=False)
                    # Add custom pooling and embedding layer
                    x = self.model.output
                    x = GlobalAveragePooling2D()(x)
                    x = Dense(128, activation='linear')(x)
                    self.model = Model(inputs=self.model.inputs, outputs=x)
                    logger.info("VGG-Face model loaded with VGGFace weights")
                except ImportError:
                    logger.debug("VGGFace library not available, using VGG16 fallback")
                    # Fallback to VGG16 with custom top layers
                    base_model = VGG16(
                        weights=None,  # Random initialization - will be trained on face data
                        include_top=False,
                        input_shape=(224, 224, 3)
                    )
                    
                    # Add custom top layers for face recognition
                    x = base_model.output
                    x = GlobalAveragePooling2D()(x)
                    x = Dense(512, activation='relu')(x)
                    x = Dropout(0.5)(x)
                    predictions = Dense(128, activation='linear')(x)
                    
                    self.model = Model(inputs=base_model.input, outputs=predictions)
                    logger.info("VGG-Face model created with VGG16 base (random weights)")
            
            # Create embedding extractor
            self.embedding_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=self.model.layers[-2].output  # Get before final embedding layer
            )
            
            logger.info("VGG-Face recognizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing VGG-Face recognizer: {e}")
            return False
    
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize face using VGG-Face"""
        if self.model is None:
            logger.error("VGG-Face model not initialized")
            return RecognitionResult(user_id=None, confidence=0.0)
        
        try:
            # Preprocess image
            processed_face = cv2.resize(face_image, self.target_size)
            processed_face = image.img_to_array(processed_face)
            processed_face = np.expand_dims(processed_face, axis=0)
            processed_face = self.preprocess_input(processed_face)
            
            # Extract embedding
            embedding = self.embedding_model.predict(processed_face)[0]
            
            # Compare with known embeddings
            best_match = self._find_best_match(embedding)
            
            return RecognitionResult(
                user_id=best_match['user_id'],
                confidence=best_match['confidence']
            )
            
        except Exception as e:
            logger.error(f"VGG-Face recognition failed: {e}")
            return RecognitionResult(user_id=None, confidence=0.0)
    
    def enroll(self, user_id: str, face_image: np.ndarray) -> bool:
        """Enroll a user with a face image"""
        return self.register_user(user_id, face_image)
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Get face embedding from image"""
        try:
            # Preprocess image
            processed_face = cv2.resize(face_image, self.target_size)
            processed_face = image.img_to_array(processed_face)
            processed_face = np.expand_dims(processed_face, axis=0)
            processed_face = self.preprocess_input(processed_face)
            
            # Extract embedding
            embedding = self.embedding_model.predict(processed_face)[0]
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return np.array([])
    
    def register_user(self, user_id: str, face_image: np.ndarray) -> bool:
        """Register a new user with their face image"""
        try:
            # Generate embedding for the face
            processed_face = cv2.resize(face_image, self.target_size)
            processed_face = image.img_to_array(processed_face)
            processed_face = np.expand_dims(processed_face, axis=0)
            processed_face = self.preprocess_input(processed_face)
            
            embedding = self.embedding_model.predict(processed_face)[0]
            
            # Store embedding and user info
            user_key = f"user_{len(self.users) + 1}"
            self.embeddings[user_key] = embedding
            self.users[user_key] = user_id
            
            logger.info(f"User {user_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering user {user_id}: {e}")
            return False
    
    def _find_best_match(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Find the best matching user for an embedding"""
        if not self.embeddings:
            return {'user_id': None, 'confidence': 0.0}
        
        similarities = []
        for user_key, stored_embedding in self.embeddings.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding, stored_embedding)
            user_id = self.users[user_key]
            similarities.append({'user_id': user_id, 'confidence': similarity})
        
        # Sort by similarity and return best match
        similarities.sort(key=lambda x: x['confidence'], reverse=True)
        best_match = similarities[0]
        
        return best_match
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_supported_modes(self) -> List[str]:
        """Return supported recognition modes"""
        return ['standard', 'batch_processing', 'incremental_learning']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['model_path', 'target_size', 'threshold']