"""Viola-Jones face detection plugin for embedded systems"""

import cv2
import numpy as np
from typing import List, Dict, Any
import os

from core.base import IDetector, DetectionResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('viola_jones_detector')


class ViolaJonesDetector(IDetector):
    """Viola-Jones face detector optimized for embedded systems"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detector = None
        self.cascade_path = self.config.get('cascade_path', None)
        self.scale_factor = self.config.get('scale_factor', 1.1)
        self.min_neighbors = self.config.get('min_neighbors', 3)
        self.min_size = tuple(self.config.get('min_size', (30, 30)))
        self.max_size = tuple(self.config.get('max_size', (300, 300)))
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='viola_jones_detector',
            version='1.0.0',
            description='Viola-Jones face detection for embedded systems',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy'],
            device_compatibility=[DeviceType.RASPBERRY_PI, DeviceType.WINDOWS, DeviceType.LINUX]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the Viola-Jones detector"""
        try:
            self.config.update(config)
            
            # Load cascade classifier
            if self.cascade_path and os.path.exists(self.cascade_path):
                self.detector = cv2.CascadeClassifier(self.cascade_path)
            else:
                # Fallback to built-in cascade
                cascade_path = os.path.join('..', 'cascades', 'haarcascade_frontalface_default.xml')
                if os.path.exists(cascade_path):
                    self.detector = cv2.CascadeClassifier(cascade_path)
                else:
                    logger.debug("No cascade classifier found, using basic detection")
                    self.detector = None
            
            if self.detector is None:
                logger.error("Could not load cascade classifier")
                return False
            
            logger.info("Viola-Jones detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Viola-Jones detector: {e}")
            return False
    
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Detect faces using Viola-Jones algorithm"""
        if self.detector is None:
            logger.error("Viola-Jones detector not initialized")
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                maxSize=self.max_size
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detection = DetectionResult(
                    bbox=(x, y, w, h),
                    confidence=1.0,  # Viola-Jones doesn't provide confidence
                    face_image=None
                )
                detections.append(detection)
            
            logger.info(f"Viola-Jones detection found {len(detections)} faces")
            return detections
        
        except Exception as e:
            logger.error(f"Viola-Jones detection failed: {e}")
            return []
            
        logger.info(f"Viola-Jones detection found {len(detections)} faces")
        return detections
    
    def get_supported_modes(self) -> List[str]:
        """Return supported detection modes"""
        return ['standard', 'embedded_optimized']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['cascade_path', 'scale_factor', 'min_neighbors', 'min_size', 'max_size']