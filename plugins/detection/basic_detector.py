"""OpenCV-based face detection plugin"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import os

from core.base import IDetector, DetectionResult, DeviceType, PluginMetadata
from utils import get_logger

logger = get_logger('basic_detector')


class BasicFaceDetector(IDetector):
    """OpenCV cascade classifier face detector"""
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='basic_detector',
            version='1.0.0',
            description='Basic face detection using OpenCV cascade classifiers',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    """Simple face detection plugin using OpenCV"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.detector = None
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.cascade_path = self.config.get('face_cascade_path', None)
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize detector with configuration"""
        self.config.update(config)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.face_cascade_path = config.get('face_cascade_path', None)
        
        try:
            # Try to load cascade classifier
            cascade_path = self.face_cascade_path
            if not cascade_path:
                # Try to find cascade files in common locations
                cascade_paths = [
                    self.face_cascade_path,  # Use configured path first
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'cascades', 'haarcascade_frontalface_default.xml'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'cascades', 'haarcascade_frontalface_alt2.xml'),
                    os.path.join('cascades', 'haarcascade_frontalface_default.xml'),
                    os.path.join('cascades', 'haarcascade_frontalface_alt2.xml'),
                    'haarcascade_frontalface_default.xml',
                    'haarcascade_frontalface_alt2.xml'
                ]
                
                for path in cascade_paths:
                    if os.path.exists(path):
                        cascade_path = path
                        break
            
            # Check if CascadeClassifier is available in current OpenCV version
            if hasattr(cv2, 'CascadeClassifier'):
                if cascade_path and os.path.exists(cascade_path):
                    self.detector = cv2.CascadeClassifier(cascade_path)
                    if self.detector.empty():
                        logger.error(f"Failed to load cascade classifier from {cascade_path}")
                        return False
                    logger.info(f"Loaded cascade classifier from {cascade_path}")
                else:
                    logger.error(f"No cascade classifier found or specified")
                    return False
            else:
                # CascadeClassifier not available in this OpenCV version, use alternative
                logger.warning("CascadeClassifier not available in current OpenCV version, using basic edge detection")
                self.detector = None
                self.use_edge_detection = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            return False
    
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Detect faces in an image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if hasattr(self, 'use_edge_detection') and self.use_edge_detection:
                # Use basic edge detection as fallback
                return self._detect_with_edges(gray)
            elif self.detector is not None:
                # Use cascade classifier
                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                results = []
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_image = image[y:y+h, x:x+w]
                    
                    result = DetectionResult(
                        bbox=(x, y, w, h),
                        confidence=0.8,  # Approximate confidence
                        face_image=face_image
                    )
                    results.append(result)
                
                return results
            else:
                logger.error("Detector not initialized")
                return []
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _detect_with_edges(self, gray_image: np.ndarray) -> List[DetectionResult]:
        """Basic face detection using edge detection as fallback"""
        try:
            # Simple edge detection approach
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find contours in the edge image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            results = []
            image_height, image_width = gray_image.shape
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio (basic face filtering)
                if (w > 50 and h > 50 and 
                    w < image_width * 0.8 and h < image_height * 0.8 and
                    0.5 < w/h < 2.0):  # Face-like aspect ratio
                    
                    # Extract face region
                    face_image = gray_image[y:y+h, x:x+w]
                    
                    result = DetectionResult(
                        bbox=(x, y, w, h),
                        confidence=0.6,  # Lower confidence for edge detection
                        face_image=face_image
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in edge detection: {e}")
            return []
    
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters"""
        return ['min_confidence']
    
    def get_supported_modes(self) -> List[str]:
        """Get supported detection modes"""
        return ['standard', 'high_speed']
    
    def cleanup(self):
        """Clean up resources"""
        self.detector = None
        logger.info("BasicFaceDetector cleaned up")
    
    def get_device_requirements(self) -> List[DeviceType]:
        """Get supported device types"""
        return [DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
    
    def get_performance_modes(self) -> List[str]:
        """Get supported performance modes"""
        return ['standard', 'high_speed']