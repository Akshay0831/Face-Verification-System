"""HOG-based face detection plugin"""

import cv2
import numpy as np
from typing import List, Dict, Any

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from core.base import IDetector, DetectionResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('hog_detector')


class HOGDetector(IDetector):
    """HOG-based face detector using dlib"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if not DLIB_AVAILABLE:
            raise ImportError("dlib is required for HOGDetector but not installed")
        
        self.config = config or {}
        self.detector = None
        self.downscale = self.config.get('downscale', 1.3)
        self.scale_sizes = self.config.get('scale_sizes', [60, 100, 140, 180, 220])
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='hog_detector',
            version='1.0.0',
            description='HOG-based face detection using dlib',
            author='Face Verification System Team',
            dependencies=['dlib', 'opencv-python', 'numpy'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the HOG detector"""
        try:
            self.config.update(config)
            
            # Create HOG face detector
            self.detector = dlib.get_frontal_face_detector()
            
            logger.info("HOG detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing HOG detector: {e}")
            return False
    
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Detect faces using HOG algorithm"""
        if self.detector is None:
            logger.error("HOG detector not initialized")
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces at multiple scales
            detections = []
            for scale in self.scale_sizes:
                # Downscale image for detection
                if scale < gray.shape[0] and scale < gray.shape[1]:
                    scaled_size = (int(gray.shape[1] * scale / max(gray.shape)), 
                                 int(gray.shape[0] * scale / max(gray.shape)))
                    scaled_gray = cv2.resize(gray, scaled_size)
                    
                    # Detect faces
                    faces = self.detector(scaled_gray, self.downscale)
                    
                    # Scale back coordinates
                    for face in faces:
                        x = int(face.left() * max(gray.shape) / scale)
                        y = int(face.top() * max(gray.shape) / scale)
                        w = int((face.right() - face.left()) * max(gray.shape) / scale)
                        h = int((face.bottom() - face.top()) * max(gray.shape) / scale)
                        
                        detection = DetectionResult(
                            bbox=(x, y, w, h),
                            confidence=0.8,  # HOG doesn't provide confidence
                            face_image=None
                        )
                        detections.append(detection)
            
            # Remove duplicates (NMS would be better but keeping it simple)
            unique_detections = []
            seen_bboxes = set()
            
            for detection in detections:
                bbox = detection.bbox
                if bbox not in seen_bboxes:
                    seen_bboxes.add(bbox)
                    unique_detections.append(detection)
            
            logger.info(f"HOG detection found {len(unique_detections)} faces")
            return unique_detections
            
        except Exception as e:
            logger.error(f"HOG detection failed: {e}")
            return []
    
    def get_supported_modes(self) -> List[str]:
        """Return supported detection modes"""
        return ['standard', 'multi_scale']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['downscale', 'scale_sizes']