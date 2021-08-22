"""Bink detection for liveness verification"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque

from core.base import ILivenessDetector, LivenessResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('blink_detector')


class BlinkDetector(ILivenessDetector):
    """Liveness detection using blink analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.eye_cascade_path = self.config.get('eye_cascade_path', 'haarcascade_eye.xml')
        self.face_cascade_path = self.config.get('face_cascade_path', 'cascades/haarcascade_frontalface_default.xml')
        self.min_eyes = self.config.get('min_eyes', 1)
        self.blink_threshold = self.config.get('blink_threshold', 20.0)
        self.blink_ratio_threshold = self.config.get('blink_ratio_threshold', 5.0)
        self.ear_threshold = self.config.get('ear_threshold', 0.25)
        self.frames_history = deque(maxlen=self.config.get('history_length', 30))
        self.eyes_detected_history = deque(maxlen=self.config.get('history_length', 30))
        
        # Cascade classifiers
        self.face_cascade = None
        self.eye_cascade = None
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='blink_detector',
            version='1.0.0',
            description='Liveness detection using blink analysis',
            author='Face Verification System Team',
            dependencies=['opencv-python'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the blink detector"""
        try:
            self.config.update(config)
            
            # Load cascade classifiers
            face_cascade_path = self.config.get('face_cascade_path', 'cascades/haarcascade_frontalface_default.xml')
            eye_cascade_path = self.config.get('eye_cascade_path', 'haarcascade_eye.xml')
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                logger.error("Could not load cascade classifiers")
                return False
            
            logger.info("Blink detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing blink detector: {e}")
            return False
    
    def check_liveness(self, face_images: List[np.ndarray]) -> LivenessResult:
        """Check liveness using blink analysis"""
        if not face_images:
            return LivenessResult(is_live=False, confidence=0.0)
        
        try:
            blink_count = 0
            total_eyes = 0
            live_faces = 0
            
            for face_image in face_images:
                # Convert to grayscale
                if len(face_image.shape) == 3:
                    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = face_image
                
                # Detect face
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    logger.warning("No faces detected in image")
                    continue
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    # Detect eyes
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    
                    if len(eyes) >= self.min_eyes:
                        total_eyes += len(eyes)
                        
                        # Calculate Eye Aspect Ratio (EAR) for blink detection
                        ear = self._calculate_ear(roi_gray, eyes)
                        
                        # Check for blink (low EAR indicates closed eyes)
                        if ear < self.ear_threshold:
                            self.frames_history.append(ear)
                            self.eyes_detected_history.append(True)
                        else:
                            self.frames_history.append(ear)
                            self.eyes_detected_history.append(False)
                        
                        # Check for blink pattern
                        if self._detect_blink():
                            blink_count += 1
                            logger.info(f"Blink detected in face at ({x}, {y})")
                        
                        live_faces += 1
                    else:
                        logger.warning(f"Insufficient eyes ({len(eyes)}) detected in face")
            
            # Calculate confidence based on blink patterns
            if total_eyes > 0 and live_faces > 0:
                blink_ratio = blink_count / live_faces
                confidence = min(1.0, blink_ratio / self.blink_ratio_threshold)
                
                # More eyes detected increases confidence
                eye_confidence = min(1.0, total_eyes / (self.min_eyes * len(face_images)))
                final_confidence = (confidence + eye_confidence) / 2
                
                is_live = final_confidence > 0.5  # Threshold for liveness
                
                details = {
                    'blink_count': blink_count,
                    'total_eyes': total_eyes,
                    'live_faces': live_faces,
                    'blink_ratio': blink_ratio,
                    'average_ear': np.mean(self.frames_history) if self.frames_history else 0.0
                }
                
                logger.info(f"Liveness check: {is_live}, confidence: {final_confidence:.2f}")
                return LivenessResult(is_live=is_live, confidence=final_confidence, details=details)
            
            logger.warning("No valid faces detected for liveness check")
            return LivenessResult(is_live=False, confidence=0.0)
            
        except Exception as e:
            logger.error(f"Error in liveness check: {e}")
            return LivenessResult(is_live=False, confidence=0.0)
    
    def _calculate_ear(self, face_region: np.ndarray, eyes: List) -> float:
        """Calculate Eye Aspect Ratio"""
        try:
            if len(eyes) < 2:
                return 0.0
            
            # Use the first two eyes for EAR calculation
            eye1 = eyes[0]
            eye2 = eyes[1] if len(eyes) > 1 else eye1
            
            # Eye 1
            x1, y1, w1, h1 = eye1
            eye1_center = (x1 + w1 // 2, y1 + h1 // 2)
            
            # Eye 2
            x2, y2, w2, h2 = eye2
            eye2_center = (x2 + w2 // 2, y2 + h2 // 2)
            
            # Calculate distances
            vertical_distance = abs(eye1_center[1] - eye2_center[1])
            horizontal_distance = abs(eye1_center[0] - eye2_center[0])
            
            # Calculate aspect ratio
            if vertical_distance > 0:
                ear = horizontal_distance / vertical_distance
                return ear
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0
    
    def _detect_blink(self) -> bool:
        """Detect blink pattern in recent frames"""
        try:
            if len(self.frames_history) < 10:  # Need enough history
                return False
            
            # Check for significant drop in EAR (eyes closing)
            recent_values = list(self.frames_history)[-10:]
            avg_ear = np.mean(recent_values)
            min_ear = np.min(recent_values)
            
            # If there's a significant drop, it indicates blink
            if (min_ear < self.ear_threshold and 
                avg_ear < self.ear_threshold * 1.5):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting blink: {e}")
            return False
    
    def get_supported_modes(self) -> List[str]:
        """Return supported liveness detection modes"""
        return ['blink_analysis', 'multi_frame_analysis']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['eye_cascade_path', 'face_cascade_path', 'min_eyes', 
                'blink_threshold', 'blink_ratio_threshold', 'ear_threshold']