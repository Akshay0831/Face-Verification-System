"""Motion analysis for liveness verification"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque

from core.base import ILivenessDetector, LivenessResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('motion_analyzer')


class MotionAnalyzer(ILivenessDetector):
    """Liveness detection using motion analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.motion_threshold = self.config.get('motion_threshold', 15.0)
        self.min_motion_frames = self.config.get('min_motion_frames', 5)
        self.max_static_frames = self.config.get('max_static_frames', 10)
        self.motion_history = deque(maxlen=self.config.get('history_length', 30))
        self.frame_history = deque(maxlen=self.config.get('frame_history_size', 5))
        self.previous_gray = None
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='motion_analyzer',
            version='1.0.0',
            description='Liveness detection using motion analysis',
            author='Face Verification System Team',
            dependencies=['opencv-python'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the motion analyzer"""
        try:
            self.config.update(config)
            
            # Reset state
            self.frame_history.clear()
            self.motion_history.clear()
            self.previous_gray = None
            
            logger.info("Motion analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing motion analyzer: {e}")
            return False
    
    def check_liveness(self, face_images: List[np.ndarray]) -> LivenessResult:
        """Check liveness using motion analysis"""
        if not face_images:
            return LivenessResult(is_live=False, confidence=0.0)
        
        try:
            motion_detected = 0
            total_frames = len(face_images)
            consecutive_static = 0
            motion_scores = []
            
            for i, face_image in enumerate(face_images):
                # Convert to grayscale
                if len(face_image.shape) == 3:
                    current_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                else:
                    current_gray = face_image
                
                # Initialize on first frame
                if self.previous_gray is None:
                    self.previous_gray = current_gray.copy()
                    self.frame_history.append(current_gray)
                    motion_scores.append(0.0)
                    continue
                
                # Calculate motion using frame difference
                motion_score = self._calculate_motion(current_gray, self.previous_gray)
                motion_scores.append(motion_score)
                self.motion_history.append(motion_score)
                
                # Check if motion exceeds threshold
                if motion_score > self.motion_threshold:
                    motion_detected += 1
                    consecutive_static = 0
                    logger.debug(f"Motion detected: {motion_score:.2f} at frame {i}")
                else:
                    consecutive_static += 1
                
                # Update previous frame
                self.previous_gray = current_gray.copy()
                self.frame_history.append(current_gray)
            
            # Analyze motion patterns
            motion_ratio = motion_detected / total_frames if total_frames > 0 else 0
            avg_motion_score = np.mean(motion_scores) if motion_scores else 0
            
            # Check for sufficient motion
            sufficient_motion = motion_ratio > (self.min_motion_frames / total_frames) if total_frames > 0 else False
            
            # Check for reasonable static periods (not completely static)
            reasonable_static = consecutive_static < self.max_static_frames
            
            # Calculate confidence
            if sufficient_motion and reasonable_static:
                confidence = min(1.0, motion_ratio * 2)
            else:
                confidence = motion_ratio
            
            # Additional confidence from average motion score
            motion_score_confidence = min(1.0, avg_motion_score / self.motion_threshold)
            final_confidence = (confidence + motion_score_confidence) / 2
            
            is_live = final_confidence > 0.3  # Threshold for liveness
            
            details = {
                'motion_detected': motion_detected,
                'total_frames': total_frames,
                'motion_ratio': motion_ratio,
                'avg_motion_score': avg_motion_score,
                'consecutive_static': consecutive_static,
                'sufficient_motion': sufficient_motion,
                'reasonable_static': reasonable_static
            }
            
            logger.info(f"Liveness check: {is_live}, confidence: {final_confidence:.2f}")
            return LivenessResult(is_live=is_live, confidence=final_confidence, details=details)
            
        except Exception as e:
            logger.error(f"Error in liveness check: {e}")
            return LivenessResult(is_live=False, confidence=0.0)
    
    def _calculate_motion(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> float:
        """Calculate motion score between two frames"""
        try:
            # Calculate absolute difference
            diff = cv2.absdiff(current_frame, previous_frame)
            
            # Threshold the difference
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Dilate to make motion areas more visible
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Calculate motion score
            motion_pixels = np.count_nonzero(dilated)
            total_pixels = current_frame.shape[0] * current_frame.shape[1]
            
            # Normalize motion score
            motion_score = (motion_pixels / total_pixels) * 1000  # Scale up for better detection
            
            return motion_score
            
        except Exception as e:
            logger.error(f"Error calculating motion: {e}")
            return 0.0
    
    def get_supported_modes(self) -> List[str]:
        """Return supported liveness detection modes"""
        return ['motion_analysis', 'temporal_consistency', 'background_subtraction']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['motion_threshold', 'min_motion_frames', 'max_static_frames', 
                'history_length', 'frame_history_size']