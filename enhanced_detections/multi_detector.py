"""Multi-detector plugin with configurable fallback"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import importlib

from core.base import IDetector, DetectionResult, PluginMetadata, DeviceType
from utils import get_logger
from .hog_detector import HOGDetector
from .viola_jones_detector import ViolaJonesDetector

logger = get_logger('multi_detector')


class MultiDetector(IDetector):
    """Multi-detector plugin with configurable fallback"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detectors = {}
        self.primary_detector = self.config.get('primary_detector', 'basic')
        self.fallback_detectors = self.config.get('fallback_detectors', ['viola_jones', 'hog'])
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.detection_results = []
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='multi_detector',
            version='1.0.0',
            description='Multi-detector with configurable fallback',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy', 'dlib'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize all detectors"""
        try:
            self.config.update(config)
            
            # Initialize basic detector (if available)
            try:
                from ..plugins.detection.basic_detector import BasicFaceDetector
                self.detectors['basic'] = BasicFaceDetector()
                self.detectors['basic'].initialize(self.config.get('basic_config', {}))
                logger.info("Basic detector initialized")
            except ImportError:
                logger.warning("Basic detector not available")
            
            # Initialize HOG detector
            self.detectors['hog'] = HOGDetector()
            self.detectors['hog'].initialize(self.config.get('hog_config', {}))
            logger.info("HOG detector initialized")
            
            # Initialize Viola-Jones detector
            self.detectors['viola_jones'] = ViolaJonesDetector()
            self.detectors['viola_jones'].initialize(self.config.get('viola_config', {}))
            logger.info("Viola-Jones detector initialized")
            
            # Set primary detector
            if self.primary_detector in self.detectors:
                logger.info(f"Primary detector set to: {self.primary_detector}")
            else:
                logger.warning(f"Primary detector {self.primary_detector} not available, using basic")
                self.primary_detector = 'basic' if 'basic' in self.detectors else 'hog'
            
            logger.info("Multi-detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing multi-detector: {e}")
            return False
    
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Detect faces using multi-detector strategy"""
        self.detection_results = []
        
        try:
            # Try primary detector first
            if self.primary_detector in self.detectors:
                primary_results = self.detectors[self.primary_detector].detect(image, **kwargs)
                self.detection_results.extend(primary_results)
                logger.info(f"Primary detector {self.primary_detector} found {len(primary_results)} faces")
            
            # Try fallback detectors if primary detector didn't find enough faces
            min_faces = self.config.get('min_faces', 1)
            
            if len(self.detection_results) < min_faces:
                for detector_name in self.fallback_detectors:
                    if (detector_name != self.primary_detector and 
                        detector_name in self.detectors and
                        len(self.detection_results) < min_faces):
                        
                        fallback_results = self.detectors[detector_name].detect(image, **kwargs)
                        # Add unique detections
                        existing_bboxes = {d.bbox for d in self.detection_results}
                        unique_results = [r for r in fallback_results if r.bbox not in existing_bboxes]
                        
                        self.detection_results.extend(unique_results)
                        logger.info(f"Fallback detector {detector_name} found {len(unique_results)} additional faces")
            
            # Filter by confidence threshold
            if self.confidence_threshold > 0:
                self.detection_results = [
                    r for r in self.detection_results 
                    if r.confidence >= self.confidence_threshold
                ]
            
            logger.info(f"Multi-detection found {len(self.detection_results)} total faces")
            return self.detection_results
            
        except Exception as e:
            logger.error(f"Multi-detection failed: {e}")
            return []
    
    def get_supported_modes(self) -> List[str]:
        """Return supported detection modes"""
        return ['primary_fallback', 'all_detectors', 'single_detector']
    
    def set_mode(self, mode: str) -> bool:
        """Set detection mode"""
        if mode == 'primary_fallback':
            # Primary with fallback - current behavior
            pass
        elif mode == 'all_detectors':
            # Use all detectors and combine results
            self.primary_detector = 'all'
        elif mode == 'single_detector':
            # Use only primary detector
            if self.primary_detector in self.detectors:
                self.fallback_detectors = []
            else:
                logger.warning(f"Primary detector {self.primary_detector} not available")
                return False
        else:
            logger.error(f"Unknown detection mode: {mode}")
            return False
        return True
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Return detection statistics"""
        stats = {
            'total_detections': len(self.detection_results),
            'detectors_used': [],
            'confidence_scores': [d.confidence for d in self.detection_results],
            'primary_detector': self.primary_detector,
            'fallback_detectors': self.fallback_detectors
        }
        
        # Add detector-specific stats
        for detector_name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_detection_stats'):
                    stats[f'{detector_name}_stats'] = detector.get_detection_stats()
                    stats['detectors_used'].append(detector_name)
            except Exception as e:
                logger.warning(f"Could not get stats from {detector_name}: {e}")
        
        return stats
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['primary_detector', 'fallback_detectors', 'confidence_threshold', 'min_faces']
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get statistics about detector performance"""
        return {
            'primary_detector': self.primary_detector,
            'available_detectors': list(self.detectors.keys()),
            'fallback_order': self.fallback_detectors,
            'total_detections': len(self.detection_results),
            'confidence_threshold': self.confidence_threshold
        }