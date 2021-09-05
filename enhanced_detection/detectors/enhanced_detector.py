"""Enhanced detector module"""

from typing import Dict, Any, List
from enhanced_detections.hog_detector import HOGDetector
from enhanced_detections.multi_detector import MultiDetector
from enhanced_detections.viola_jones_detector import ViolaJonesDetector


class HaarDetector(ViolaJonesDetector):
    """Haar Detector alias of ViolaJonesDetector for test suite compatibility"""
    pass


class EnhancedDetector(MultiDetector):
    """Enhanced face detector inheriting multi-detector with fallback capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config)
