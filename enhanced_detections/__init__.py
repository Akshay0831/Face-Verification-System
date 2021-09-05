"""Enhanced detection plugin implementation"""

from core.base import IDetector, DetectionResult, PluginMetadata
from .multi_detector import MultiDetector
from utils import get_logger

logger = get_logger('enhanced_detection')

__all__ = ['MultiDetector']

class EnhancedDetectionPlugin(IDetector):
    """Enhanced detection plugin wrapper"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.multi_detector = MultiDetector()
        self.multi_detector.initialize(self.config)
    
    def get_metadata(self):
        """Return plugin metadata"""
        return PluginMetadata(
            name='enhanced_detection',
            version='1.0.0',
            description='Enhanced face detection with multi-strategy approach',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy', 'dlib'],
            device_compatibility=['windows', 'linux', 'raspberry_pi']
        )
    
    def detect(self, image, **kwargs):
        """Detect faces using enhanced detection"""
        return self.multi_detector.detect(image, **kwargs)

__all__.append('EnhancedDetectionPlugin')