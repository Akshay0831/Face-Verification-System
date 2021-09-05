"""Enhanced liveness detection plugin implementation"""

from core.base import ILivenessDetector, LivenessResult, PluginMetadata
from .motion_analyzer import MotionAnalyzer
from utils import get_logger

logger = get_logger('enhanced_liveness')

__all__ = ['MotionAnalyzer']

class EnhancedLivenessPlugin(ILivenessDetector):
    """Enhanced liveness detection plugin wrapper"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.motion_analyzer = MotionAnalyzer()
        self.motion_analyzer.initialize(self.config)
    
    def get_metadata(self):
        """Return plugin metadata"""
        return PluginMetadata(
            name='enhanced_liveness',
            version='1.0.0',
            description='Enhanced liveness detection with motion analysis',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy'],
            device_compatibility=['windows', 'linux', 'raspberry_pi']
        )
    
    def check_liveness(self, face_images, **kwargs):
        """Check liveness using enhanced methods"""
        return self.motion_analyzer.check_liveness(face_images, **kwargs)

__all__.append('EnhancedLivenessPlugin')