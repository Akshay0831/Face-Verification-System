"""Enhanced recognition plugin implementation"""

from core.base import IRecognizer, RecognitionResult, PluginMetadata
from .vgg_face import VGGFaceRecognizer
from .lbph import LBPHRecognizer
from utils import get_logger

logger = get_logger('enhanced_recognition')

__all__ = ['VGGFaceRecognizer', 'LBPHRecognizer']

class EnhancedRecognitionPlugin(IRecognizer):
    """Enhanced recognition plugin wrapper"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.vgg = VGGFaceRecognizer()
        self.lbph = LBPHRecognizer()
        self.vgg.initialize(self.config)
        self.lbph.initialize(self.config)
    
    def get_metadata(self):
        """Return plugin metadata"""
        return PluginMetadata(
            name='enhanced_recognition',
            version='1.0.0',
            description='Enhanced face recognition with VGG-Face and LBPH',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy', 'tensorflow'],
            device_compatibility=['windows', 'linux', 'raspberry_pi']
        )
    
    def recognize(self, face_image, **kwargs):
        """Recognize face using enhanced methods"""
        # Try VGG-Face first
        vgg_result = self.vgg.recognize(face_image)
        if vgg_result.user_id:
            return vgg_result
        
        # Fall back to LBPH
        return self.lbph.recognize(face_image)

__all__.append('EnhancedRecognitionPlugin')