"""Enhanced recognizer module"""

from typing import Dict, Any, Optional
import numpy as np
from core.base import IRecognizer, RecognitionResult
from enhanced_recognition.vgg_face import VGGFaceRecognizer
from enhanced_recognition.lbph import LBPHRecognizer


class EnhancedRecognizer(IRecognizer):
    """Enhanced face recognizer supporting ensemble of deep learning & traditional methods"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vgg = VGGFaceRecognizer(self.config)
        self.lbph = LBPHRecognizer(self.config)

    def recognize(self, face_image: np.ndarray, **kwargs) -> RecognitionResult:
        try:
            res = self.vgg.recognize(face_image, **kwargs)
            if res and getattr(res, 'match_found', False):
                return res
        except Exception:
            pass
        return self.lbph.recognize(face_image, **kwargs)
