"""Modular Face Verification System"""

__version__ = "0.0.1"
__author__ = "Face Verification System Team"


# Import core components
from core import (
    DeviceType, PerformanceMode, DetectionResult, RecognitionResult,
    LivenessResult, NotificationResult, PluginMetadata,
    IDetector, IRecognizer, ILivenessDetector, INotifier, 
    IDevice, IProcessor, IStorage, IPlugin,
    PluginManager, DeviceManager
)

# Import main system class
from system import FaceVerificationSystem

__all__ = [
    'DeviceType', 'PerformanceMode', 'DetectionResult', 'RecognitionResult',
    'LivenessResult', 'NotificationResult', 'PluginMetadata',
    'IDetector', 'IRecognizer', 'ILivenessDetector', 'INotifier',
    'IDevice', 'IProcessor', 'IStorage', 'IPlugin',
    'PluginManager', 'DeviceManager', 'FaceVerificationSystem',
    '__version__', '__author__'
]