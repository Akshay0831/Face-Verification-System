"""
Core framework components for the modular face verification system.
"""

from .base import (
    DeviceType, PerformanceMode, DetectionResult, RecognitionResult,
    LivenessResult, NotificationResult, IDetector, IRecognizer, 
    ILivenessDetector, INotifier, IDevice, IProcessor, IStorage, 
    PluginMetadata, IPlugin, ICamera
)
from .plugin_manager import PluginManager
from .device_manager import DeviceManager, Device, CameraDevice
from .core_system import FaceVerificationSystem

__all__ = [
    'DeviceType', 'PerformanceMode', 'DetectionResult', 'RecognitionResult',
    'LivenessResult', 'NotificationResult', 'IDetector', 'IRecognizer',
    'ILivenessDetector', 'INotifier', 'IDevice', 'IProcessor', 'IStorage',
    'PluginMetadata', 'IPlugin', 'PluginManager', 'DeviceManager', 
    'Device', 'CameraDevice', 'FaceVerificationSystem', 'ICamera'
]