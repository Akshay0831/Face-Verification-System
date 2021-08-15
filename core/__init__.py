"""
Core framework components for the modular face verification system.
"""

from .base import (
    DeviceType, PerformanceMode, DetectionResult, RecognitionResult,
    LivenessResult, NotificationResult, IDetector, IRecognizer, 
    ILivenessDetector, INotifier, IDevice, IProcessor, IStorage, 
    PluginMetadata, IPlugin
)
from .plugin_manager import PluginManager
from .device_abstraction import DeviceManager, FallbackDevice

__all__ = [
    'DeviceType', 'PerformanceMode', 'DetectionResult', 'RecognitionResult',
    'LivenessResult', 'NotificationResult', 'IDetector', 'IRecognizer',
    'ILivenessDetector', 'INotifier', 'IDevice', 'IProcessor', 'IStorage',
    'PluginMetadata', 'IPlugin', 'PluginManager', 'DeviceManager', 
    'FallbackDevice'
]