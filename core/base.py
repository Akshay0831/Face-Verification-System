"""Abstract base classes for modular face verification system plugins."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
from utils import get_logger

logger = get_logger('base')


class DeviceType(Enum):
    """Device types for deployment targets"""
    RASPBERRY_PI = "raspberry_pi"
    WINDOWS = "windows"
    LINUX = "linux"
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"
    CAMERA = "camera"


class PerformanceMode(Enum):
    """Performance processing modes"""
    STANDARD = "standard"      # 15-30 FPS
    HIGH_SPEED = "high_speed"  # 30-60 FPS  
    ULTRA_HIGH = "ultra_high"  # 60+ FPS


class DetectionResult:
    """Face detection plugin result"""
    def __init__(self, 
                 bbox: Tuple[int, int, int, int],  # (x, y, w, h)
                 confidence: float,
                 face_image: Optional[np.ndarray] = None):
        self.bbox = bbox
        self.confidence = confidence
        self.face_image = face_image


class RecognitionResult:
    """Face recognition plugin result"""
    def __init__(self, 
                 user_id: Optional[str] = None,
                 confidence: float = 0.0,
                 embedding: Optional[np.ndarray] = None):
        self.user_id = user_id
        self.confidence = confidence
        self.embedding = embedding


class LivenessResult:
    """Result from liveness detection plugins"""
    def __init__(self, 
                 is_live: bool = False,
                 confidence: float = 0.0,
                 details: Optional[Dict[str, Any]] = None):
        self.is_live = is_live
        self.confidence = confidence
        self.details = details or {}


class NotificationResult:
    """Result from notification plugins"""
    def __init__(self, 
                 success: bool = False,
                 message: str = "",
                 details: Optional[Dict[str, Any]] = None):
        self.success = success
        self.message = message
        self.details = details or {}


# Base interfaces for all plugins

class IDetector(ABC):
    """Abstract base class for face detection plugins"""
    
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Detect faces in an image"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the detector with configuration"""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[str]:
        """Get supported detection modes"""
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters"""
        pass


class IRecognizer(ABC):
    """Abstract base class for face recognition plugins"""
    
    @abstractmethod
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize a face from an image"""
        pass
    
    @abstractmethod
    def enroll(self, user_id: str, face_image: np.ndarray) -> bool:
        """Enroll a user with a face image"""
        pass
    
    @abstractmethod
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Get face embedding from image"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the recognizer with configuration"""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[str]:
        """Get supported recognition modes"""
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters"""
        pass


class ILivenessDetector(ABC):
    """Abstract base class for liveness detection plugins"""
    
    @abstractmethod
    def check_liveness(self, face_images: List[np.ndarray]) -> LivenessResult:
        """Check if faces are live using temporal analysis"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the liveness detector with configuration"""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[str]:
        """Get supported liveness detection modes"""
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters"""
        pass


class INotifier(ABC):
    """Abstract base class for notification plugins"""
    
    @abstractmethod
    def send_notification(self, 
                         user_id: Optional[str] = None,
                         message: str = "",
                         **kwargs) -> NotificationResult:
        """Send a notification"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the notifier with configuration"""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[str]:
        """Get supported notification modes"""
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters"""
        pass


class IDevice(ABC):
    """Abstract base class for device-specific implementations"""
    
    @abstractmethod
    def get_device_type(self) -> DeviceType:
        """Get the device type"""
        pass
    
    @abstractmethod
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        """Get supported performance modes for this device"""
        pass
    
    @abstractmethod
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        """Optimize device for specific performance mode"""
        pass
    
    @abstractmethod
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resources"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the device with configuration"""
        pass


class IProcessor(ABC):
    """Abstract base class for processing pipeline plugins"""
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process data through this pipeline stage"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the processor with configuration"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported input/output formats"""
        pass


class IStorage(ABC):
    """Abstract base class for storage plugins"""
    
    @abstractmethod
    def store(self, key: str, data: Any, **kwargs) -> bool:
        """Store data with given key"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str, **kwargs) -> Any:
        """Retrieve data by key"""
        pass
    
    @abstractmethod
    def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key"""
        pass
    
    @abstractmethod
    def list_keys(self, **kwargs) -> List[str]:
        """List all available keys"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the storage with configuration"""
        pass


class PluginMetadata:
    """Metadata for plugin information"""
    def __init__(self, 
                 name: str,
                 version: str,
                 description: str = "",
                 author: str = "",
                 dependencies: Optional[List[str]] = None,
                 device_compatibility: Optional[List[DeviceType]] = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.device_compatibility = device_compatibility or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'device_compatibility': [dt.value for dt in self.device_compatibility] if self.device_compatibility else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create PluginMetadata from dictionary"""
        device_compat = [DeviceType(dt) for dt in data.get('device_compatibility', [])] if data.get('device_compatibility') else None
        return cls(
            name=data.get('name', ''),
            version=data.get('version', ''),
            description=data.get('description', ''),
            author=data.get('author', ''),
            dependencies=data.get('dependencies', []),
            device_compatibility=device_compat
        )


class IPlugin(ABC):
    """Base interface for all plugins"""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup resources when plugin is unloaded"""
        pass


class ICamera(ABC):
    """Abstract base class for camera devices"""
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        pass
    
    @abstractmethod
    def start_capture(self) -> bool:
        """Start frame capture"""
        pass
    
    @abstractmethod
    def stop_capture(self) -> bool:
        """Stop frame capture"""
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution"""
        pass
    
    @abstractmethod
    def set_fps(self, fps: int) -> bool:
        """Set camera frame rate"""
        pass