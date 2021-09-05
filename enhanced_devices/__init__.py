"""Enhanced device management plugin implementation"""

from core.base import IDevice, DeviceType, PluginMetadata
from core.device_manager import DeviceManager, Device
from utils import get_logger

logger = get_logger('enhanced_devices')

__all__ = ['DeviceManager', 'Device', 'DeviceType']

class EnhancedDevicePlugin(IDevice):
    """Enhanced device management plugin wrapper"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.device_manager = DeviceManager()
        self.device_type = DeviceType.WINDOWS  # Default type
    
    def get_metadata(self):
        """Return plugin metadata"""
        return PluginMetadata(
            name='enhanced_devices',
            version='1.0.0',
            description='Enhanced device management with multiple device support',
            author='Face Verification System Team',
            dependencies=['opencv-python', 'numpy'],
            device_compatibility=['windows', 'linux', 'raspberry_pi', 'camera']
        )
    
    def get_device_type(self):
        """Get the device type"""
        return self.device_type
    
    def get_supported_performance_modes(self):
        """Get supported performance modes"""
        from core.base import PerformanceMode
        return [PerformanceMode.STANDARD]
    
    def optimize_for_performance(self, mode):
        """Optimize device for specific performance mode"""
        return True
    
    def get_system_resources(self):
        """Get current system resources"""
        return {'cpu': 0, 'memory': 0, 'gpu': 0}
    
    def initialize(self, config):
        """Initialize the device with configuration"""
        self.config.update(config)
        return True

__all__.append('EnhancedDevicePlugin')