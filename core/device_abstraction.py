"""Device abstraction layer for different deployment targets."""

import os
import sys
import threading
import time
from typing import Dict, Any, List, Optional
import psutil
import cv2

from .base import IDevice, DeviceType, PerformanceMode
from utils import get_logger

logger = get_logger('device_abstraction')


class DeviceManager:
    """Device-specific implementations and optimizations"""
    
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.current_device = None
        self.device_optimizations = {}
        self.performance_mode = PerformanceMode.STANDARD
        
    def detect_current_device(self) -> DeviceType:
        """Auto-detect current device type"""
        try:
            # Check for Android
            if os.path.exists('/system/framework') or 'android' in sys.platform.lower():
                return DeviceType.ANDROID
            
            # Check for iOS
            if sys.platform == 'darwin' and os.path.exists('/Applications'):
                return DeviceType.IOS
                
            # Check for Raspberry Pi
            if os.path.exists('/proc/device-tree/model') and 'raspberry' in open('/proc/device-tree/model').read().lower():
                return DeviceType.RASPBERRY_PI
                
            # Check for Windows
            if sys.platform == 'win32':
                return DeviceType.WINDOWS
                
            # Check for Linux
            if sys.platform.startswith('linux'):
                return DeviceType.LINUX
                
            # Default to web/server
            return DeviceType.WEB
            
        except Exception as e:
            logger.warning(f"Error detecting device type: {e}")
            return DeviceType.LINUX
    
    def initialize_device(self, config: Dict[str, Any]) -> bool:
        """Initialize the appropriate device plugin"""
        device_type = self.detect_current_device()
        logger.info(f"Detected device type: {device_type}")
        
        # Get device plugins
        device_plugins = self.plugin_manager.get_device_plugins()
        
        # Try to find a compatible device plugin
        device_plugin = device_plugins.get(device_type)
        
        if device_plugin is None:
            logger.warning(f"No device plugin found for {device_type}, using fallback")
            # Create a basic device plugin
            device_plugin = FallbackDevice(device_type)
        
        # Initialize device plugin
        device_config = config.get('device', {})
        if device_plugin.initialize(device_config):
            self.current_device = device_plugin
            logger.info(f"Initialized device plugin: {device_plugin.get_device_type()}")
            return True
        
        logger.error(f"Failed to initialize device plugin")
        return False
    
    def get_current_device(self) -> IDevice:
        """Get the current device instance"""
        if self.current_device is None:
            self.initialize_device({})
        return self.current_device
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        """Get performance modes supported by current device"""
        device = self.get_current_device()
        return device.get_supported_performance_modes()
    
    def set_performance_mode(self, mode: PerformanceMode) -> bool:
        """Set the performance mode for the current device"""
        try:
            device = self.get_current_device()
            if device.optimize_for_performance(mode):
                self.performance_mode = mode
                logger.info(f"Set performance mode to: {mode}")
                return True
            else:
                logger.error(f"Failed to optimize for performance mode: {mode}")
                return False
        except Exception as e:
            logger.error(f"Error setting performance mode: {e}")
            return False
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resources"""
        device = self.get_current_device()
        return device.get_system_resources()
    
    def optimize_for_environment(self) -> bool:
        """Optimize settings based on current environment"""
        try:
            # Get system resources
            resources = self.get_system_resources()
            
            # Set performance mode based on available resources
            cpu_percent = resources.get('cpu_percent', 0)
            memory_percent = resources.get('memory_percent', 0)
            available_memory = resources.get('available_memory_mb', 0)
            
            # Determine optimal performance mode
            if cpu_percent > 80 or memory_percent > 80:
                # High system load, use standard mode
                target_mode = PerformanceMode.STANDARD
            elif available_memory > 2000 and cpu_percent < 50:
                # Plenty of resources, use high speed mode
                target_mode = PerformanceMode.HIGH_SPEED
            else:
                # Default to standard mode
                target_mode = PerformanceMode.STANDARD
            
            return self.set_performance_mode(target_mode)
            
        except Exception as e:
            logger.error(f"Error optimizing for environment: {e}")
            return False


class FallbackDevice(IDevice):
    """Fallback device implementation for unknown systems"""
    
    def __init__(self, device_type: DeviceType):
        self.device_type = device_type
    
    def get_device_type(self) -> DeviceType:
        return self.device_type
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        return [PerformanceMode.STANDARD]
    
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        # Basic optimization for fallback device
        logger.info(f"Fallback device optimization: {mode}")
        return True
    
    def get_system_resources(self) -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'available_memory_mb': memory.available // (1024 * 1024),
                'total_memory_mb': memory.total // (1024 * 1024),
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'platform': sys.platform,
                'python_version': sys.version
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        logger.info(f"Initializing fallback device for {self.device_type}")
        return True


class RaspberryPiDevice(IDevice):
    """Optimized device implementation for Raspberry Pi"""
    
    def __init__(self):
        self.device_type = DeviceType.RASPBERRY_PI
        self.camera_available = False
        self.gpu_acceleration = False
        self.thermal_threshold = 75  # Celsius
    
    def get_device_type(self) -> DeviceType:
        return self.device_type
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        return [PerformanceMode.STANDARD, PerformanceMode.HIGH_SPEED]
    
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        try:
            if mode == PerformanceMode.HIGH_SPEED:
                # Enable GPU acceleration if available
                try:
                    import picamera
                    self.camera_available = True
                    self.gpu_acceleration = True
                    logger.info("Enabled GPU acceleration for high speed mode")
                except ImportError:
                    self.camera_available = False
                    self.gpu_acceleration = False
                    logger.warning("picamera not available, using OpenCV")
                
                # Reduce frame quality for performance
                self._set_camera_settings()
            else:
                # Standard mode - more conservative settings
                self.gpu_acceleration = False
                logger.info("Using standard performance mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing Raspberry Pi for performance: {e}")
            return False
    
    def _set_camera_settings(self):
        """Set camera-specific optimizations for Raspberry Pi"""
        try:
            # This would typically interface with Raspistill or similar
            # For now, just set OpenCV preferences
            cv2.ocl.setUseOpenCL(self.gpu_acceleration)
        except:
            pass  # Ignore if not available
    
    def get_system_resources(self) -> Dict[str, Any]:
        try:
            # Get Raspberry Pi specific metrics
            import psutil
            
            cpu_temp = self._get_cpu_temperature()
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_temp_celsius': cpu_temp,
                'memory_percent': memory.percent,
                'available_memory_mb': memory.available // (1024 * 1024),
                'total_memory_mb': memory.total // (1024 * 1024),
                'camera_available': self.camera_available,
                'gpu_acceleration': self.gpu_acceleration,
                'thermal_status': 'OK' if cpu_temp < self.thermal_threshold else 'HOT'
            }
        except Exception as e:
            logger.error(f"Error getting Raspberry Pi resources: {e}")
            return {}
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature for Raspberry Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = float(f.read()) / 1000.0
                return temp_c
        except:
            return 0.0
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        logger.info("Initializing Raspberry Pi device")
        
        # Apply Raspberry Pi specific optimizations
        try:
            # Disable desktop environment for better performance
            os.environ['DISPLAY'] = ':0'
            
            # Set OpenCL preferences
            cv2.ocl.setUseOpenCL(config.get('use_gpu', False))
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Raspberry Pi device: {e}")
            return False


class WindowsDevice(IDevice):
    """Optimized device implementation for Windows systems"""
    
    def __init__(self):
        self.device_type = DeviceType.WINDOWS
        self.gpu_acceleration = True
        self.multithreading_enabled = True
    
    def get_device_type(self) -> DeviceType:
        return self.device_type
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        return list(PerformanceMode)
    
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        try:
            if mode == PerformanceMode.ULTRA_HIGH:
                # Enable maximum optimizations
                self.gpu_acceleration = True
                self.multithreading_enabled = True
                # Set OpenCV to use GPU if available
                cv2.ocl.setUseOpenCL(True)
            elif mode == PerformanceMode.HIGH_SPEED:
                # High performance settings
                self.gpu_acceleration = True
                self.multithreading_enabled = True
                cv2.ocl.setUseOpenCL(True)
            else:
                # Standard mode - balanced settings
                self.gpu_acceleration = False
                self.multithreading_enabled = False
                cv2.ocl.setUseOpenCL(False)
            
            logger.info(f"Windows device optimized for {mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing Windows for performance: {e}")
            return False
    
    def get_system_resources(self) -> Dict[str, Any]:
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('C:\\')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'available_memory_mb': memory.available // (1024 * 1024),
                'total_memory_mb': memory.total // (1024 * 1024),
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'gpu_acceleration': self.gpu_acceleration,
                'multithreading_enabled': self.multithreading_enabled,
                'platform': 'Windows',
                'cpu_cores': psutil.cpu_count(),
                'available_disk_gb': disk.free // (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting Windows system resources: {e}")
            return {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        logger.info("Initializing Windows device")
        
        # Apply Windows specific optimizations
        try:
            # Enable GPU acceleration if requested
            use_gpu = config.get('gpu_acceleration', True)
            cv2.ocl.setUseOpenCL(use_gpu)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Windows device: {e}")
            return False


class LinuxDevice(IDevice):
    """Optimized device implementation for Linux systems"""
    
    def __init__(self):
        self.device_type = DeviceType.LINUX
        self.gpu_acceleration = False
        self.hardware_acceleration = False
    
    def get_device_type(self) -> DeviceType:
        return self.device_type
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        return list(PerformanceMode)
    
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        try:
            # Check for hardware acceleration
            self.hardware_acceleration = self._check_hardware_acceleration()
            
            if mode == PerformanceMode.ULTRA_HIGH and self.hardware_acceleration:
                # Enable maximum optimizations
                self.gpu_acceleration = True
                cv2.ocl.setUseOpenCL(True)
            elif mode == PerformanceMode.HIGH_SPEED:
                # High performance settings
                self.gpu_acceleration = self.hardware_acceleration
                cv2.ocl.setUseOpenCL(self.hardware_acceleration)
            else:
                # Standard mode - conservative settings
                self.gpu_acceleration = False
                cv2.ocl.setUseOpenCL(False)
            
            logger.info(f"Linux device optimized for {mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing Linux for performance: {e}")
            return False
    
    def _check_hardware_acceleration(self) -> bool:
        """Check if hardware acceleration is available"""
        try:
            # Check for OpenCL support
            return cv2.ocl.haveOpenCL()
        except:
            return False
    
    def get_system_resources(self) -> Dict[str, Any]:
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'available_memory_mb': memory.available // (1024 * 1024),
                'total_memory_mb': memory.total // (1024 * 1024),
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'gpu_acceleration': self.gpu_acceleration,
                'hardware_acceleration': self.hardware_acceleration,
                'platform': 'Linux',
                'cpu_cores': psutil.cpu_count(),
                'available_disk_gb': disk.free // (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting Linux system resources: {e}")
            return {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        logger.info("Initializing Linux device")
        
        # Apply Linux specific optimizations
        try:
            # Check for and enable hardware acceleration
            use_gpu = config.get('gpu_acceleration', False)
            if use_gpu:
                cv2.ocl.setUseOpenCL(self.hardware_acceleration)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Linux device: {e}")
            return False