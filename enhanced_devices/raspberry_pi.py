"""Raspberry Pi device optimization plugin"""

import psutil
import time
import subprocess
from typing import List, Dict, Any, Optional
import os

from core.base import IDevice, DeviceType, PerformanceMode, PluginMetadata
from utils import get_logger

logger = get_logger('raspberry_pi')


class RaspberryPi(IDevice):
    """Raspberry Pi device optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cpu_freq_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'
        self.thermal_zone = '/sys/class/thermal/thermal_zone0/temp'
        self.gpu_mem_file = '/sys/class/graphics/fb0/mem'
        self.arm_freq = self.config.get('arm_freq', 600)  # MHz
        self.gpu_mem = self.config.get('gpu_mem', 64)  # MB
        self.overclock_enabled = self.config.get('overclock_enabled', False)
        self.thermal_threshold = self.config.get('thermal_threshold', 80)  # Celsius
        self.power_saving_enabled = False
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='raspberry_pi',
            version='1.0.0',
            description='Raspberry Pi device optimization',
            author='Face Verification System Team',
            dependencies=['psutil'],
            device_compatibility=[DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the Raspberry Pi device"""
        try:
            self.config.update(config)
            
            # Check if we're actually running on a Raspberry Pi
            if not self._is_raspberry_pi():
                logger.error("Not running on a Raspberry Pi")
                return False
            
            # Apply initial configuration
            self._apply_configuration()
            
            logger.info("Raspberry Pi device initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Raspberry Pi device: {e}")
            return False
    
    def get_device_type(self) -> DeviceType:
        """Get the device type"""
        return DeviceType.RASPBERRY_PI
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        """Get supported performance modes for Raspberry Pi"""
        return [
            PerformanceMode.STANDARD,
            PerformanceMode.HIGH_SPEED,
            PerformanceMode.ULTRA_HIGH
        ]
    
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        """Optimize Raspberry Pi for specific performance mode"""
        try:
            logger.info(f"Optimizing Raspberry Pi for {mode.value} mode")
            
            if mode == PerformanceMode.STANDARD:
                return self._set_standard_mode()
            elif mode == PerformanceMode.HIGH_SPEED:
                return self._set_high_speed_mode()
            elif mode == PerformanceMode.ULTRA_HIGH:
                return self._set_ultra_high_mode()
            else:
                logger.error(f"Unsupported performance mode: {mode}")
                return False
                
        except Exception as e:
            logger.error(f"Error optimizing for {mode.value}: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get Raspberry Pi system information"""
        try:
            info = {
                'device_type': 'Raspberry Pi',
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'temperature': self._get_temperature(),
                'cpu_frequency': self._get_cpu_frequency(),
                'gpu_memory': self._get_gpu_memory(),
                'uptime': time.time() - psutil.boot_time(),
                'power_saving': self.power_saving_enabled
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def set_power_saving(self, enabled: bool) -> bool:
        """Enable or disable power saving mode"""
        try:
            self.power_saving_enabled = enabled
            
            if enabled:
                # Reduce CPU frequency
                self._set_cpu_frequency(400)  # MHz
                logger.info("Power saving mode enabled")
            else:
                # Restore default frequency
                self._set_cpu_frequency(self.arm_freq)
                logger.info("Power saving mode disabled")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting power saving: {e}")
            return False
    
    def _is_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi"""
        try:
            # Check for Raspberry Pi specific files
            return os.path.exists(self.thermal_zone) or os.path.exists(self.cpu_freq_file)
        except Exception:
            return False
    
    def _apply_configuration(self):
        """Apply configuration settings"""
        try:
            # Set ARM frequency
            self._set_cpu_frequency(self.arm_freq)
            
            # Set GPU memory
            self._set_gpu_memory(self.gpu_mem)
            
            logger.info("Applied configuration settings")
            
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
    
    def _set_standard_mode(self) -> bool:
        """Set standard performance mode"""
        try:
            # Moderate CPU frequency
            self._set_cpu_frequency(800)
            
            # Balanced GPU memory
            self._set_gpu_memory(128)
            
            # Disable power saving
            self.power_saving_enabled = False
            
            logger.info("Standard mode configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting standard mode: {e}")
            return False
    
    def _set_high_speed_mode(self) -> bool:
        """Set high speed performance mode"""
        try:
            # Higher CPU frequency
            self._set_cpu_frequency(1000)
            
            # Less GPU memory for more system memory
            self._set_gpu_memory(64)
            
            # Disable power saving
            self.power_saving_enabled = False
            
            logger.info("High speed mode configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting high speed mode: {e}")
            return False
    
    def _set_ultra_high_mode(self) -> bool:
        """Set ultra high performance mode"""
        try:
            # Maximum CPU frequency (overclock)
            self._set_cpu_frequency(1200)
            
            # Minimum GPU memory
            self._set_gpu_memory(32)
            
            # Disable power saving
            self.power_saving_enabled = False
            
            logger.info("Ultra high mode configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting ultra high mode: {e}")
            return False
    
    def _set_cpu_frequency(self, freq_mhz: int) -> bool:
        """Set CPU frequency"""
        try:
            freq_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'
            if os.path.exists(freq_file):
                with open(freq_file, 'w') as f:
                    f.write(str(freq_mhz * 1000))  # Convert to kHz
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting CPU frequency: {e}")
            return False
    
    def _get_cpu_frequency(self) -> Optional[int]:
        """Get current CPU frequency"""
        try:
            if os.path.exists(self.cpu_freq_file):
                with open(self.cpu_freq_file, 'r') as f:
                    freq_khz = int(f.read().strip())
                    return freq_khz // 1000  # Convert to MHz
            return None
        except Exception as e:
            logger.error(f"Error getting CPU frequency: {e}")
            return None
    
    def _set_gpu_memory(self, mem_mb: int) -> bool:
        """Set GPU memory"""
        try:
            # This requires root privileges
            cmd = f"sudo raspi-config nonint memory_split {mem_mb}"
            subprocess.run(cmd, shell=True, check=True)
            return True
        except Exception as e:
            logger.error(f"Error setting GPU memory: {e}")
            return False
    
    def _get_gpu_memory(self) -> Optional[int]:
        """Get current GPU memory"""
        try:
            if os.path.exists(self.gpu_mem_file):
                with open(self.gpu_mem_file, 'r') as f:
                    return int(f.read().strip())
            return None
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return None
    
    def _get_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            if os.path.exists(self.thermal_zone):
                with open(self.thermal_zone, 'r') as f:
                    temp_millidegree = int(f.read().strip())
                    return temp_millidegree / 1000.0  # Convert to Celsius
            return None
        except Exception as e:
            logger.error(f"Error getting temperature: {e}")
            return None
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['arm_freq', 'gpu_mem', 'overclock_enabled', 'thermal_threshold']