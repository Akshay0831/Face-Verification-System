"""Android device optimization plugin"""

import subprocess
import psutil
import time
import os
from typing import List, Dict, Any, Optional

from core.base import IDevice, DeviceType, PerformanceMode, PluginMetadata
from utils import get_logger

logger = get_logger('android_plugin')


class AndroidPlugin(IDevice):
    """Android device optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.adb_path = self.config.get('adb_path', 'adb')
        self.device_serial = self.config.get('device_serial', '')
        self.cpu_cores = self.config.get('cpu_cores', 4)
        self.memory_limit = self.config.get('memory_limit', 512)  # MB
        self.battery_threshold = self.config.get('battery_threshold', 20)  # Percent
        self.power_saving_enabled = False
        self.hardware_acceleration = True
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='android_plugin',
            version='1.0.0',
            description='Android device optimization',
            author='Face Verification System Team',
            dependencies=['psutil'],
            device_compatibility=[DeviceType.ANDROID]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the Android device"""
        try:
            self.config.update(config)
            
            # Check if ADB is available and device connected
            if not self._check_adb_connection():
                logger.error("Android device not connected or ADB not available")
                return False
            
            # Get device information
            self._get_device_info()
            
            logger.info("Android device initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Android device: {e}")
            return False
    
    def get_device_type(self) -> DeviceType:
        """Get the device type"""
        return DeviceType.ANDROID
    
    def get_supported_performance_modes(self) -> List[PerformanceMode]:
        """Get supported performance modes for Android"""
        return [
            PerformanceMode.STANDARD,
            PerformanceMode.HIGH_SPEED,
            PerformanceMode.ULTRA_HIGH
        ]
    
    def optimize_for_performance(self, mode: PerformanceMode) -> bool:
        """Optimize Android device for specific performance mode"""
        try:
            logger.info(f"Optimizing Android device for {mode.value} mode")
            
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
        """Get Android system information"""
        try:
            info = {
                'device_type': 'Android',
                'serial': self.device_serial,
                'cpu_usage': self._get_android_cpu_usage(),
                'memory_usage': self._get_android_memory_usage(),
                'battery_level': self._get_battery_level(),
                'temperature': self._get_temperature(),
                'cpu_cores': self.cpu_cores,
                'power_saving': self.power_saving_enabled,
                'hardware_acceleration': self.hardware_acceleration
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def set_power_saving(self, enabled: bool) -> bool:
        """Enable or disable Android power saving"""
        try:
            if enabled:
                # Enable power saving mode
                self._execute_adb_command('settings put global power_saver_enabled 1')
                self._execute_adb_command('settings put global low_power 1')
                self.hardware_acceleration = False
                logger.info("Android power saving enabled")
            else:
                # Disable power saving mode
                self._execute_adb_command('settings put global power_saver_enabled 0')
                self._execute_adb_command('settings put global low_power 0')
                self.hardware_acceleration = True
                logger.info("Android power saving disabled")
            
            self.power_saving_enabled = enabled
            return True
            
        except Exception as e:
            logger.error(f"Error setting power saving: {e}")
            return False
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get Android system resources"""
        try:
            resources = {
                'cpu_cores': self.cpu_cores,
                'memory_limit': self.memory_limit,
                'battery_threshold': self.battery_threshold,
                'power_saving_enabled': self.power_saving_enabled,
                'hardware_acceleration': self.hardware_acceleration
            }
            return resources
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
            return True
            
        except Exception as e:
            logger.error(f"Error setting Android power saving: {e}")
            return False
    
    def start_app(self, package_name: str, activity: str = None) -> bool:
        """Start Android application"""
        try:
            cmd = f'{self.adb_path} -s {self.device_serial} shell am start -n {package_name}'
            if activity:
                cmd += f'/{activity}'
            
            result = self._execute_adb_command(cmd)
            if result:
                logger.info(f"Started app: {package_name}")
                return True
            else:
                logger.error(f"Failed to start app: {package_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting app {package_name}: {e}")
            return False
    
    def stop_app(self, package_name: str) -> bool:
        """Stop Android application"""
        try:
            cmd = f'{self.adb_path} -s {self.device_serial} shell am force-stop {package_name}'
            result = self._execute_adb_command(cmd)
            
            if result:
                logger.info(f"Stopped app: {package_name}")
                return True
            else:
                logger.error(f"Failed to stop app: {package_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping app {package_name}: {e}")
            return False
    
    def _check_adb_connection(self) -> bool:
        """Check if ADB device is connected"""
        try:
            result = self._execute_adb_command('devices')
            if result and 'device' in result:
                return True
            return False
        except Exception:
            return False
    
    def _get_device_info(self):
        """Get Android device information"""
        try:
            # Get device serial
            devices_output = self._execute_adb_command('devices')
            if devices_output:
                lines = devices_output.split('\n')
                for line in lines:
                    if 'device' in line and line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            self.device_serial = parts[0]
                            break
            
            # Get CPU cores
            cores_output = self._execute_adb_command('shell cat /proc/cpuinfo | grep -c "^processor"')
            if cores_output:
                self.cpu_cores = int(cores_output.strip())
            
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
    
    def _set_standard_mode(self) -> bool:
        """Set standard performance mode"""
        try:
            # Balanced settings
            self._execute_adb_command('shell setprop debug.sf.hw 1')  # Enable hardware acceleration
            self._set_cpu_governor('ondemand')
            self.power_saving_enabled = False
            
            logger.info("Standard mode configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting standard mode: {e}")
            return False
    
    def _set_high_speed_mode(self) -> bool:
        """Set high speed performance mode"""
        try:
            # High performance settings
            self._execute_adb_command('shell setprop debug.sf.hw 1')  # Enable hardware acceleration
            self._set_cpu_governor('performance')
            self.power_saving_enabled = False
            
            logger.info("High speed mode configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting high speed mode: {e}")
            return False
    
    def _set_ultra_high_mode(self) -> bool:
        """Set ultra high performance mode"""
        try:
            # Maximum performance settings
            self._execute_adb_command('shell setprop debug.sf.hw 1')  # Enable hardware acceleration
            self._set_cpu_governor('performance')
            self.power_saving_enabled = False
            
            # Additional optimizations
            self._execute_adb_command('shell setprop ro.media.dec.jpeg.quality 100')
            self._execute_adb_command('shell setprop debug.gralloc.enable_fb_ubwc 0')
            
            logger.info("Ultra high mode configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting ultra high mode: {e}")
            return False
    
    def _set_cpu_governor(self, governor: str) -> bool:
        """Set CPU governor"""
        try:
            # This requires root access
            cmd = f'shell su -c "echo {governor} > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"'
            self._execute_adb_command(cmd)
            return True
        except Exception as e:
            logger.error(f"Error setting CPU governor: {e}")
            return False
    
    def _execute_adb_command(self, command: str) -> Optional[str]:
        """Execute ADB command"""
        try:
            full_cmd = f'{self.adb_path} -s {self.device_serial} {command}'
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"ADB command failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timed out: {command}")
            return None
        except Exception as e:
            logger.error(f"Error executing ADB command: {e}")
            return None
    
    def _get_android_cpu_usage(self) -> Optional[float]:
        """Get Android CPU usage"""
        try:
            output = self._execute_adb_command('shell dumpsys cpuinfo')
            if output:
                # Parse CPU usage from dumpsys output
                lines = output.split('\n')
                for line in lines:
                    if '% total' in line:
                        cpu_str = line.split('%')[0].strip()
                        return float(cpu_str)
            return None
        except Exception as e:
            logger.error(f"Error getting Android CPU usage: {e}")
            return None
    
    def _get_android_memory_usage(self) -> Optional[float]:
        """Get Android memory usage"""
        try:
            output = self._execute_adb_command('shell dumpsys meminfo')
            if output:
                # Parse memory usage from dumpsys output
                lines = output.split('\n')
                for line in lines:
                    if 'TOTAL:' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            total_memory = int(parts[1])  # KB
                            free_memory = int(parts[2])   # KB
                            usage_percent = ((total_memory - free_memory) / total_memory) * 100
                            return usage_percent
            return None
        except Exception as e:
            logger.error(f"Error getting Android memory usage: {e}")
            return None
    
    def _get_battery_level(self) -> Optional[int]:
        """Get battery level"""
        try:
            output = self._execute_adb_command('shell dumpsys battery')
            if output:
                # Parse battery level from dumpsys output
                lines = output.split('\n')
                for line in lines:
                    if 'level:' in line:
                        level_str = line.split(':')[1].strip()
                        return int(level_str)
            return None
        except Exception as e:
            logger.error(f"Error getting battery level: {e}")
            return None
    
    def _get_temperature(self) -> Optional[float]:
        """Get device temperature"""
        try:
            # Try different temperature sensors
            sensors = [
                'shell cat /sys/class/thermal/thermal_zone0/temp',
                'shell dumpsys thermalservice | grep "Temperature"',
                'shell cat /sys/class/thermal/thermal_zone7/temp'  # Some devices use this
            ]
            
            for sensor_cmd in sensors:
                output = self._execute_adb_command(sensor_cmd)
                if output:
                    try:
                        # Convert temperature to Celsius
                        temp_str = output.strip()
                        if temp_str.isdigit():
                            temp_celsius = int(temp_str) / 1000.0
                            return temp_celsius
                        elif 'Temperature:' in output:
                            temp_part = output.split('Temperature:')[1].split()[0]
                            return float(temp_part)
                    except (ValueError, IndexError):
                        continue
            
            return None
        except Exception as e:
            logger.error(f"Error getting temperature: {e}")
            return None
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['adb_path', 'device_serial', 'cpu_cores', 'memory_limit', 'battery_threshold']