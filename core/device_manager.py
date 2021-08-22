"""Device management for face verification system."""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Type
from enum import Enum
from threading import Lock
import logging

from .base import IDevice, DeviceType, ICamera
from utils import get_logger

logger = get_logger('device_manager')


class Device:
    """Base device class"""
    
    def __init__(self, device_id: str, name: str, device_type: DeviceType):
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.is_connected = False
        self.properties = {}
        
    def connect(self) -> bool:
        """Connect to the device"""
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from the device"""
        self.is_connected = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get device status"""
        return {
            'device_id': self.device_id,
            'name': self.name,
            'device_type': self.device_type.value,
            'is_connected': self.is_connected,
            'properties': self.properties
        }


class CameraDevice(Device, ICamera):
    """Camera device implementation"""
    
    def __init__(self, device_id: str, name: str, camera_index: int = 0):
        super().__init__(device_id, name, DeviceType.CAMERA)
        self.camera_index = camera_index
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
    def connect(self) -> bool:
        """Connect to camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                self.is_connected = True
                logger.info(f"Connected to camera {self.device_id}")
                return True
            else:
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to camera {self.device_id}: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from camera"""
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
            self.is_connected = False
            logger.info(f"Disconnected from camera {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from camera {self.device_id}: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        if not self.is_connected or not self.camera:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            logger.error(f"Failed to read frame from camera {self.device_id}")
            return None
    
    def start_capture(self) -> bool:
        """Start frame capture"""
        if self.is_connected:
            logger.info(f"Started capture on camera {self.device_id}")
            return True
        return False
    
    def stop_capture(self) -> bool:
        """Stop frame capture"""
        logger.info(f"Stopped capture on camera {self.device_id}")
        return True
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution"""
        if not self.is_connected or not self.camera:
            return False
        
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.frame_width = width
            self.frame_height = height
            logger.info(f"Set resolution to {width}x{height} for camera {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set resolution for camera {self.device_id}: {e}")
            return False
    
    def set_fps(self, fps: int) -> bool:
        """Set camera frame rate"""
        if not self.is_connected or not self.camera:
            return False
        
        try:
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
            logger.info(f"Set FPS to {fps} for camera {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set FPS for camera {self.device_id}: {e}")
            return False


class DeviceManager:
    """Manages all devices in the system"""
    
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self.device_types: Dict[DeviceType, List[Device]] = {}
        self.lock = Lock()
        
        # Initialize device type dictionaries
        for device_type in DeviceType:
            self.device_types[device_type] = []
    
    def add_device(self, device: Device) -> bool:
        """Add a device to the manager"""
        with self.lock:
            if device.device_id in self.devices:
                logger.warning(f"Device {device.device_id} already exists")
                return False
            
            self.devices[device.device_id] = device
            self.device_types[device.device_type].append(device)
            logger.info(f"Added device {device.device_id} of type {device.device_type}")
            return True
    
    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the manager"""
        with self.lock:
            if device_id not in self.devices:
                logger.warning(f"Device {device_id} not found")
                return False
            
            device = self.devices[device_id]
            self.device_types[device.device_type].remove(device)
            del self.devices[device_id]
            
            # Disconnect device
            device.disconnect()
            
            logger.info(f"Removed device {device_id}")
            return True
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get a device by ID"""
        return self.devices.get(device_id)
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """Get all devices of a specific type"""
        return self.device_types.get(device_type, [])
    
    def get_all_devices(self) -> List[Device]:
        """Get all devices"""
        return list(self.devices.values())
    
    def connect_device(self, device_id: str) -> bool:
        """Connect to a device"""
        device = self.get_device(device_id)
        if device:
            return device.connect()
        return False
    
    def disconnect_device(self, device_id: str) -> bool:
        """Disconnect from a device"""
        device = self.get_device(device_id)
        if device:
            return device.disconnect()
        return False
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device status"""
        device = self.get_device(device_id)
        if device:
            return device.get_status()
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_devices = len(self.devices)
        connected_devices = sum(1 for d in self.devices.values() if d.is_connected)
        
        type_counts = {}
        for device_type, devices in self.device_types.items():
            type_counts[device_type.value] = len(devices)
        
        return {
            'total_devices': total_devices,
            'connected_devices': connected_devices,
            'device_types': type_counts
        }