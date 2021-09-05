"""Core face verification system."""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import threading
import time

from .plugin_manager import PluginManager
from .device_manager import DeviceManager
from .base import IDetector, IRecognizer, ILivenessDetector, INotifier, ICamera
from utils import get_logger
from .base import DeviceType

logger = get_logger('core_system')


class FaceVerificationSystem:
    """Main face verification system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.plugin_manager = PluginManager()
        self.device_manager = DeviceManager()
        self.is_running = False
        self.processing_thread = None
        
        # System state
        self.detection_count = 0
        self.recognition_count = 0
        self.liveness_count = 0
        self.notification_count = 0
        
        # Initialize with defaults
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default configuration"""
        defaults = {
            'detection_confidence_threshold': 0.5,
            'recognition_confidence_threshold': 0.7,
            'liveness_confidence_threshold': 0.8,
            'max_fps': 30,
            'frame_skip': 1,
            'enable_logging': True,
            'log_level': 'INFO',
            'save_processed_frames': False,
            'output_directory': 'output',
            'max_detection_size': (640, 480)
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def initialize(self) -> bool:
        """Initialize the system"""
        try:
            logger.info("Initializing face verification system...")
            
            # Create output directory
            output_dir = self.config.get('output_directory', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Load plugins
            plugin_count = self.plugin_manager.load_plugins()
            logger.info(f"Loaded {plugin_count} plugins")
            
            # Initialize devices
            self._initialize_devices()
            
            logger.info("Face verification system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def _initialize_devices(self):
        """Initialize default devices"""
        # Add default camera
        try:
            from .device_manager import CameraDevice
            camera = CameraDevice("default_camera", "Default Camera", 0)
            self.device_manager.add_device(camera)
            logger.info("Added default camera device")
        except Exception as e:
            logger.error(f"Failed to add default camera: {e}")
    
    def start(self) -> bool:
        """Start the system"""
        if self.is_running:
            logger.warning("System is already running")
            return True
        
        try:
            logger.info("Starting face verification system...")
            
            # Connect devices
            for device in self.device_manager.get_devices_by_type(DeviceType.CAMERA):
                if self.device_manager.connect_device(device.device_id):
                    logger.info(f"Connected camera {device.device_id}")
            
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("Face verification system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the system"""
        if not self.is_running:
            logger.debug("System is not running")
            return True
        
        try:
            logger.info("Stopping face verification system...")
            
            self.is_running = False
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            # Disconnect devices
            for device in self.device_manager.get_all_devices():
                self.device_manager.disconnect_device(device.device_id)
            
            logger.info("Face verification system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            return False
    
    def _process_loop(self):
        """Main processing loop"""
        frame_skip = self.config.get('frame_skip', 1)
        frame_counter = 0
        last_time = time.time()
        
        while self.is_running:
            try:
                # Check if system is still running
                if not self.is_running:
                    break
                
                # Get frame from camera
                # Use a generic approach to get camera devices
                devices = []
                for device in self.device_manager.get_all_devices():
                    if hasattr(device, 'camera_index') or 'camera' in device.device_id.lower():
                        devices.append(device)
                if not devices:
                    time.sleep(0.1)
                    continue
                
                camera = devices[0]
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Frame skipping
                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    continue
                
                # Process frame
                self._process_frame(frame)
                
                # FPS limiting
                current_time = time.time()
                elapsed = current_time - last_time
                target_interval = 1.0 / self.config.get('max_fps', 30)
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                last_time = current_time
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame"""
        try:
            # Detect faces
            detectors = self.plugin_manager.get_plugins_by_type('detector')
            if not detectors:
                logger.warning("No face detectors available")
                return
            
            for detector in detectors:
                detections = detector.detect(frame)
                self.detection_count += len(detections)
                
                for detection in detections:
                    # Extract face region
                    x, y, w, h = detection['bbox']
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Recognize face
                    recognizers = self.plugin_manager.get_plugins_by_type('recognizer')
                    if recognizers:
                        for recognizer in recognizers:
                            result = recognizer.recognize(face_img)
                            self.recognition_count += 1
                            
                            # Check liveness
                            liveness_detectors = self.plugin_manager.get_plugins_by_type('liveness')
                            if liveness_detectors:
                                for liveness_detector in liveness_detectors:
                                    liveness_result = liveness_detector.check_liveness(face_img)
                                    self.liveness_count += 1
                                    
                                    # Send notifications if face is recognized
                                    if result.get('identity'):
                                        self._send_notifications(result, liveness_result)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _send_notifications(self, recognition_result: Dict[str, Any], liveness_result: Dict[str, Any]):
        """Send notifications for recognized face"""
        try:
            notifiers = self.plugin_manager.get_plugins_by_type('notifier')
            if not notifiers:
                return
            
            notification_data = {
                'timestamp': datetime.now().isoformat(),
                'identity': recognition_result.get('identity'),
                'confidence': recognition_result.get('confidence', 0.0),
                'is_liveness_passed': liveness_result.get('is_liveness_passed', False),
                'liveness_confidence': liveness_result.get('confidence', 0.0),
                'detection_count': self.detection_count,
                'recognition_count': self.recognition_count,
                'liveness_count': self.liveness_count
            }
            
            for notifier in notifiers:
                notifier.send(notification_data)
                self.notification_count += 1
                
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'is_running': self.is_running,
            'detection_count': self.detection_count,
            'recognition_count': self.recognition_count,
            'liveness_count': self.liveness_count,
            'notification_count': self.notification_count,
            'plugin_count': len(self.plugin_manager.loaded_plugins),
            'device_count': len(self.device_manager.get_all_devices()),
            'device_status': self.device_manager.get_system_status(),
            'config': self.config
        }
    
    def get_plugin_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        status = {}
        for plugin_type, plugins in self.plugin_manager.plugin_types.items():
            status[plugin_type] = []
            for plugin in plugins:
                try:
                    metadata = plugin.get_metadata()
                    status[plugin_type].append({
                        'name': metadata.name,
                        'version': metadata.version,
                        'description': metadata.description
                    })
                except Exception as e:
                    logger.error(f"Error getting metadata for {plugin_type} plugin: {e}")
        
        return status