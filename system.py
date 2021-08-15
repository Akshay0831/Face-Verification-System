"""Main system orchestrates all face verification components."""

import os
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2

from core import (
    PluginManager, DeviceManager, DeviceType, PerformanceMode,
    IDetector, IRecognizer, ILivenessDetector, INotifier,
    DetectionResult, RecognitionResult, LivenessResult, NotificationResult
)
from utils import get_logger

logger = get_logger('system')


class FaceVerificationSystem:
    """Coordinates all face verification system components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize face verification system"""
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = config_path or os.path.join(
            package_dir, 'face_verification_system', 'config', 'system.yaml'
        )
        logger.debug(f"Using config path: {self.config_path}")
        self.config = {}
        
        # Initialize core components
        self.plugin_manager = PluginManager()
        self.device_manager = DeviceManager(self.plugin_manager)
        
        # System state
        self.initialized = False
        self.running = False
        self.detection_plugin = None
        self.recognition_plugin = None
        self.liveness_plugin = None
        self.notifier_plugin = None
        
        # System metrics
        self.metrics = {
            'total_detections': 0,
            'successful_recognitions': 0,
            'intruder_detections': 0,
            'processing_time_avg': 0.0,
            'last_detection_time': None
        }
        
    def initialize(self) -> bool:
        """Initialize the entire system"""
        try:
            logger.info("Initializing Face Verification System...")
            
            # Load configuration
            self._load_configuration()
            
            # Initialize device manager
            if not self.device_manager.initialize_device(self.config.get('device', {})):
                logger.error("Failed to initialize device manager")
                return False
            
            # Load and initialize plugins
            if not self._load_plugins():
                logger.error("Failed to load plugins")
                return False
            
            # Set performance mode
            performance_mode = self.config.get('performance', {}).get('mode', 'standard')
            try:
                mode = PerformanceMode(performance_mode)
                if not self.device_manager.set_performance_mode(mode):
                    logger.warning(f"Failed to set performance mode: {mode}")
            except ValueError:
                logger.warning(f"Invalid performance mode: {performance_mode}")
            
            self.initialized = True
            logger.info("Face Verification System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def _load_configuration(self) -> bool:
        """Load system configuration from YAML file"""
        try:
            import yaml
            
            if not os.path.exists(self.config_path):
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded configuration from: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _load_plugins(self) -> bool:
        """Load and initialize all required plugins"""
        try:
            # Load all plugins
            if not self.plugin_manager.load_plugins():
                logger.error("Failed to load plugins")
                return False
            
            # Initialize plugins with configuration
            if not self.plugin_manager.initialize_plugins(self.config.get('plugins', {})):
                logger.error("Failed to initialize plugins")
                return False
            
            # Get specific plugin instances
            self._get_plugin_instances()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            return False
    
    def _get_plugin_instances(self):
        """Get instances of specific plugins needed for face verification"""
        plugins_config = self.config.get('plugins', {})
        
        # Get detection plugin
        if 'basic_detector' in plugins_config:
            detection_plugin = self.plugin_manager.get_plugin_by_name('basic_detector')
            if isinstance(detection_plugin, IDetector):
                self.detection_plugin = detection_plugin
                logger.info("Loaded detection plugin: basic_detector")
            else:
                logger.warning("basic_detector plugin is not a valid IDetector")
        
        # Get recognition plugin
        if 'vgg_face' in plugins_config:
            recognition_plugin = self.plugin_manager.get_plugin_by_name('vgg_face')
            if isinstance(recognition_plugin, IRecognizer):
                self.recognition_plugin = recognition_plugin
                logger.info("Loaded recognition plugin: vgg_face")
            else:
                logger.warning("vgg_face plugin is not a valid IRecognizer")
        
        # Get liveness plugin
        if 'blink_detector' in plugins_config:
            liveness_plugin = self.plugin_manager.get_plugin_by_name('blink_detector')
            if isinstance(liveness_plugin, ILivenessDetector):
                self.liveness_plugin = liveness_plugin
                logger.info("Loaded liveness plugin: blink_detector")
            else:
                logger.warning("blink_detector plugin is not a valid ILivenessDetector")
        
        # Get notifier plugin
        if 'email_notifier' in plugins_config:
            notifier_plugin = self.plugin_manager.get_plugin_by_name('email_notifier')
            if isinstance(notifier_plugin, INotifier):
                self.notifier_plugin = notifier_plugin
                logger.info("Loaded notifier plugin: email_notifier")
            else:
                logger.warning("email_notifier plugin is not a valid INotifier")
    
    def detect_faces(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces in an image"""
        if not self.initialized or not self.detection_plugin:
            logger.error("System not initialized or no detection plugin available")
            return []
        
        try:
            start_time = time.time()
            detections = self.detection_plugin.detect(image)
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics['total_detections'] += len(detections)
            self.metrics['last_detection_time'] = time.time()
            self.metrics['processing_time_avg'] = (
                self.metrics['processing_time_avg'] * 0.9 + processing_time * 0.1
            )
            
            logger.debug(f"Detected {len(detections)} faces in {processing_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def recognize_faces(self, detections: List[DetectionResult], image: np.ndarray) -> List[RecognitionResult]:
        """Recognize detected faces"""
        if not self.initialized or not self.recognition_plugin:
            logger.error("System not initialized or no recognition plugin available")
            return []
        
        results = []
        try:
            for detection in detections:
                if detection.face_image is None:
                    # Extract face region from image
                    x, y, w, h = detection.bbox
                    face_image = image[y:y+h, x:x+w]
                    detection.face_image = face_image
                
                recognition_result = self.recognition_plugin.recognize(detection.face_image)
                results.append(recognition_result)
                
                if recognition_result.user_id:
                    self.metrics['successful_recognitions'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return []
    
    def verify_liveness(self, detections: List[DetectionResult], image: np.ndarray) -> List[LivenessResult]:
        """Verify liveness of detected faces"""
        if not self.initialized or not self.liveness_plugin:
            logger.warning("Liveness detection not available")
            # Return dummy results indicating liveness is not verified
            return [LivenessResult(is_live=True, confidence=1.0) for _ in detections]
        
        face_images = [detection.face_image for detection in detections if detection.face_image is not None]
        if not face_images:
            # Extract face images if not already available
            face_images = []
            for detection in detections:
                x, y, w, h = detection.bbox
                face_images.append(image[y:y+h, x:x+w])
        
        try:
            return [self.liveness_plugin.check_liveness(face_images)]
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            return [LivenessResult(is_live=True, confidence=0.5)]  # Default to live with medium confidence
    
    def process_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the complete pipeline"""
        if not self.initialized:
            logger.error("System not initialized")
            return {}
        
        start_time = time.time()
        
        # Step 1: Face detection
        detections = self.detect_faces(image)
        
        if not detections:
            return {
                'detections': [],
                'recognitions': [],
                'liveness_results': [],
                'intruders': [],
                'processing_time': time.time() - start_time
            }
        
        # Step 2: Face recognition
        recognitions = self.recognize_faces(detections, image)
        
        # Step 3: Liveness detection
        liveness_results = self.verify_liveness(detections, image)
        
        # Step 4: Process results and handle intruders
        intruders = []
        security_config = self.config.get('security', {})
        liveness_required = security_config.get('enable_liveness_detection', True)
        
        for i, (detection, recognition, liveness) in enumerate(zip(detections, recognitions, liveness_results)):
            is_known = recognition.user_id is not None
            confidence_threshold = security_config.get('minimum_confidence_threshold', 0.6)
            
            # Check if this is an intruder
            if not is_known and recognition.confidence >= confidence_threshold:
                if not liveness_required or liveness.is_live:
                    intruder = {
                        'bbox': detection.bbox,
                        'confidence': recognition.confidence,
                        'timestamp': time.time(),
                        'face_image': detection.face_image,
                        'user_id': None
                    }
                    intruders.append(intruder)
                    self.metrics['intruder_detections'] += 1
        
        # Step 5: Send notifications for intruders
        if intruders and self.notifier_plugin:
            self._send_intruder_notifications(intruders)
        
        return {
            'detections': detections,
            'recognitions': recognitions,
            'liveness_results': liveness_results,
            'intruders': intruders,
            'processing_time': time.time() - start_time
        }
    
    def _send_intruder_notifications(self, intruders: List[Dict[str, Any]]):
        """Send notifications for detected intruders"""
        try:
            if self.notifier_plugin:
                notification_config = self.config.get('notifications', {})
                include_image = notification_config.get('include_image_in_alert', True)
                
                for intruder in intruders:
                    message = f"Intruder detected with confidence: {intruder['confidence']:.2f}"
                    
                    notification_result = self.notifier_plugin.send_notification(
                        message=message,
                        include_image=include_image and intruder.get('face_image') is not None,
                        intruder_data=intruder
                    )
                    
                    if notification_result.success:
                        logger.info(f"Sent intruder notification successfully")
                    else:
                        logger.error(f"Failed to send intruder notification: {notification_result.message}")
                        
        except Exception as e:
            logger.error(f"Failed to send intruder notifications: {e}")
    
    def enroll_user(self, user_id: str, face_image: np.ndarray) -> bool:
        """Enroll a new user with a face image"""
        if not self.initialized or not self.recognition_plugin:
            logger.error("System not initialized or no recognition plugin available")
            return False
        
        try:
            # Preprocess the image if needed
            processed_image = self._preprocess_image_for_recognition(face_image)
            
            # Enroll the user
            success = self.recognition_plugin.enroll(user_id, processed_image)
            
            if success:
                logger.info(f"Successfully enrolled user: {user_id}")
                return True
            else:
                logger.error(f"Failed to enroll user: {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to enroll user {user_id}: {e}")
            return False
    
    def _preprocess_image_for_recognition(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for recognition using configured processing plugin"""
        # For now, simple preprocessing - can be extended with processing plugins
        if image.shape[0] < 100 or image.shape[1] < 100:
            # Resize small images
            image = cv2.resize(image, (224, 224))
        
        return image
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return self.metrics.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'initialized': self.initialized,
            'running': self.running,
            'device_type': self.device_manager.get_current_device().get_device_type().value,
            'performance_mode': self.device_manager.performance_mode.value,
            'plugins_loaded': self.plugin_manager.get_plugin_info(),
            'system_resources': self.device_manager.get_system_resources()
        }
    
    def start_camera_capture(self, source: int = 0) -> None:
        """Start continuous camera capture"""
        if not self.initialized:
            logger.error("System not initialized")
            return
        
        self.running = True
        logger.info(f"Starting camera capture from source: {source}")
        
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error("Failed to open camera source")
                return
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process the frame
                result = self.process_frame(frame)
                
                # Display results (for debugging)
                self._display_results(frame, result)
                
                # Small delay to control frame rate
                time.sleep(0.1)
                
                # Check for exit condition (ESC key)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Camera capture error: {e}")
        finally:
            self.running = False
    
    def _display_results(self, frame: np.ndarray, result: Dict[str, Any]):
        """Display processing results on frame (for debugging)"""
        # Draw detection boxes
        for detection in result.get('detections', []):
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence
            conf_text = f"{detection.confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw recognition results
        for i, recognition in enumerate(result.get('recognitions', [])):
            if recognition.user_id:
                text = f"User: {recognition.user_id}"
                cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                text = "Unknown"
                cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw processing time
        processing_time = result.get('processing_time', 0)
        time_text = f"Time: {processing_time:.3f}s"
        cv2.putText(frame, time_text, (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Face Verification', frame)
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down Face Verification System...")
        self.running = False
        self.initialized = False
        
        # Cleanup plugins
        for plugin_name, plugin in self.plugin_manager.loaded_plugins.items():
            try:
                plugin.cleanup()
                logger.debug(f"Cleaned up plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to cleanup plugin {plugin_name}: {e}")
        
        logger.info("Face Verification System shutdown complete")
    
    def reload_plugins(self):
        """Reload all plugins"""
        logger.info("Reloading plugins...")
        self.shutdown()
        
        # Clear loaded plugins
        self.plugin_manager.loaded_plugins.clear()
        
        # Reinitialize
        success = self.initialize()
        if success:
            logger.info("Plugins reloaded successfully")
        else:
            logger.error("Failed to reload plugins")
        
        return success