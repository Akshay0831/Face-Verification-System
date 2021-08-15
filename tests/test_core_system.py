"""Tests for core face verification system components"""

import unittest
import os
import sys
import tempfile
import yaml
import numpy as np
from unittest.mock import Mock, patch
from abc import ABC, abstractmethod

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from system import FaceVerificationSystem
from core import (
    DeviceType, PerformanceMode, DetectionResult, RecognitionResult,
    LivenessResult, PluginMetadata, IDetector, IRecognizer, 
    ILivenessDetector, INotifier
)


class TestFaceVerificationSystem(unittest.TestCase):
    """Test cases for FaceVerificationSystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary configuration file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_data = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0'
            },
            'plugins': {
                'detection': {
                    'plugin': 'mock_detector'
                },
                'recognition': {
                    'plugin': 'mock_recognizer'
                },
                'liveness': {
                    'plugin': 'mock_liveness'
                },
                'notifications': {
                    'plugin': 'mock_notifier'
                }
            },
            'performance': {
                'mode': 'standard'
            }
        }
        
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()
        
        # Create system instance
        self.system = FaceVerificationSystem(self.temp_config.name)
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_initialization(self):
        """Test system initialization"""
        # Mock the plugin manager and device manager
        with patch.object(self.system, 'plugin_manager') as mock_pm, \
             patch.object(self.system, 'device_manager') as mock_dm:
            
            # Mock successful initialization
            mock_pm.load_plugins.return_value = True
            mock_pm.initialize_plugins.return_value = True
            mock_dm.initialize_device.return_value = True
            mock_dm.set_performance_mode.return_value = True
            
            # Test initialization
            result = self.system.initialize()
            self.assertTrue(result)
            self.assertTrue(self.system.initialized)
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        self.assertTrue(os.path.exists(self.temp_config.name))
        result = self.system._load_configuration()
        self.assertTrue(result)
        self.assertIn('system', self.system.config)
        self.assertEqual(self.system.config['system']['name'], 'Test System')
    
    def test_metrics_tracking(self):
        """Test system metrics tracking"""
        # Initial metrics should be empty
        metrics = self.system.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_detections', metrics)
        self.assertEqual(metrics['total_detections'], 0)
        
        # Simulate some processing
        self.system.metrics['total_detections'] = 5
        self.system.metrics['successful_recognitions'] = 3
        
        updated_metrics = self.system.get_metrics()
        self.assertEqual(updated_metrics['total_detections'], 5)
        self.assertEqual(updated_metrics['successful_recognitions'], 3)
    
    def test_system_info(self):
        """Test system information retrieval"""
        info = self.system.get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn('initialized', info)
        self.assertIn('running', info)
        self.assertIn('plugins_loaded', info)
    
    def test_system_shutdown(self):
        """Test system shutdown"""
        self.system.initialized = True
        with patch.object(self.system, 'plugin_manager') as mock_pm:
            mock_plugin = Mock()
            mock_pm.loaded_plugins = {'test': mock_plugin}
            
            self.system.shutdown()
            
            # Check that cleanup was called on plugins
            mock_plugin.cleanup.assert_called_once()
            self.assertFalse(self.system.running)


class TestCoreComponents(unittest.TestCase):
    """Test core system components"""
    
    def test_detection_result(self):
        """Test DetectionResult class"""
        bbox = (10, 20, 100, 150)
        confidence = 0.85
        face_image = Mock()
        
        result = DetectionResult(bbox, confidence, face_image)
        
        self.assertEqual(result.bbox, bbox)
        self.assertEqual(result.confidence, confidence)
        self.assertEqual(result.face_image, face_image)
    
    def test_recognition_result(self):
        """Test RecognitionResult class"""
        user_id = "test_user"
        confidence = 0.92
        embedding = [0.1, 0.2, 0.3]
        
        result = RecognitionResult(user_id, confidence, embedding)
        
        self.assertEqual(result.user_id, user_id)
        self.assertEqual(result.confidence, confidence)
        self.assertEqual(result.embedding, embedding)
    
    def test_liveness_result(self):
        """Test LivenessResult class"""
        is_live = True
        confidence = 0.95
        details = {'blink_count': 2, 'eye_movement': 'normal'}
        
        result = LivenessResult(is_live, confidence, details)
        
        self.assertTrue(result.is_live)
        self.assertEqual(result.confidence, confidence)
        self.assertEqual(result.details, details)
    
    def test_plugin_metadata(self):
        """Test PluginMetadata class"""
        name = "test_plugin"
        version = "1.0.0"
        description = "Test plugin"
        author = "Test Author"
        dependencies = ["opencv-python", "numpy"]
        
        metadata = PluginMetadata(
            name, version, description, author, dependencies
        )
        
        self.assertEqual(metadata.name, name)
        self.assertEqual(metadata.version, version)
        self.assertEqual(metadata.description, description)
        self.assertEqual(metadata.author, author)
        self.assertEqual(metadata.dependencies, dependencies)
    
    def test_device_type_enum(self):
        """Test DeviceType enum"""
        self.assertEqual(DeviceType.RASPBERRY_PI.value, "raspberry_pi")
        self.assertEqual(DeviceType.WINDOWS.value, "windows")
        self.assertEqual(DeviceType.LINUX.value, "linux")
        self.assertEqual(DeviceType.ANDROID.value, "android")
    
    def test_performance_mode_enum(self):
        """Test PerformanceMode enum"""
        self.assertEqual(PerformanceMode.STANDARD.value, "standard")
        self.assertEqual(PerformanceMode.HIGH_SPEED.value, "high_speed")
        self.assertEqual(PerformanceMode.ULTRA_HIGH.value, "ultra_high")


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary config with all necessary sections
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_data = {
            'system': {
                'name': 'Integration Test System',
                'version': '1.0.0'
            },
            'plugins': {
                'detection': {
                    'plugin': 'mock_detector',
                    'config': {
                        'confidence_threshold': 0.7
                    }
                },
                'recognition': {
                    'plugin': 'mock_recognizer',
                    'config': {
                        'similarity_threshold': 0.45
                    }
                },
                'liveness': {
                    'plugin': 'mock_liveness',
                    'config': {
                        'confidence_threshold': 0.8
                    }
                },
                'notifications': {
                    'plugin': 'mock_notifier',
                    'config': {
                        'cooldown_period': 300
                    }
                }
            },
            'performance': {
                'mode': 'standard'
            },
            'security': {
                'enable_liveness_detection': True,
                'minimum_confidence_threshold': 0.6
            }
        }
        
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()
        
        # Create system instance
        self.system = FaceVerificationSystem(self.temp_config.name)
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    

            
            def get_supported_modes(self):
                return ['standard']
            
            def get_required_config(self):
                return []
            def enroll(self, user_id, face_image):
                return True
            
            def get_embedding(self, face_image):
                return np.array([0.1, 0.2, 0.3])
            
            def get_supported_modes(self):
                return ['standard']
            
            def get_required_config(self):
                return []
            
            def initialize(self, config=None):
                return True
    
            def check_liveness(self, face_images):
                return LivenessResult(True, 0.95)
            
            def get_supported_modes(self):
                return ['standard']
            
            def get_required_config(self):
                return []
            
            def initialize(self, config=None):
                return True
    
            def send_notification(self, result):
                return NotificationResult(True, "Mock notification")
            
            def get_supported_modes(self):
                return ['standard']
            
            def get_required_config(self):
                return []
            mock_detector = MockDetector()
            mock_recognizer = MockRecognizer()
            mock_liveness = MockLivenessDetector()
            mock_notifier = MockNotifier()
            
            # Set up plugin manager
            mock_pm.load_plugins.return_value = True
            mock_pm.initialize_plugins.return_value = True
            mock_pm.get_plugin_by_name.side_effect = lambda name: {
                'mock_detector': mock_detector,
                'mock_recognizer': mock_recognizer,
                'mock_liveness': mock_liveness,
                'mock_notifier': mock_notifier
            }.get(name)
            
            mock_dm.initialize_device.return_value = True
            mock_dm.set_performance_mode.return_value = True
            
            # Initialize system
            result = self.system.initialize()
            self.assertTrue(result)
            
            # Create mock image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test frame processing
            result = self.system.process_frame(image)
            
            # Verify that plugins were called
            mock_detector.detect.assert_called_once()
            mock_recognizer.recognize.assert_called()
            mock_liveness.check_liveness.assert_called()
            
            # Check results
            self.assertIsInstance(result, dict)
            self.assertIn('detections', result)
            self.assertIn('recognitions', result)
            self.assertIn('liveness_results', result)
            self.assertIn('intruders', result)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)