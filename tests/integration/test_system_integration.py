"""Test system integration functionality"""

import unittest
import os
import sys
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from system import FaceVerificationSystem
from enhanced_detection.detectors.enhanced_detector import EnhancedDetector
from enhanced_recognition.recognizers.enhanced_recognizer import EnhancedRecognizer
from enhanced_liveness.detectors.enhanced_liveness_detector import EnhancedLivenessDetector
from enhanced_notifications.notifiers.enhanced_notifier import EnhancedNotifier
from enhanced_device.optimizers.enhanced_device_optimizer import EnhancedDeviceOptimizer

class TestSystemIntegration(unittest.TestCase):
    """Test cases for System Integration"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'database': {
                'type': 'sqlite',
                'path': os.path.join(self.temp_dir, 'test.db')
            },
            'logging': {
                'level': 'INFO',
                'file': os.path.join(self.temp_dir, 'test.log'),
                'max_size': '10MB',
                'backup_count': 5
            },
            'security': {
                'encryption_enabled': True,
                'session_timeout': 3600,
                'max_login_attempts': 5
            },
            'features': {
                'face_detection': True,
                'face_recognition': True,
                'liveness_detection': True,
                'notification_system': True,
                'device_optimization': True,
                'enterprise_features': True
            },
            'enterprise': {
                'analytics_enabled': True,
                'business_intelligence': True,
                'scalability_layer': True
            }
        }
        
        # Create test image files
        self.test_images = []
        for i in range(3):
            image_path = os.path.join(self.temp_dir, f'test_image_{i}.jpg')
            # Create a simple mock image file
            with open(image_path, 'wb') as f:
                f.write(b'fake_image_data')
            self.test_images.append(image_path)
        
        # Create test face data
        self.test_face_data = {
            'person_id': 'test_person_001',
            'name': 'Test Person',
            'faces': [
                {
                    'image_id': 'img_001',
                    'encoding': [0.1, 0.2, 0.3, 0.4],  # Mock encoding
                    'timestamp': datetime.now(),
                    'location': 'entrance'
                }
            ],
            'metadata': {
                'department': 'IT',
                'access_level': 'employee',
                'last_seen': datetime.now()
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_system_initialization(self):
        """Test system initialization"""
        # Save test config
        config_file = os.path.join(self.temp_dir, 'test_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # Initialize system
        system = FaceVerificationSystem(config_file=config_file)
        
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.config)
        self.assertIsNotNone(system.detector)
        self.assertIsNotNone(system.recognizer)
        self.assertIsNotNone(system.liveness_detector)
        self.assertIsNotNone(system.notifier)
        self.assertIsNotNone(system.device_optimizer)
        self.assertIsNotNone(system.database)
        self.assertIsNotNone(system.security_manager)
        
        # Check features enabled
        self.assertTrue(system.features['face_detection'])
        self.assertTrue(system.features['face_recognition'])
        self.assertTrue(system.features['liveness_detection'])
        self.assertTrue(system.features['notification_system'])
        self.assertTrue(system.features['device_optimization'])
        self.assertTrue(system.features['enterprise_features'])
    
    def test_system_initialization_with_enhanced_plugins(self):
        """Test system initialization with enhanced plugins"""
        # Save test config with enhanced plugins
        config = self.test_config.copy()
        config['plugins'] = {
            'detection': {
                'module': 'enhanced_detection.detectors.enhanced_detector.EnhancedDetector',
                'config': {
                    'confidence_threshold': 0.85,
                    'enable_gpu': True,
                    'max_workers': 4
                }
            },
            'recognition': {
                'module': 'enhanced_recognition.recognizers.enhanced_recognizer.EnhancedRecognizer',
                'config': {
                    'model_path': 'models/vgg_face.h5',
                    'batch_size': 32,
                    'enable_gpu': True
                }
            },
            'liveness': {
                'module': 'enhanced_liveness.detectors.enhanced_liveness_detector.EnhancedLivenessDetector',
                'config': {
                    'threshold': 0.9,
                    'enable_motion_analysis': True
                }
            }
        }
        
        config_file = os.path.join(self.temp_dir, 'test_config_enhanced.json')
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Initialize system with enhanced plugins
        system = FaceVerificationSystem(config_file=config_file)
        
        self.assertIsNotNone(system)
        self.assertIsInstance(system.detector, EnhancedDetector)
        self.assertIsInstance(system.recognizer, EnhancedRecognizer)
        self.assertIsInstance(system.liveness_detector, EnhancedLivenessDetector)
        
        # Check plugin configurations
        self.assertEqual(system.detector.config['confidence_threshold'], 0.85)
        self.assertEqual(system.recognizer.config['batch_size'], 32)
        self.assertEqual(system.liveness_detector.config['threshold'], 0.9)
    
    def test_system_face_detection_pipeline(self):
        """Test complete face detection pipeline"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock detector
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            # Mock database
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Run face detection
                result = system.detect_faces(self.test_images[0])
                
                self.assertIsNotNone(result)
                self.assertIn('success', result)
                self.assertIn('detections', result)
                self.assertIn('processing_time', result)
                self.assertIn('confidence_scores', result)
                
                # Check result values
                self.assertTrue(result['success'])
                self.assertIsInstance(result['detections'], list)
                self.assertEqual(len(result['detections']), 1)
                self.assertIsInstance(result['processing_time'], float)
                self.assertIsInstance(result['confidence_scores'], list)
                
                # Verify detector and database were called
                mock_detect.assert_called_once_with(self.test_images[0])
                mock_save.assert_called()
    
    def test_system_face_recognition_pipeline(self):
        """Test complete face recognition pipeline"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock face detection
        mock_detection = {
            'face_id': 'face_001',
            'bbox': [10, 20, 100, 120],
            'confidence': 0.95,
            'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
            'timestamp': datetime.now()
        }
        
        # Mock recognizer
        with patch.object(system.recognizer, 'recognize_face') as mock_recognize:
            mock_recognize.return_value = {
                'person_id': 'test_person_001',
                'confidence': 0.92,
                'match': True,
                'name': 'Test Person'
            }
            
            # Mock database
            with patch.object(system.database, 'get_face_data') as mock_get:
                mock_get.return_value = self.test_face_data
                
                # Run face recognition
                result = system.recognize_face(mock_detection)
                
                self.assertIsNotNone(result)
                self.assertIn('success', result)
                self.assertIn('person_id', result)
                self.assertIn('confidence', result)
                self.assertIn('match', result)
                self.assertIn('name', result)
                self.assertIn('processing_time', result)
                
                # Check result values
                self.assertTrue(result['success'])
                self.assertEqual(result['person_id'], 'test_person_001')
                self.assertEqual(result['confidence'], 0.92)
                self.assertTrue(result['match'])
                self.assertEqual(result['name'], 'Test Person')
                self.assertIsInstance(result['processing_time'], float)
                
                # Verify recognizer and database were called
                mock_recognize.assert_called_once_with(mock_detection)
                mock_get.assert_called_once_with('test_person_001')
    
    def test_system_liveness_detection_pipeline(self):
        """Test complete liveness detection pipeline"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock face detection
        mock_detection = {
            'face_id': 'face_001',
            'bbox': [10, 20, 100, 120],
            'confidence': 0.95,
            'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
            'timestamp': datetime.now()
        }
        
        # Mock liveness detector
        with patch.object(system.liveness_detector, 'detect_liveness') as mock_liveness:
            mock_liveness.return_value = {
                'is_live': True,
                'confidence': 0.88,
                'liveness_score': 0.92,
                'method_used': 'motion_analysis',
                'timestamp': datetime.now()
            }
            
            # Mock database
            with patch.object(system.database, 'save_liveness_result') as mock_save:
                mock_save.return_value = True
                
                # Run liveness detection
                result = system.detect_liveness(mock_detection)
                
                self.assertIsNotNone(result)
                self.assertIn('success', result)
                self.assertIn('is_live', result)
                self.assertIn('confidence', result)
                self.assertIn('liveness_score', result)
                self.assertIn('method_used', result)
                self.assertIn('processing_time', result)
                
                # Check result values
                self.assertTrue(result['success'])
                self.assertTrue(result['is_live'])
                self.assertEqual(result['confidence'], 0.88)
                self.assertEqual(result['liveness_score'], 0.92)
                self.assertEqual(result['method_used'], 'motion_analysis')
                self.assertIsInstance(result['processing_time'], float)
                
                # Verify liveness detector and database were called
                mock_liveness.assert_called_once_with(mock_detection)
                mock_save.assert_called()
    
    def test_system_complete_verification_pipeline(self):
        """Test complete face verification pipeline"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock face detection
        mock_detection = {
            'face_id': 'face_001',
            'bbox': [10, 20, 100, 120],
            'confidence': 0.95,
            'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
            'timestamp': datetime.now()
        }
        
        # Mock face recognition
        mock_recognition = {
            'person_id': 'test_person_001',
            'confidence': 0.92,
            'match': True,
            'name': 'Test Person'
        }
        
        # Mock liveness detection
        mock_liveness = {
            'is_live': True,
            'confidence': 0.88,
            'liveness_score': 0.92,
            'method_used': 'motion_analysis',
            'timestamp': datetime.now()
        }
        
        # Mock database
        with patch.object(system.database, 'get_face_data') as mock_get:
            mock_get.return_value = self.test_face_data
            
            with patch.object(system.database, 'save_verification') as mock_save:
                mock_save.return_value = True
                
                with patch.object(system.notifier, 'send_verification_notification') as mock_notify:
                    mock_notify.return_value = True
                    
                    # Run complete verification
                    result = system.verify_face(self.test_images[0])
                    
                    self.assertIsNotNone(result)
                    self.assertIn('success', result)
                    self.assertIn('person_id', result)
                    self.assertIn('name', result)
                    self.assertIn('verification_result', result)
                    self.assertIn('confidence', result)
                    self.assertIn('processing_time', result)
                    self.assertIn('notifications_sent', result)
                    
                    # Check result values
                    self.assertTrue(result['success'])
                    self.assertEqual(result['person_id'], 'test_person_001')
                    self.assertEqual(result['name'], 'Test Person')
                    self.assertEqual(result['verification_result'], 'verified')
                    self.assertEqual(result['confidence'], 0.92)
                    self.assertTrue(result['notifications_sent'])
                    self.assertIsInstance(result['processing_time'], float)
                    
                    # Verify all components were called
                    mock_get.assert_called_once_with('test_person_001')
                    mock_save.assert_called_once()
                    mock_notify.assert_called_once()
    
    def test_system_device_optimization(self):
        """Test device optimization"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock device optimizer
        with patch.object(system.device_optimizer, 'optimize_device') as mock_optimize:
            mock_optimize.return_value = {
                'success': True,
                'optimizations_applied': ['gpu_acceleration', 'memory_management'],
                'performance_improvement': 0.25,
                'resource_usage': {'cpu': 0.45, 'memory': 0.6}
            }
            
            # Run device optimization
            result = system.optimize_device_performance()
            
            self.assertIsNotNone(result)
            self.assertIn('success', result)
            self.assertIn('optimizations_applied', result)
            self.assertIn('performance_improvement', result)
            self.assertIn('resource_usage', result)
            self.assertIn('optimization_time', result)
            
            # Check result values
            self.assertTrue(result['success'])
            self.assertIsInstance(result['optimizations_applied'], list)
            self.assertEqual(len(result['optimizations_applied']), 2)
            self.assertEqual(result['performance_improvement'], 0.25)
            self.assertIsInstance(result['resource_usage'], dict)
            self.assertIsInstance(result['optimization_time'], float)
            
            # Verify device optimizer was called
            mock_optimize.assert_called_once()
    
    def test_system_load_balancing(self):
        """Test system load balancing"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock device instances
        mock_devices = [
            {'id': 'device_001', 'status': 'active', 'load': 0.3},
            {'id': 'device_002', 'status': 'active', 'load': 0.7},
            {'id': 'device_003', 'status': 'maintenance', 'load': 0.0}
        ]
        
        with patch.object(system.device_optimizer, 'get_available_devices') as mock_get_devices:
            mock_get_devices.return_value = mock_devices
            
            with patch.object(system.device_optimizer, 'select_optimal_device') as mock_select:
                mock_select.return_value = 'device_001'
                
                # Test device selection
                selected_device = system.select_device_for_processing()
                
                self.assertIsNotNone(selected_device)
                self.assertEqual(selected_device, 'device_001')
                
                # Verify device selection was called
                mock_get_devices.assert_called_once()
                mock_select.assert_called_once_with(mock_devices)
    
    def test_system_error_handling(self):
        """Test system error handling"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Test database connection error
        with patch.object(system.database, 'save_detection') as mock_save:
            mock_save.side_effect = Exception("Database connection failed")
            
            # Run face detection with error
            result = system.detect_faces(self.test_images[0])
            
            self.assertIsNotNone(result)
            self.assertIn('success', result)
            self.assertIn('error', result)
            self.assertFalse(result['success'])
            self.assertIn('Database connection failed', result['error'])
        
        # Test detector error
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.side_effect = Exception("Detection failed")
            
            # Mock database to handle saving
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Run face detection with detector error
                result = system.detect_faces(self.test_images[0])
                
                self.assertIsNotNone(result)
                self.assertIn('success', result)
                self.assertIn('error', result)
                self.assertFalse(result['success'])
                self.assertIn('Detection failed', result['error'])
    
    def test_system_concurrent_processing(self):
        """Test concurrent processing"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock detector
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            # Mock database
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Test concurrent processing
                results = []
                threads = []
                
                def process_image(image_path):
                    result = system.detect_faces(image_path)
                    results.append(result)
                
                # Start multiple threads
                for image_path in self.test_images:
                    thread = threading.Thread(target=process_image, args=(image_path,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                # Check results
                self.assertEqual(len(results), len(self.test_images))
                for result in results:
                    self.assertTrue(result['success'])
                    self.assertIn('detections', result)
                    self.assertEqual(len(result['detections']), 1)
                
                # Verify detector was called for each image
                self.assertEqual(mock_detect.call_count, len(self.test_images))
    
    def test_system_performance_monitoring(self):
        """Test system performance monitoring"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Run some operations to generate performance data
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Run multiple detections
                for _ in range(5):
                    system.detect_faces(self.test_images[0])
        
        # Get performance metrics
        metrics = system.get_performance_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertIn('total_detections', metrics)
        self.assertIn('average_processing_time', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('system_load', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('cpu_usage', metrics)
        
        # Check metrics values
        self.assertEqual(metrics['total_detections'], 5)
        self.assertIsInstance(metrics['average_processing_time'], float)
        self.assertEqual(metrics['success_rate'], 1.0)
        self.assertEqual(metrics['error_rate'], 0.0)
        self.assertIsInstance(metrics['system_load'], float)
        self.assertIsInstance(metrics['memory_usage'], float)
        self.assertIsInstance(metrics['cpu_usage'], float)
    
    def test_system_backup_recovery(self):
        """Test system backup and recovery"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock database backup
        with patch.object(system.database, 'backup') as mock_backup:
            mock_backup.return_value = True
            
            # Create backup
            backup_file = os.path.join(self.temp_dir, 'system_backup.db')
            success = system.create_system_backup(backup_file)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(backup_file))
            
            # Verify backup was called
            mock_backup.assert_called_once_with(backup_file)
        
        # Mock database recovery
        with patch.object(system.database, 'restore') as mock_restore:
            mock_restore.return_value = True
            
            # Restore from backup
            success = system.restore_from_backup(backup_file)
            
            self.assertTrue(success)
            
            # Verify restore was called
            mock_restore.assert_called_once_with(backup_file)
    
    def test_system_configuration_update(self):
        """Test system configuration update"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Update configuration
        new_config = {
            'detection_threshold': 0.88,
            'recognition_confidence': 0.92,
            'liveness_threshold': 0.90,
            'enable_notifications': True
        }
        
        success = system.update_configuration(new_config)
        
        self.assertTrue(success)
        
        # Check configuration was updated
        self.assertEqual(system.config['detection_threshold'], 0.88)
        self.assertEqual(system.config['recognition_confidence'], 0.92)
        self.assertEqual(system.config['liveness_threshold'], 0.90)
        self.assertTrue(system.config['enable_notifications'])
    
    def test_system_feature_toggle(self):
        """Test system feature toggle"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Test enabling feature
        success = system.enable_feature('face_detection')
        self.assertTrue(success)
        self.assertTrue(system.features['face_detection'])
        
        # Test disabling feature
        success = system.disable_feature('face_detection')
        self.assertTrue(success)
        self.assertFalse(system.features['face_detection'])
        
        # Test enabling all features
        success = system.enable_all_features()
        self.assertTrue(success)
        for feature in system.features.values():
            self.assertTrue(feature)
        
        # Test disabling all features
        success = system.disable_all_features()
        self.assertTrue(success)
        for feature in system.features.values():
            self.assertFalse(feature)
    
    def test_system_status_report(self):
        """Test system status report"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Run some operations to generate data
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Run multiple detections
                for _ in range(3):
                    system.detect_faces(self.test_images[0])
        
        # Get status report
        status = system.get_system_status()
        
        self.assertIsNotNone(status)
        self.assertIn('system_overview', status)
        self.assertIn('component_status', status)
        self.assertIn('performance_metrics', status)
        self.assertIn('resource_usage', status)
        self.assertIn('features_enabled', status)
        self.assertIn('last_updated', status)
        
        # Check status values
        self.assertIsInstance(status['system_overview'], str)
        self.assertIsInstance(status['component_status'], dict)
        self.assertIsInstance(status['performance_metrics'], dict)
        self.assertIsInstance(status['resource_usage'], dict)
        self.assertIsInstance(status['features_enabled'], dict)
        self.assertIsInstance(status['last_updated'], datetime)
        
        # Check component status
        self.assertIn('detector', status['component_status'])
        self.assertIn('recognizer', status['component_status'])
        self.assertIn('liveness_detector', status['component_status'])
        self.assertIn('notifier', status['component_status'])
        self.assertIn('device_optimizer', status['component_status'])
        
        # Check feature status
        for feature, enabled in system.features.items():
            self.assertEqual(status['features_enabled'][feature], enabled)
    
    def test_system_export_data(self):
        """Test system data export"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Run some operations to generate data
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Run multiple detections
                for _ in range(3):
                    system.detect_faces(self.test_images[0])
        
        # Export data
        export_file = os.path.join(self.temp_dir, 'system_export.json')
        success = system.export_system_data(export_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_file))
        
        # Check exported data
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
            self.assertIn('detection_history', exported_data)
            self.assertIn('system_metrics', exported_data)
            self.assertIn('configuration', exported_data)
            self.assertIn('export_timestamp', exported_data)
            self.assertIn('export_version', exported_data)
    
    def test_system_health_check(self):
        """Test system health check"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Run health check
        health = system.perform_health_check()
        
        self.assertIsNotNone(health)
        self.assertIn('overall_status', health)
        self.assertIn('component_health', health)
        self.assertIn('performance_health', health)
        self.assertIn('resource_health', health)
        self.assertIn('recommendations', health)
        self.assertIn('check_timestamp', health)
        
        # Check health values
        self.assertIsInstance(health['overall_status'], str)
        self.assertIn(health['overall_status'], ['healthy', 'degraded', 'unhealthy'])
        self.assertIsInstance(health['component_health'], dict)
        self.assertIsInstance(health['performance_health'], dict)
        self.assertIsInstance(health['resource_health'], dict)
        self.assertIsInstance(health['recommendations'], list)
        self.assertIsInstance(health['check_timestamp'], datetime)
        
        # Check component health
        self.assertIn('detector', health['component_health'])
        self.assertIn('recognizer', health['component_health'])
        self.assertIn('liveness_detector', health['component_health'])
        self.assertIn('database', health['component_health'])
        
        # Check that all components are healthy by default
        for component in health['component_health'].values():
            self.assertEqual(component, 'healthy')

if __name__ == '__main__':
    unittest.main()