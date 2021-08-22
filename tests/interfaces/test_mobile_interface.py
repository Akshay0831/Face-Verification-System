"""Test mobile interface functionality"""

import unittest
import os
import sys
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from interfaces.mobile_interface import MobileInterface, KivyFaceApp, MobileFaceRecognitionAPI

class TestMobileInterface(unittest.TestCase):
    """Test cases for Mobile Interface"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'mobile': {
                'platform': 'android',
                'orientation': 'portrait',
                'fullscreen': True,
                'show_fps': False,
                'min_fps': 15,
                'max_fps': 30
            },
            'camera': {
                'resolution': (1280, 720),
                'fps': 30,
                'exposure_mode': 'auto',
                'focus_mode': 'continuous'
            },
            'network': {
                'enable_upload': True,
                'max_upload_size': 10,  # MB
                'timeout': 30,
                'retry_attempts': 3
            }
        }
        
        # Create test user data
        self.test_user = {
            'user_id': 'mobile_user',
            'device_id': 'android_device_001',
            'last_sync': time.time()
        }
        
        # Create test face data
        self.test_face_data = {
            'person_id': 'mobile_person',
            'name': 'Mobile User',
            'embeddings': np.random.randn(512).tolist(),
            'metadata': {
                'device_id': 'android_device_001',
                'timestamp': time.time(),
                'location': 'home',
                'confidence_threshold': 0.85
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_mobile_interface_initialization(self):
        """Test MobileInterface initialization"""
        interface = MobileInterface()
        
        self.assertIsNotNone(interface)
        self.assertIsNotNone(interface.config)
        self.assertIsNotNone(interface.app)
        self.assertEqual(interface.platform, 'android')
        self.assertEqual(interface.orientation, 'portrait')
        self.assertTrue(interface.fullscreen)
    
    def test_mobile_interface_custom_config(self):
        """Test MobileInterface with custom configuration"""
        config_file = os.path.join(self.temp_dir, 'mobile_config.json')
        
        # Save test config
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # Initialize with custom config
        interface = MobileInterface(config_file=config_file)
        
        self.assertEqual(interface.platform, 'android')
        self.assertEqual(interface.orientation, 'portrait')
        self.assertTrue(interface.fullscreen)
        self.assertEqual(interface.camera_resolution, (1280, 720))
        self.assertEqual(interface.camera_fps, 30)
    
    def test_mobile_interface_start_stop(self):
        """Test MobileInterface start and stop"""
        interface = MobileInterface()
        
        # Mock kivy app
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_kivy_app.return_value = mock_app_instance
            
            # Start interface
            success = interface.start()
            self.assertTrue(success)
            mock_kivy_app.assert_called_once()
            mock_app_instance.run.assert_called_once()
            
            # Stop interface
            interface.stop()
            mock_app_instance.stop.assert_called_once()
    
    def test_kivy_face_app_initialization(self):
        """Test KivyFaceApp initialization"""
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_kivy_app.return_value = mock_app_instance
            
            app = KivyFaceApp()
            
            self.assertIsNotNone(app)
            self.assertIsNotNone(app.interface)
            self.assertIsNotNone(app.camera)
            self.assertIsNotNone(app.recognition_system)
    
    def test_kivy_face_app_ui_elements(self):
        """Test KivyFaceApp UI elements"""
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_kivy_app.return_value = mock_app_instance
            
            app = KivyFaceApp()
            
            # Test UI element creation
            app.create_ui_elements()
            
            # Verify UI elements were created
            self.assertIsNotNone(app.camera_view)
            self.assertIsNotNone(app.recognition_button)
            self.assertIsNotNone(app.status_label)
            self.assertIsNotNone(app.results_display)
    
    def test_kivy_face_app_camera_control(self):
        """Test KivyFaceApp camera control"""
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_app_instance.camera = Mock()
            mock_kivy_app.return_value = mock_app_instance
            
            app = KivyFaceApp()
            
            # Test camera start
            mock_app_instance.camera.start.return_value = True
            success = app.start_camera()
            self.assertTrue(success)
            
            # Test camera stop
            mock_app_instance.camera.stop.return_value = True
            success = app.stop_camera()
            self.assertTrue(success)
            
            # Test camera capture
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_app_instance.camera.capture.return_value = test_image
            image = app.capture_frame()
            self.assertIsNotNone(image)
            np.testing.assert_array_equal(image, test_image)
    
    def test_kivy_face_app_face_recognition(self):
        """Test KivyFaceApp face recognition"""
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_app_instance.recognition_system = Mock()
            mock_app_instance.recognition_system.recognize.return_value = {
                'person_id': 'mobile_person',
                'name': 'Mobile User',
                'confidence': 0.95
            }
            mock_kivy_app.return_value = mock_app_instance
            
            app = KivyFaceApp()
            
            # Test face recognition
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = app.recognize_face(test_image)
            
            self.assertIsNotNone(result)
            self.assertEqual(result['person_id'], 'mobile_person')
            self.assertEqual(result['name'], 'Mobile User')
            self.assertEqual(result['confidence'], 0.95)
    
    def test_kivy_face_app_error_handling(self):
        """Test KivyFaceApp error handling"""
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_app_instance.recognition_system = Mock()
            mock_app_instance.recognition_system.recognize.side_effect = Exception("Recognition failed")
            mock_kivy_app.return_value = mock_app_instance
            
            app = KivyFaceApp()
            
            # Test face recognition with error
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = app.recognize_face(test_image)
            
            self.assertIsNone(result)
            # Error should be logged or handled appropriately
    
    def test_kivy_face_app_performance_monitoring(self):
        """Test KivyFaceApp performance monitoring"""
        with patch('interfaces.mobile_interface.KivyFaceApp') as mock_kivy_app:
            mock_app_instance = Mock()
            mock_app_instance.camera = Mock()
            mock_kivy_app.return_value = mock_app_instance
            
            app = KivyFaceApp()
            
            # Test performance tracking
            app.start_performance_tracking()
            self.assertTrue(app.is_tracking_performance)
            
            # Test performance metrics
            metrics = app.get_performance_metrics()
            self.assertIsInstance(metrics, dict)
            self.assertIn('frame_count', metrics)
            self.assertIn('fps', metrics)
            self.assertIn('processing_time', metrics)
            
            # Stop tracking
            app.stop_performance_tracking()
            self.assertFalse(app.is_tracking_performance)
    
    def test_mobile_face_recognition_api_initialization(self):
        """Test MobileFaceRecognitionAPI initialization"""
        api = MobileFaceRecognitionAPI()
        
        self.assertIsNotNone(api)
        self.assertIsNotNone(api.system)
        self.assertIsNotNone(api.config)
        self.assertIsNotNone(app.storage)
    
    def test_mobile_face_recognition_api_upload_face(self):
        """Test mobile face upload"""
        api = MobileFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.add_to_database.return_value = True
        api.system = mock_system
        
        # Create test image data
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Upload face
        success = api.upload_face(
            image=test_image,
            person_name='Mobile User',
            person_id='mobile_person'
        )
        
        self.assertTrue(success)
        mock_system.add_to_database.assert_called_once()
    
    def test_mobile_face_recognition_api_recognize_face(self):
        """Test mobile face recognition"""
        api = MobileFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.recognize.return_value = {
            'person_id': 'mobile_person',
            'name': 'Mobile User',
            'confidence': 0.95
        }
        api.system = mock_system
        
        # Create test image data
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Recognize face
        result = api.recognize_face(test_image)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['person_id'], 'mobile_person')
        self.assertEqual(result['name'], 'Mobile User')
        self.assertEqual(result['confidence'], 0.95)
    
    def test_mobile_face_recognition_api_sync_data(self):
        """Test data synchronization"""
        api = MobileFaceRecognitionAPI()
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.get_faces.return_value = [self.test_face_data]
        mock_storage.sync.return_value = True
        api.storage = mock_storage
        
        # Mock system
        mock_system = Mock()
        mock_system.add_to_database.return_value = True
        api.system = mock_system
        
        # Sync data
        success = api.sync_data()
        
        self.assertTrue(success)
        mock_storage.get_faces.assert_called_once()
        mock_storage.sync.assert_called_once()
        mock_system.add_to_database.assert_called_once()
    
    def test_mobile_face_recognition_api_cache_management(self):
        """Test cache management"""
        api = MobileFaceRecognitionAPI()
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.get_cached_faces.return_value = [self.test_face_data]
        mock_storage.clear_cache.return_value = True
        mock_storage.cleanup_old_cache.return_value = True
        api.storage = mock_storage
        
        # Test cache operations
        cached_faces = api.get_cached_faces()
        self.assertEqual(len(cached_faces), 1)
        
        success = api.clear_cache()
        self.assertTrue(success)
        
        success = api.cleanup_old_cache(max_age=86400)  # 24 hours
        self.assertTrue(success)
    
    def test_mobile_interface_offline_mode(self):
        """Test offline mode functionality"""
        interface = MobileInterface()
        
        # Enable offline mode
        interface.enable_offline_mode()
        
        # Check if offline mode is enabled
        self.assertTrue(interface.offline_mode)
        
        # Test offline operations
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = interface.recognize_face_offline(test_image)
        
        # Should work in offline mode
        self.assertIsNotNone(result)
    
    def test_mobile_interface_network_handling(self):
        """Test network handling"""
        interface = MobileInterface()
        
        # Mock network operations
        with patch('interfaces.mobile_interface.NetworkManager') as mock_network:
            mock_network_instance = Mock()
            mock_network_instance.is_connected.return_value = True
            mock_network_instance.upload_data.return_value = True
            mock_network_instance.download_data.return_value = True
            mock_network.return_value = mock_network_instance
            
            # Test network operations
            is_connected = interface.check_network_connection()
            self.assertTrue(is_connected)
            
            success = interface.upload_data(self.test_face_data)
            self.assertTrue(success)
            
            success = interface.download_data('person_id')
            self.assertTrue(success)
    
    def test_mobile_interface_battery_monitoring(self):
        """Test battery monitoring"""
        interface = MobileInterface()
        
        # Mock battery info
        with patch('interfaces.mobile_interface.BatteryInfo') as mock_battery:
            mock_battery_instance = Mock()
            mock_battery_instance.level.return_value = 0.75  # 75%
            mock_battery_instance.is_charging.return_value = False
            mock_battery.return_value = mock_battery_instance
            
            # Test battery monitoring
            battery_level = interface.get_battery_level()
            self.assertEqual(battery_level, 0.75)
            
            is_charging = interface.is_charging()
            self.assertFalse(is_charging)
            
            # Low battery handling
            mock_battery_instance.level.return_value = 0.15  # 15%
            low_battery = interface.is_low_battery()
            self.assertTrue(low_battery)
    
    def test_mobile_interface_sensor_fusion(self):
        """Test sensor fusion"""
        interface = MobileInterface()
        
        # Mock sensor data
        mock_sensor_data = {
            'accelerometer': [0.1, 0.2, 0.3],
            'gyroscope': [0.01, 0.02, 0.03],
            'magnetometer': [0.5, 0.6, 0.7]
        }
        
        with patch('interfaces.mobile_interface.SensorManager') as mock_sensor:
            mock_sensor_instance = Mock()
            mock_sensor_instance.get_sensor_data.return_value = mock_sensor_data
            mock_sensor.return_value = mock_sensor_instance
            
            # Test sensor fusion
            fused_data = interface.get_sensor_fusion()
            
            self.assertIsNotNone(fused_data)
            self.assertIn('orientation', fused_data)
            self.assertIn('motion_state', fused_data)
    
    def test_mobile_interface_permission_handling(self):
        """Test permission handling"""
        interface = MobileInterface()
        
        # Mock permission manager
        with patch('interfaces.mobile_interface.PermissionManager') as mock_permission:
            mock_permission_instance = Mock()
            mock_permission_instance.check_permission.return_value = True
            mock_permission_instance.request_permission.return_value = True
            mock_permission.return_value = mock_permission_instance
            
            # Test permission checking
            has_permission = interface.check_camera_permission()
            self.assertTrue(has_permission)
            
            # Test permission requesting
            success = interface.request_camera_permission()
            self.assertTrue(success)
    
    def test_mobile_interface_performance_optimization(self):
        """Test performance optimization"""
        interface = MobileInterface()
        
        # Mock performance profiler
        with patch('interfaces.mobile_interface.PerformanceProfiler') as mock_profiler:
            mock_profiler_instance = Mock()
            mock_profiler_instance.get_system_resources.return_value = {
                'cpu_usage': 0.5,
                'memory_usage': 0.6,
                'battery_level': 0.7
            }
            mock_profiler_instance.optimize_resources.return_value = True
            mock_profiler.return_value = mock_profiler_instance
            
            # Test performance optimization
            resources = interface.get_system_resources()
            self.assertIn('cpu_usage', resources)
            self.assertIn('memory_usage', resources)
            self.assertIn('battery_level', resources)
            
            success = interface.optimize_resources()
            self.assertTrue(success)
    
    def test_mobile_interface_locale_handling(self):
        """Test locale handling"""
        interface = MobileInterface()
        
        # Test locale detection
        current_locale = interface.get_current_locale()
        self.assertIsNotNone(current_locale)
        
        # Test locale setting
        success = interface.set_locale('en_US')
        self.assertTrue(success)
        
        # Test translation
        translation = interface.get_translation('welcome_message')
        self.assertIsInstance(translation, str)
    
    def test_mobile_interface_orientation_handling(self):
        """Test orientation handling"""
        interface = MobileInterface()
        
        # Test orientation detection
        current_orientation = interface.get_orientation()
        self.assertIn(current_orientation, ['portrait', 'landscape'])
        
        # Test orientation setting
        success = interface.set_orientation('landscape')
        self.assertTrue(success)
        
        # Test orientation change handling
        interface.on_orientation_change()
        # Should handle orientation change appropriately
    
    def test_mobile_interface_theme_management(self):
        """Test theme management"""
        interface = MobileInterface()
        
        # Test theme detection
        current_theme = interface.get_current_theme()
        self.assertIn(current_theme, ['light', 'dark'])
        
        # Test theme setting
        success = interface.set_theme('dark')
        self.assertTrue(success)
        
        # Test theme switching
        interface.toggle_theme()
        # Should switch to opposite theme
    
    def test_mobile_interface_voice_commands(self):
        """Test voice commands"""
        interface = MobileInterface()
        
        # Mock voice recognition
        with patch('interfaces.mobile_interface.VoiceRecognition') as mock_voice:
            mock_voice_instance = Mock()
            mock_voice_instance.recognize.return_value = 'capture face'
            mock_voice.return_value = mock_voice_instance
            
            # Test voice command recognition
            command = interface.listen_for_command()
            self.assertEqual(command, 'capture face')
            
            # Test command execution
            success = interface.execute_voice_command(command)
            self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()