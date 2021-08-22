"""Test web interface functionality"""

import unittest
import os
import sys
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock, call
from io import BytesIO
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from interfaces.web_interface import WebInterface, WebFaceRecognitionAPI, WebConfigManager

class TestWebInterface(unittest.TestCase):
    """Test cases for Web Interface"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'web': {
                'host': 'localhost',
                'port': 5000,
                'debug': True,
                'secret_key': 'test_secret_key',
                'max_content_length': 16777216
            },
            'security': {
                'session_timeout': 3600,
                'max_login_attempts': 5,
                'lockout_duration': 300
            },
            'features': {
                'enable_realtime': True,
                'enable_batch_processing': True,
                'enable_api_access': True,
                'enable_admin_panel': True
            }
        }
        
        # Create test user data
        self.test_user = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password_hash': 'hashed_password',
            'role': 'user',
            'created_at': time.time()
        }
        
        # Create test face data
        self.test_face_data = {
            'person_id': 'test_person',
            'name': 'Test Person',
            'embeddings': np.random.randn(512).tolist(),
            'metadata': {
                'source': 'web_upload',
                'timestamp': time.time(),
                'confidence_threshold': 0.8
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_web_interface_initialization(self):
        """Test WebInterface initialization"""
        interface = WebInterface()
        
        self.assertIsNotNone(interface)
        self.assertIsNotNone(interface.app)
        self.assertIsNotNone(interface.api)
        self.assertEqual(interface.host, 'localhost')
        self.assertEqual(interface.port, 5000)
        self.assertTrue(interface.debug)
    
    def test_web_interface_custom_config(self):
        """Test WebInterface with custom configuration"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        
        # Save test config
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # Initialize with custom config
        interface = WebInterface(config_file=config_file)
        
        self.assertEqual(interface.host, 'localhost')
        self.assertEqual(interface.port, 5000)
        self.assertTrue(interface.debug)
    
    def test_web_interface_start_stop(self):
        """Test WebInterface start and stop"""
        interface = WebInterface()
        
        # Mock threading
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            # Start interface
            success = interface.start()
            self.assertTrue(success)
            mock_thread.start.assert_called_once()
            
            # Stop interface
            interface.stop()
            mock_thread_instance.join.assert_called_once()
    
    def test_web_face_recognition_api_initialization(self):
        """Test WebFaceRecognitionAPI initialization"""
        api = WebFaceRecognitionAPI()
        
        self.assertIsNotNone(api)
        self.assertIsNotNone(api.system)
        self.assertIsNotNone(api.app)
        self.assertIsNotNone(api.config)
        
        # Check routes are registered
        self.assertTrue(hasattr(api, 'register_routes'))
    
    def test_web_face_recognition_api_register_routes(self):
        """Test route registration"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        api.system = mock_system
        
        # Register routes
        api.register_routes()
        
        # Check that routes were registered (this is basic validation)
        # In a real test, we would check the Flask app routes
        self.assertTrue(hasattr(api.app, 'route'))
    
    def test_web_config_manager_initialization(self):
        """Test WebConfigManager initialization"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        
        # Save test config
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        manager = WebConfigManager(config_file=config_file)
        
        self.assertIsNotNone(manager)
        self.assertEqual(manager.config_file, config_file)
        self.assertIsNotNone(manager.config)
        self.assertEqual(manager.config['web']['host'], 'localhost')
        self.assertEqual(manager.config['web']['port'], 5000)
    
    def test_web_config_manager_save_load_config(self):
        """Test config saving and loading"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        
        # Initialize manager
        manager = WebConfigManager(config_file=config_file)
        
        # Modify config
        manager.config['web']['port'] = 8080
        
        # Save config
        success = manager.save_config()
        self.assertTrue(success)
        self.assertTrue(os.path.exists(config_file))
        
        # Create new manager and load config
        new_manager = WebConfigManager(config_file=config_file)
        success = new_manager.load_config()
        self.assertTrue(success)
        self.assertEqual(new_manager.config['web']['port'], 8080)
    
    def test_web_config_manager_get_set_values(self):
        """Test getting and setting config values"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        
        manager = WebConfigManager(config_file=config_file)
        
        # Get web config
        web_config = manager.get_config('web')
        self.assertEqual(web_config['host'], 'localhost')
        self.assertEqual(web_config['port'], 5000)
        
        # Set web config
        success = manager.set_config('web', {'host': '127.0.0.1', 'port': 9000})
        self.assertTrue(success)
        
        # Verify change
        web_config = manager.get_config('web')
        self.assertEqual(web_config['host'], '127.0.0.1')
        self.assertEqual(web_config['port'], 9000)
    
    def test_web_config_manager_update_config(self):
        """Test config updating"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        
        manager = WebConfigManager(config_file=config_file)
        
        # Update config
        update_data = {
            'web': {
                'port': 3000,
                'debug': False
            },
            'security': {
                'session_timeout': 1800
            }
        }
        
        success = manager.update_config(update_data)
        self.assertTrue(success)
        
        # Verify update
        self.assertEqual(manager.config['web']['port'], 3000)
        self.assertEqual(manager.config['web']['debug'], False)
        self.assertEqual(manager.config['security']['session_timeout'], 1800)
        
        # Check that unchanged values remain
        self.assertEqual(manager.config['web']['host'], 'localhost')
    
    def test_web_config_manager_validation(self):
        """Test config validation"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        
        manager = WebConfigManager(config_file=config_file)
        
        # Test valid config
        valid_config = {
            'web': {
                'host': 'localhost',
                'port': 5000
            }
        }
        
        is_valid = manager.validate_config(valid_config)
        self.assertTrue(is_valid)
        
        # Test invalid config (missing required fields)
        invalid_config = {
            'web': {
                'host': 'localhost'
                # Missing port
            }
        }
        
        is_valid = manager.validate_config(invalid_config)
        self.assertFalse(is_valid)
    
    def test_web_face_recognition_api_upload_face(self):
        """Test face upload API endpoint"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.add_to_database.return_value = True
        mock_system.recognize.return_value = {
            'person_id': 'test_person',
            'name': 'Test Person',
            'confidence': 0.95
        }
        api.system = mock_system
        
        # Create test image data
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        image_data = BytesIO()
        import cv2
        cv2.imencode('.jpg', test_image)[1].tofile(image_data)
        
        # Mock request
        mock_request = Mock()
        mock_request.files = {'face_image': ('test.jpg', image_data, 'image/jpeg')}
        mock_request.form = {
            'person_name': 'Test Person',
            'person_id': 'test_person'
        }
        
        # Mock Flask request context
        with patch('interfaces.web_interface.request', mock_request):
            with api.app.test_request_context('/api/upload_face', method='POST'):
                response = api.upload_face()
                
                self.assertEqual(response.status_code, 200)
                response_data = json.loads(response.data)
                self.assertIn('success', response_data)
                self.assertIn('person_id', response_data)
                self.assertEqual(response_data['person_id'], 'test_person')
    
    def test_web_face_recognition_api_recognize_face(self):
        """Test face recognition API endpoint"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.recognize.return_value = {
            'person_id': 'test_person',
            'name': 'Test Person',
            'confidence': 0.95
        }
        api.system = mock_system
        
        # Create test image data
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        image_data = BytesIO()
        import cv2
        cv2.imencode('.jpg', test_image)[1].tofile(image_data)
        
        # Mock request
        mock_request = Mock()
        mock_request.files = {'face_image': ('test.jpg', image_data, 'image/jpeg')}
        
        # Mock Flask request context
        with patch('interfaces.web_interface.request', mock_request):
            with api.app.test_request_context('/api/recognize', method='POST'):
                response = api.recognize_face()
                
                self.assertEqual(response.status_code, 200)
                response_data = json.loads(response.data)
                self.assertIn('person_id', response_data)
                self.assertIn('name', response_data)
                self.assertIn('confidence', response_data)
                self.assertEqual(response_data['person_id'], 'test_person')
    
    def test_web_face_recognition_api_get_person_list(self):
        """Test get person list API endpoint"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.get_all_persons.return_value = [
            {'person_id': 'person1', 'name': 'Person 1'},
            {'person_id': 'person2', 'name': 'Person 2'}
        ]
        api.system = mock_system
        
        # Mock Flask request context
        with api.app.test_request_context('/api/persons'):
            response = api.get_person_list()
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.data)
            self.assertIn('persons', response_data)
            self.assertEqual(len(response_data['persons']), 2)
    
    def test_web_face_recognition_api_delete_person(self):
        """Test delete person API endpoint"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.remove_from_database.return_value = True
        api.system = mock_system
        
        # Mock Flask request context
        with api.app.test_request_context('/api/persons/person1', method='DELETE'):
            response = api.delete_person()
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.data)
            self.assertIn('success', response_data)
            self.assertTrue(response_data['success'])
    
    def test_web_face_recognition_api_error_handling(self):
        """Test API error handling"""
        api = WebFaceRecognitionAPI()
        
        # Mock system with error
        mock_system = Mock()
        mock_system.recognize.side_effect = Exception("Recognition failed")
        api.system = mock_system
        
        # Create test image data
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        image_data = BytesIO()
        import cv2
        cv2.imencode('.jpg', test_image)[1].tofile(image_data)
        
        # Mock request
        mock_request = Mock()
        mock_request.files = {'face_image': ('test.jpg', image_data, 'image/jpeg')}
        
        # Mock Flask request context
        with patch('interfaces.web_interface.request', mock_request):
            with api.app.test_request_context('/api/recognize', method='POST'):
                response = api.recognize_face()
                
                self.assertEqual(response.status_code, 500)
                response_data = json.loads(response.data)
                self.assertIn('error', response_data)
    
    def test_web_face_recognition_api_authentication(self):
        """Test API authentication"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        api.system = mock_system
        
        # Mock Flask request without authentication
        with api.app.test_request_context('/api/protected'):
            response = api.get_system_stats()
            
            self.assertEqual(response.status_code, 401)
    
    def test_web_face_recognition_api_rate_limiting(self):
        """Test API rate limiting"""
        api = WebFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        api.system = mock_system
        
        # Create test image data
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        image_data = BytesIO()
        import cv2
        cv2.imencode('.jpg', test_image)[1].tofile(image_data)
        
        # Mock request with fake IP
        mock_request = Mock()
        mock_request.remote_addr = '127.0.0.1'
        mock_request.files = {'face_image': ('test.jpg', image_data, 'image/jpeg')}
        
        # Mock rate limiter
        with patch('interfaces.web_interface.rate_limiter') as mock_rate_limiter:
            mock_rate_limiter.is_limited.return_value = True
            
            with patch('interfaces.web_interface.request', mock_request):
                with api.app.test_request_context('/api/recognize', method='POST'):
                    response = api.recognize_face()
                    
                    self.assertEqual(response.status_code, 429)
    
    def test_web_interface_cors_handling(self):
        """Test CORS handling"""
        interface = WebInterface()
        
        # Mock Flask request
        mock_request = Mock()
        mock_request.headers = {
            'Origin': 'http://example.com',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        # Mock Flask request context
        with interface.app.test_request_context('/api/test', method='OPTIONS'):
            response = interface.handle_cors_preflight()
            
            self.assertEqual(response.status_code, 200)
            self.assertIn('Access-Control-Allow-Origin', response.headers)
            self.assertIn('Access-Control-Allow-Methods', response.headers)
            self.assertIn('Access-Control-Allow-Headers', response.headers)
    
    def test_web_interface_session_management(self):
        """Test session management"""
        interface = WebInterface()
        
        # Mock session
        mock_session = {
            'user_id': 'testuser',
            'last_activity': time.time(),
            'login_attempts': 0
        }
        
        # Test session creation
        session_id = interface.create_session('testuser')
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, interface.sessions)
        
        # Test session validation
        is_valid = interface.is_session_valid(session_id)
        self.assertTrue(is_valid)
        
        # Test session cleanup
        interface.cleanup_sessions()
        self.assertIn(session_id, interface.sessions)  # Should still exist
        
        # Test session expiration
        interface.sessions[session_id]['last_activity'] = time.time() - 4000  # Expired
        interface.cleanup_sessions()
        self.assertNotIn(session_id, interface.sessions)  # Should be removed
    
    def test_web_interface_user_management(self):
        """Test user management"""
        interface = WebInterface()
        
        # Test user registration
        success = interface.register_user(
            username='testuser',
            email='test@example.com',
            password='testpass'
        )
        self.assertTrue(success)
        self.assertIn('testuser', interface.users)
        
        # Test user login
        auth_result = interface.authenticate_user('testuser', 'testpass')
        self.assertTrue(auth_result['success'])
        self.assertIsNotNone(auth_result['session_id'])
        
        # Test user logout
        session_id = auth_result['session_id']
        success = interface.logout_user(session_id)
        self.assertTrue(success)
        self.assertNotIn(session_id, interface.sessions)
        
        # Test failed authentication
        auth_result = interface.authenticate_user('testuser', 'wrongpass')
        self.assertFalse(auth_result['success'])
    
    def test_web_interface_error_handling(self):
        """Test web interface error handling"""
        interface = WebInterface()
        
        # Test 404 handling
        with interface.app.test_client() as client:
            response = client.get('/nonexistent')
            self.assertEqual(response.status_code, 404)
        
        # Test 500 handling
        @interface.app.route('/error')
        def error_route():
            raise Exception("Test error")
        
        with interface.app.test_client() as client:
            response = client.get('/error')
            self.assertEqual(response.status_code, 500)
    
    def test_web_interface_performance_monitoring(self):
        """Test web interface performance monitoring"""
        interface = WebInterface()
        
        # Mock request timing
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 0.1]  # Start and end times
            
            # Test request timing
            with interface.app.test_client() as client:
                response = client.get('/api/status')
                self.assertEqual(response.status_code, 200)
        
        # Check performance metrics
        self.assertIn('api_status', interface.performance_metrics)
        self.assertGreater(interface.performance_metrics['api_status']['total_requests'], 0)

if __name__ == '__main__':
    unittest.main()