"""Test desktop interface functionality"""

import unittest
import os
import sys
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from interfaces.desktop_interface import DesktopInterface, PyQtFaceApp, DesktopFaceRecognitionAPI

class TestDesktopInterface(unittest.TestCase):
    """Test cases for Desktop Interface"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'desktop': {
                'theme': 'light',
                'language': 'en',
                'window_size': (1280, 720),
                'fullscreen': False,
                'show_fps': True,
                'min_fps': 15,
                'max_fps': 60
            },
            'camera': {
                'resolution': (1280, 720),
                'fps': 30,
                'exposure_mode': 'auto',
                'focus_mode': 'continuous'
            },
            'ui': {
                'enable_notifications': True,
                'enable_sound': True,
                'auto_start_camera': True,
                'display_preferences': True
            }
        }
        
        # Create test user data
        self.test_user = {
            'user_id': 'desktop_user',
            'username': 'Desktop User',
            'permissions': ['camera_access', 'face_recognition', 'admin']
        }
        
        # Create test face data
        self.test_face_data = {
            'person_id': 'desktop_person',
            'name': 'Desktop User',
            'embeddings': np.random.randn(512).tolist(),
            'metadata': {
                'user_id': 'desktop_user',
                'timestamp': time.time(),
                'confidence_threshold': 0.9
            }
        }
        
        # Create QApplication instance for testing
        self.app = QApplication(sys.argv)
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
        
        # Close QApplication
        self.app.quit()
    
    def test_desktop_interface_initialization(self):
        """Test DesktopInterface initialization"""
        interface = DesktopInterface()
        
        self.assertIsNotNone(interface)
        self.assertIsNotNone(interface.config)
        self.assertIsNotNone(interface.app)
        self.assertIsNotNone(interface.main_window)
        self.assertEqual(interface.theme, 'light')
        self.assertEqual(interface.language, 'en')
        self.assertEqual(interface.window_size, (1280, 720))
        self.assertFalse(interface.fullscreen)
    
    def test_desktop_interface_custom_config(self):
        """Test DesktopInterface with custom configuration"""
        config_file = os.path.join(self.temp_dir, 'desktop_config.json')
        
        # Save test config
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # Initialize with custom config
        interface = DesktopInterface(config_file=config_file)
        
        self.assertEqual(interface.theme, 'light')
        self.assertEqual(interface.language, 'en')
        self.assertEqual(interface.window_size, (1280, 720))
        self.assertFalse(interface.fullscreen)
        self.assertTrue(interface.show_fps)
        self.assertEqual(interface.camera_resolution, (1280, 720))
        self.assertEqual(interface.camera_fps, 30)
    
    def test_desktop_interface_start_stop(self):
        """Test DesktopInterface start and stop"""
        interface = DesktopInterface()
        
        # Test starting interface
        success = interface.start()
        self.assertTrue(success)
        self.assertIsNotNone(interface.main_window)
        
        # Test stopping interface
        interface.stop()
        # Should close main window
        if interface.main_window:
            interface.main_window.close()
    
    def test_pyqt_face_app_initialization(self):
        """Test PyQtFaceApp initialization"""
        window = QMainWindow()
        pyqt_app = PyQtFaceApp(window)
        
        self.assertIsNotNone(pyqt_app)
        self.assertIsNotNone(pyqt_app.interface)
        self.assertIsNotNone(pyqt_app.camera)
        self.assertIsNotNone(pyqt_app.recognition_system)
        self.assertIsNotNone(pyqt_app.ui)
    
    def test_pyqt_face_app_ui_creation(self):
        """Test PyQtFaceApp UI creation"""
        window = QMainWindow()
        pyqt_app = PyQtFaceApp(window)
        
        # Test UI creation
        pyqt_app.create_ui()
        
        # Verify UI elements were created
        self.assertIsNotNone(pyqt_app.camera_view)
        self.assertIsNotNone(pyqt_app.recognition_button)
        self.assertIsNotNone(pyqt_app.status_label)
        self.assertIsNotNone(pyqt_app.results_display)
        self.assertIsNotNone(pyqt_app.menu_bar)
        self.assertIsNotNone(pyqt_app.toolbar)
    
    def test_pyqt_face_app_camera_control(self):
        """Test PyQtFaceApp camera control"""
        window = QMainWindow()
        pyqt_app = PyQtFaceApp(window)
        
        # Mock camera
        mock_camera = Mock()
        mock_camera.start.return_value = True
        mock_camera.stop.return_value = True
        mock_camera.is_running.return_value = False
        pyqt_app.camera = mock_camera
        
        # Test camera start
        success = pyqt_app.start_camera()
        self.assertTrue(success)
        mock_camera.start.assert_called_once()
        
        # Test camera stop
        success = pyqt_app.stop_camera()
        self.assertTrue(success)
        mock_camera.stop.assert_called_once()
        
        # Test camera capture
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.capture.return_value = test_image
        
        image = pyqt_app.capture_frame()
        self.assertIsNotNone(image)
        np.testing.assert_array_equal(image, test_image)
    
    def test_pyqt_face_app_face_recognition(self):
        """Test PyQtFaceApp face recognition"""
        window = QMainWindow()
        pyqt_app = PyQtFaceApp(window)
        
        # Mock recognition system
        mock_recognition = Mock()
        mock_recognition.recognize.return_value = {
            'person_id': 'desktop_person',
            'name': 'Desktop User',
            'confidence': 0.95
        }
        pyqt_app.recognition_system = mock_recognition
        
        # Test face recognition
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pyqt_app.recognize_face(test_image)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['person_id'], 'desktop_person')
        self.assertEqual(result['name'], 'Desktop User')
        self.assertEqual(result['confidence'], 0.95)
    
    def test_pyqt_face_app_ui_update(self):
        """Test PyQtFaceApp UI updates"""
        window = QMainWindow()
        pyqt_app = PyQtFaceApp(window)
        
        # Test status update
        pyqt_app.update_status("Ready")
        self.assertEqual(pyqt_app.status_label.text(), "Ready")
        
        # Test result display
        result = {
            'person_id': 'desktop_person',
            'name': 'Desktop User',
            'confidence': 0.95
        }
        pyqt_app.display_result(result)
        
        # Check that result was displayed
        self.assertTrue(len(pyqt_app.results_display.text()) > 0)
    
    def test_pyqt_face_app_error_handling(self):
        """Test PyQtFaceApp error handling"""
        window = QMainWindow()
        pyqt_app = PyQtFaceApp(window)
        
        # Mock recognition system with error
        mock_recognition = Mock()
        mock_recognition.recognize.side_effect = Exception("Recognition failed")
        pyqt_app.recognition_system = mock_recognition
        
        # Test face recognition with error
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pyqt_app.recognize_face(test_image)
        
        self.assertIsNone(result)
        # Should show error message in UI
        self.assertTrue("error" in pyqt_app.status_label.text().lower())
    
    def test_desktop_face_recognition_api_initialization(self):
        """Test DesktopFaceRecognitionAPI initialization"""
        api = DesktopFaceRecognitionAPI()
        
        self.assertIsNotNone(api)
        self.assertIsNotNone(api.system)
        self.assertIsNotNone(api.config)
        self.assertIsNotNone(api.storage)
    
    def test_desktop_face_recognition_api_upload_face(self):
        """Test desktop face upload"""
        api = DesktopFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.add_to_database.return_value = True
        api.system = mock_system
        
        # Create test image
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Upload face
        success = api.upload_face(
            image=test_image,
            person_name='Desktop User',
            person_id='desktop_person'
        )
        
        self.assertTrue(success)
        mock_system.add_to_database.assert_called_once()
    
    def test_desktop_face_recognition_api_recognize_face(self):
        """Test desktop face recognition"""
        api = DesktopFaceRecognitionAPI()
        
        # Mock system
        mock_system = Mock()
        mock_system.recognize.return_value = {
            'person_id': 'desktop_person',
            'name': 'Desktop User',
            'confidence': 0.95
        }
        api.system = mock_system
        
        # Create test image
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Recognize face
        result = api.recognize_face(test_image)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['person_id'], 'desktop_person')
        self.assertEqual(result['name'], 'Desktop User')
        self.assertEqual(result['confidence'], 0.95)
    
    def test_desktop_face_recognition_api_export_data(self):
        """Test data export"""
        api = DesktopFaceRecognitionAPI()
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.export_faces.return_value = [self.test_face_data]
        api.storage = mock_storage
        
        # Export data
        export_data = api.export_data()
        
        self.assertEqual(len(export_data), 1)
        self.assertEqual(export_data[0]['person_id'], 'desktop_person')
    
    def test_desktop_face_recognition_api_import_data(self):
        """Test data import"""
        api = DesktopFaceRecognitionAPI()
        
        # Mock system and storage
        mock_system = Mock()
        mock_system.add_to_database.return_value = True
        api.system = mock_system
        
        mock_storage = Mock()
        mock_storage.save_faces.return_value = True
        api.storage = mock_storage
        
        # Import data
        success = api.import_data([self.test_face_data])
        
        self.assertTrue(success)
        mock_system.add_to_database.assert_called_once()
        mock_storage.save_faces.assert_called_once()
    
    def test_desktop_interface_theme_management(self):
        """Test theme management"""
        interface = DesktopInterface()
        
        # Test theme detection
        current_theme = interface.get_current_theme()
        self.assertIn(current_theme, ['light', 'dark'])
        
        # Test theme setting
        success = interface.set_theme('dark')
        self.assertTrue(success)
        self.assertEqual(interface.theme, 'dark')
        
        # Test theme switching
        interface.toggle_theme()
        self.assertEqual(interface.theme, 'light')
    
    def test_desktop_interface_window_management(self):
        """Test window management"""
        interface = DesktopInterface()
        
        # Test window creation
        interface.create_main_window()
        self.assertIsNotNone(interface.main_window)
        
        # Test window state
        interface.set_window_fullscreen(True)
        self.assertTrue(interface.main_window.isFullScreen())
        
        interface.set_window_fullscreen(False)
        self.assertFalse(interface.main_window.isFullScreen())
        
        # Test window size
        interface.set_window_size(1024, 768)
        self.assertEqual(interface.main_window.width(), 1024)
        self.assertEqual(interface.main_window.height(), 768)
    
    def test_desktop_interface_notification_system(self):
        """Test notification system"""
        interface = DesktopInterface()
        
        # Mock notification system
        with patch('interfaces.desktop_interface.NotificationManager') as mock_notification:
            mock_notification_instance = Mock()
            mock_notification_instance.show_notification.return_value = True
            mock_notification.return_value = mock_notification_instance
            
            # Test notification
            success = interface.show_notification(
                title='Test Notification',
                message='This is a test notification'
            )
            
            self.assertTrue(success)
            mock_notification_instance.show_notification.assert_called_once()
    
    def test_desktop_interface_shortcut_handling(self):
        """Test shortcut handling"""
        interface = DesktopInterface()
        
        # Test shortcut registration
        success = interface.register_shortcut(
            key_sequence='Ctrl+R',
            callback=lambda: print("Recognition triggered")
        )
        
        self.assertTrue(success)
        
        # Test shortcut execution
        interface.execute_shortcut('Ctrl+R')
        # Should trigger callback
    
    def test_desktop_interface_drag_drop_support(self):
        """Test drag and drop support"""
        interface = DesktopInterface()
        
        # Test drag and drop enable
        interface.enable_drag_drop()
        self.assertTrue(interface.drag_drop_enabled)
        
        # Test drag and drop handling
        test_files = ['image1.jpg', 'image2.jpg']
        interface.handle_drop_files(test_files)
        # Should process dropped files
    
    def test_desktop_interface_clipboard_integration(self):
        """Test clipboard integration"""
        interface = DesktopInterface()
        
        # Test clipboard reading
        # Set up clipboard with test data
        clipboard = self.app.clipboard()
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        pixmap = QPixmap.fromImage(QImage(test_image.data, test_image.shape[1], test_image.shape[0], test_image.shape[1] * 3, QImage.Format_RGB888))
        clipboard.setPixmap(pixmap)
        
        # Read from clipboard
        image = interface.read_from_clipboard()
        self.assertIsNotNone(image)
        np.testing.assert_array_equal(image, test_image)
    
    def test_desktop_interface_printing_support(self):
        """Test printing support"""
        interface = DesktopInterface()
        
        # Mock printer
        with patch('interfaces.desktop_interface.QPrintDialog') as mock_print:
            mock_print_instance = Mock()
            mock_print_instance.exec.return_value = 1  # Accepted
            mock_print.return_value = mock_print_instance
            
            # Test print
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            success = interface.print_image(test_image)
            
            self.assertTrue(success)
            mock_print_instance.exec.assert_called_once()
    
    def test_desktop_interface_file_operations(self):
        """Test file operations"""
        interface = DesktopInterface()
        
        # Test file dialog
        with patch('interfaces.desktop_interface.QFileDialog') as mock_dialog:
            mock_dialog_instance = Mock()
            mock_dialog_instance.getOpenFileNames.return_value = ([os.path.join(self.temp_dir, 'test.jpg')], '')
            mock_dialog.return_value = mock_dialog_instance
            
            # Test file selection
            files = interface.select_files()
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith('test.jpg'))
    
    def test_desktop_interface_preferences_management(self):
        """Test preferences management"""
        interface = DesktopInterface()
        
        # Test preference setting
        interface.set_preference('theme', 'dark')
        self.assertEqual(interface.get_preference('theme'), 'dark')
        
        # Test preference saving
        success = interface.save_preferences()
        self.assertTrue(success)
        
        # Test preference loading
        interface.preferences = {}  # Clear preferences
        success = interface.load_preferences()
        self.assertTrue(success)
        self.assertEqual(interface.get_preference('theme'), 'dark')
    
    def test_desktop_interface_backup_system(self):
        """Test backup system"""
        interface = DesktopInterface()
        
        # Mock backup system
        with patch('interfaces.desktop_interface.BackupManager') as mock_backup:
            mock_backup_instance = Mock()
            mock_backup_instance.create_backup.return_value = '/path/to/backup'
            mock_backup_instance.restore_backup.return_value = True
            mock_backup.return_value = mock_backup_instance
            
            # Test backup creation
            backup_path = interface.create_backup()
            self.assertIsNotNone(backup_path)
            
            # Test backup restoration
            success = interface.restore_backup(backup_path)
            self.assertTrue(success)
    
    def test_desktop_interface_update_system(self):
        """Test update system"""
        interface = DesktopInterface()
        
        # Mock update system
        with patch('interfaces.desktop_interface.UpdateManager') as mock_update:
            mock_update_instance = Mock()
            mock_update_instance.check_for_updates.return_value = {'version': '2.0.0', 'available': True}
            mock_update_instance.download_update.return_value = True
            mock_update_instance.install_update.return_value = True
            mock_update.return_value = mock_update_instance
            
            # Test update checking
            update_info = interface.check_for_updates()
            self.assertTrue(update_info['available'])
            self.assertEqual(update_info['version'], '2.0.0')
            
            # Test update download
            success = interface.download_update()
            self.assertTrue(success)
            
            # Test update installation
            success = interface.install_update()
            self.assertTrue(success)
    
    def test_desktop_interface_performance_monitoring(self):
        """Test performance monitoring"""
        interface = DesktopInterface()
        
        # Test performance tracking
        interface.start_performance_tracking()
        self.assertTrue(interface.is_tracking_performance)
        
        # Get performance metrics
        metrics = interface.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('fps', metrics)
        
        # Stop tracking
        interface.stop_performance_tracking()
        self.assertFalse(interface.is_tracking_performance)

if __name__ == '__main__':
    unittest.main()