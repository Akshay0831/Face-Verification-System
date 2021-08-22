"""Test enhanced device functionality"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import cv2
import tempfile
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced.devices.cameras import *
from enhanced.devices.optimization import DeviceOptimizer
from enhanced.devices.monitoring import DeviceMonitor

class TestEnhancedDevices(unittest.TestCase):
    """Test cases for Enhanced Device Systems"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_image[100:200, 200:300] = 255  # Add face
        
        # Create test camera parameters
        self.camera_params = {
            'resolution': (640, 480),
            'fps': 30,
            'exposure': 100,
            'gain': 50,
            'white_balance': 'auto',
            'focus': 'auto'
        }
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_usb_camera_initialization(self):
        """Test USBCamera initialization"""
        camera = USBCamera(camera_id=0)
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera.camera_id, 0)
        self.assertEqual(camera.resolution, (640, 480))
        self.assertEqual(camera.fps, 30)
        self.assertIsNone(camera.cap)
    
    def test_usb_camera_initialization_with_params(self):
        """Test USBCamera initialization with parameters"""
        camera = USBCamera(
            camera_id=1,
            resolution=(1280, 720),
            fps=60
        )
        
        self.assertEqual(camera.camera_id, 1)
        self.assertEqual(camera.resolution, (1280, 720))
        self.assertEqual(camera.fps, 60)
    
    def test_usb_camera_open_close(self):
        """Test USB camera open and close operations"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Open camera
            success = camera.open()
            self.assertTrue(success)
            self.assertIsNotNone(camera.cap)
            
            # Close camera
            camera.close()
            mock_cap.release.assert_called_once()
    
    def test_usb_camera_capture_frame(self):
        """Test USB camera frame capture"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Open and capture frame
            camera.open()
            frame = camera.capture_frame()
            
            self.assertIsNotNone(frame)
            np.testing.assert_array_equal(frame, self.test_image)
            mock_cap.read.assert_called_once()
    
    def test_usb_camera_capture_frame_failure(self):
        """Test USB camera frame capture failure"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture with failure
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            
            # Open and capture frame
            camera.open()
            frame = camera.capture_frame()
            
            self.assertIsNone(frame)
    
    def test_usb_camera_multiple_frames(self):
        """Test USB camera multiple frame capture"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            
            # Return different frames
            frames = [self.test_image, self.test_image + 1, self.test_image + 2]
            mock_cap.read.side_effect = [(True, frame) for frame in frames]
            
            # Open and capture multiple frames
            camera.open()
            captured_frames = []
            
            for _ in range(3):
                frame = camera.capture_frame()
                if frame is not None:
                    captured_frames.append(frame)
            
            self.assertEqual(len(captured_frames), 3)
            
            # Verify all frames were different
            self.assertFalse(np.array_equal(captured_frames[0], captured_frames[1]))
            self.assertFalse(np.array_equal(captured_frames[1], captured_frames[2]))
    
    def test_ip_camera_initialization(self):
        """Test IPCamera initialization"""
        camera = IPCamera(
            url='rtsp://localhost:554/stream',
            timeout=5
        )
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera.url, 'rtsp://localhost:554/stream')
        self.assertEqual(camera.timeout, 5)
    
    def test_ip_camera_initialization_with_auth(self):
        """Test IPCamera initialization with authentication"""
        camera = IPCamera(
            url='rtsp://user:pass@localhost:554/stream',
            timeout=10
        )
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera.url, 'rtsp://user:pass@localhost:554/stream')
        self.assertEqual(camera.timeout, 10)
    
    def test_ip_camera_open_close(self):
        """Test IP camera open and close operations"""
        camera = IPCamera(url='rtsp://localhost:554/stream')
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Open camera
            success = camera.open()
            self.assertTrue(success)
            self.assertIsNotNone(camera.cap)
            
            # Close camera
            camera.close()
            mock_cap.release.assert_called_once()
    
    def test_file_camera_initialization(self):
        """Test FileCamera initialization"""
        video_file = os.path.join(self.temp_dir, 'test_video.mp4')
        
        # Create a dummy video file
        with open(video_file, 'wb') as f:
            f.write(b'dummy video content')
        
        camera = FileCamera(file_path=video_file)
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera.file_path, video_file)
    
    def test_file_camera_capture_frame(self):
        """Test FileCamera frame capture"""
        video_file = os.path.join(self.temp_dir, 'test_video.mp4')
        
        # Create a dummy video file
        with open(video_file, 'wb') as f:
            f.write(b'dummy video content')
        
        camera = FileCamera(file_path=video_file)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Open and capture frame
            camera.open()
            frame = camera.capture_frame()
            
            self.assertIsNotNone(frame)
            np.testing.assert_array_equal(frame, self.test_image)
    
    def test_virtual_camera_initialization(self):
        """Test VirtualCamera initialization"""
        camera = VirtualCamera(
            resolution=(640, 480),
            fps=30,
            color='random'
        )
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera.resolution, (640, 480))
        self.assertEqual(camera.fps, 30)
        self.assertEqual(camera.color, 'random')
    
    def test_virtual_camera_generate_frame(self):
        """Test VirtualCamera frame generation"""
        camera = VirtualCamera(
            resolution=(640, 480),
            fps=30,
            color='blue'
        )
        
        # Generate frame
        frame = camera.generate_frame()
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        
        # Check if frame is blue (all blue channel high)
        blue_channel = frame[:, :, 2]
        self.assertTrue(np.all(blue_channel > 200))
    
    def test_virtual_camera_random_color_frames(self):
        """Test VirtualCamera with random color"""
        camera = VirtualCamera(
            resolution=(640, 480),
            fps=30,
            color='random'
        )
        
        # Generate multiple frames
        frames = []
        for _ in range(5):
            frame = camera.generate_frame()
            frames.append(frame)
        
        # Check that frames are different
        unique_frames = set()
        for frame in frames:
            unique_frames.add(tuple(frame.flatten()))
        
        self.assertGreater(len(unique_frames), 1)
    
    def test_device_optimizer_initialization(self):
        """Test DeviceOptimizer initialization"""
        optimizer = DeviceOptimizer()
        
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.optimization_level, 'balanced')
        self.assertEqual(optimizer.target_fps, 30)
        self.assertEqual(optimizer.max_memory_usage, 1024)  # MB
        self.assertIsInstance(optimizer.performance_metrics, dict)
    
    def test_device_optimizer_optimize_camera(self):
        """Test camera optimization"""
        optimizer = DeviceOptimizer()
        
        # Create mock camera
        mock_camera = Mock()
        mock_camera.resolution = (1920, 1080)
        mock_camera.fps = 60
        mock_camera.optimize.return_value = True
        
        # Optimize camera
        success = optimizer.optimize_device(mock_camera)
        
        self.assertTrue(success)
        mock_camera.optimize.assert_called_once()
    
    def test_device_optimizer_adjust_resolution(self):
        """Test resolution adjustment"""
        optimizer = DeviceOptimizer()
        
        # Create mock camera
        mock_camera = Mock()
        mock_camera.resolution = (1920, 1080)
        mock_camera.set_resolution.return_value = True
        
        # Adjust resolution down
        success = optimizer.adjust_resolution(mock_camera, (640, 480))
        
        self.assertTrue(success)
        mock_camera.set_resolution.assert_called_once_with((640, 480))
    
    def test_device_optimizer_adjust_fps(self):
        """Test FPS adjustment"""
        optimizer = DeviceOptimizer()
        
        # Create mock camera
        mock_camera = Mock()
        mock_camera.fps = 60
        mock_camera.set_fps.return_value = True
        
        # Adjust FPS down
        success = optimizer.adjust_fps(mock_camera, 30)
        
        self.assertTrue(success)
        mock_camera.set_fps.assert_called_once_with(30)
    
    def test_device_optimizer_memory_optimization(self):
        """Test memory optimization"""
        optimizer = DeviceOptimizer()
        
        # Create mock camera
        mock_camera = Mock()
        mock_camera.optimize_memory.return_value = True
        
        # Optimize memory
        success = optimizer.optimize_memory(mock_camera)
        
        self.assertTrue(success)
        mock_camera.optimize_memory.assert_called_once()
    
    def test_device_monitor_initialization(self):
        """Test DeviceMonitor initialization"""
        monitor = DeviceMonitor()
        
        self.assertIsNotNone(monitor)
        self.assertIsInstance(monitor.devices, dict)
        self.assertIsInstance(monitor.metrics, dict)
        self.assertEqual(len(monitor.devices), 0)
        self.assertEqual(len(monitor.metrics), 0)
    
    def test_device_monitor_register_device(self):
        """Test device registration"""
        monitor = DeviceMonitor()
        
        # Create mock device
        mock_device = Mock()
        mock_device.name = 'test_camera'
        mock_device.device_type = 'camera'
        mock_device.get_info.return_value = {'resolution': '1920x1080'}
        
        # Register device
        monitor.register_device(mock_device)
        
        self.assertIn('test_camera', monitor.devices)
        self.assertIn('test_camera', monitor.metrics)
        self.assertEqual(len(monitor.devices), 1)
        self.assertEqual(len(monitor.metrics), 1)
    
    def test_device_monitor_monitor_device(self):
        """Test device monitoring"""
        monitor = DeviceMonitor()
        
        # Create mock device
        mock_device = Mock()
        mock_device.name = 'test_camera'
        mock_device.device_type = 'camera'
        mock_device.get_info.return_value = {'resolution': '1920x1080'}
        mock_device.get_performance_stats.return_value = {
            'fps': 29.5,
            'latency': 0.03,
            'cpu_usage': 25.0,
            'memory_usage': 100.0
        }
        
        # Register and monitor device
        monitor.register_device(mock_device)
        
        # Monitor device
        monitor.monitor_device('test_camera')
        
        # Check metrics
        self.assertIn('test_camera', monitor.metrics)
        metrics = monitor.metrics['test_camera']
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)
        
        # Check metric structure
        latest_metric = metrics[-1]
        self.assertIn('timestamp', latest_metric)
        self.assertIn('fps', latest_metric)
        self.assertIn('latency', latest_metric)
        self.assertIn('cpu_usage', latest_metric)
        self.assertIn('memory_usage', latest_metric)
    
    def test_device_monitor_get_device_status(self):
        """Test device status retrieval"""
        monitor = DeviceMonitor()
        
        # Create mock device
        mock_device = Mock()
        mock_device.name = 'test_camera'
        mock_device.device_type = 'camera'
        mock_device.get_info.return_value = {'resolution': '1920x1080'}
        mock_device.get_status.return_value = 'active'
        
        # Register and monitor device
        monitor.register_device(mock_device)
        monitor.monitor_device('test_camera')
        
        # Get device status
        status = monitor.get_device_status('test_camera')
        
        self.assertIsNotNone(status)
        self.assertEqual(status['name'], 'test_camera')
        self.assertEqual(status['device_type'], 'camera')
        self.assertEqual(status['status'], 'active')
        self.assertIn('last_metrics', status)
    
    def test_device_monitor_get_system_overview(self):
        """Test system overview retrieval"""
        monitor = DeviceMonitor()
        
        # Create mock devices
        mock_device1 = Mock()
        mock_device1.name = 'camera1'
        mock_device1.device_type = 'camera'
        mock_device1.get_info.return_value = {'resolution': '1920x1080'}
        mock_device1.get_status.return_value = 'active'
        
        mock_device2 = Mock()
        mock_device2.name = 'camera2'
        mock_device2.device_type = 'camera'
        mock_device2.get_info.return_value = {'resolution': '1280x720'}
        mock_device2.get_status.return_value = 'inactive'
        
        # Register devices
        monitor.register_device(mock_device1)
        monitor.register_device(mock_device2)
        
        # Monitor devices
        monitor.monitor_device('camera1')
        monitor.monitor_device('camera2')
        
        # Get system overview
        overview = monitor.get_system_overview()
        
        self.assertIsNotNone(overview)
        self.assertIn('total_devices', overview)
        self.assertIn('active_devices', overview)
        self.assertIn('device_types', overview)
        self.assertIn('performance_summary', overview)
        
        self.assertEqual(overview['total_devices'], 2)
        self.assertEqual(overview['active_devices'], 1)
        self.assertIn('camera', overview['device_types'])
    
    def test_device_monitor_performance_analysis(self):
        """Test performance analysis"""
        monitor = DeviceMonitor()
        
        # Create mock device
        mock_device = Mock()
        mock_device.name = 'test_camera'
        mock_device.device_type = 'camera'
        mock_device.get_info.return_value = {'resolution': '1920x1080'}
        
        # Register device
        monitor.register_device(mock_device)
        
        # Add some mock metrics
        metrics_data = [
            {'timestamp': time.time() - 100, 'fps': 30.0, 'cpu_usage': 20.0},
            {'timestamp': time.time() - 50, 'fps': 28.5, 'cpu_usage': 25.0},
            {'timestamp': time.time(), 'fps': 29.0, 'cpu_usage': 22.0}
        ]
        
        monitor.metrics['test_camera'] = metrics_data
        
        # Analyze performance
        analysis = monitor.analyze_device_performance('test_camera')
        
        self.assertIsNotNone(analysis)
        self.assertIn('avg_fps', analysis)
        self.assertIn('avg_cpu_usage', analysis)
        self.assertIn('fps_trend', analysis)
        self.assertIn('performance_score', analysis)
        
        self.assertEqual(analysis['avg_fps'], (30.0 + 28.5 + 29.0) / 3)
        self.assertEqual(analysis['avg_cpu_usage'], (20.0 + 25.0 + 22.0) / 3)
    
    def test_camera_error_handling(self):
        """Test camera error handling"""
        # Test camera opening failure
        camera = USBCamera(camera_id=999)  # Non-existent camera
        
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = False
            
            success = camera.open()
            self.assertFalse(success)
            self.assertIsNone(camera.cap)
        
        # Test frame capture from unopened camera
        frame = camera.capture_frame()
        self.assertIsNone(frame)
    
    def test_camera_parameter_validation(self):
        """Test camera parameter validation"""
        # Test invalid resolution
        with self.assertRaises(ValueError):
            camera = USBCamera(camera_id=0, resolution=(0, 0))
        
        # Test invalid FPS
        with self.assertRaises(ValueError):
            camera = USBCamera(camera_id=0, fps=0)
        
        # Test valid parameters
        camera = USBCamera(camera_id=0, resolution=(640, 480), fps=30)
        self.assertEqual(camera.resolution, (640, 480))
        self.assertEqual(camera.fps, 30)
    
    def test_camera_performance_monitoring(self):
        """Test camera performance monitoring"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture with performance tracking
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Open camera
            camera.open()
            
            # Capture multiple frames and measure performance
            start_time = time.time()
            for _ in range(10):
                frame = camera.capture_frame()
            end_time = time.time()
            
            # Check performance
            avg_time_per_frame = (end_time - start_time) / 10
            self.assertLess(avg_time_per_frame, 0.1)  # Should be fast
            
            # Test performance metrics
            metrics = camera.get_performance_metrics()
            self.assertIsInstance(metrics, dict)
            self.assertIn('fps', metrics)
            self.assertIn('latency', metrics)
            self.assertIn('frame_count', metrics)
    
    def test_camera_frame_processing(self):
        """Test camera frame processing"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Open camera
            camera.open()
            
            # Test frame processing
            processed_frame = camera.process_frame(self.test_image, resize=(320, 240))
            
            self.assertIsNotNone(processed_frame)
            self.assertEqual(processed_frame.shape, (240, 320, 3))
            
            # Test frame conversion
            rgb_frame = camera.convert_frame(self.test_image, 'rgb')
            self.assertEqual(rgb_frame.shape, (480, 640, 3))
            
            # Test frame normalization
            normalized_frame = camera.normalize_frame(self.test_image)
            self.assertTrue(np.all(normalized_frame >= 0))
            self.assertTrue(np.all(normalized_frame <= 1))
    
    def test_camera_batch_capture(self):
        """Test camera batch capture"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            
            # Return different frames
            frames = [self.test_image + i for i in range(5)]
            mock_cap.read.side_effect = [(True, frame) for frame in frames]
            
            # Open camera
            camera.open()
            
            # Capture batch
            batch = camera.capture_batch(batch_size=5)
            
            self.assertEqual(len(batch), 5)
            self.assertTrue(all(frame is not None for frame in batch))
    
    def test_camera_automatic_settings_adjustment(self):
        """Test camera automatic settings adjustment"""
        camera = USBCamera(camera_id=0)
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, self.test_image)
            
            # Mock automatic settings
            mock_cap.set.return_value = True
            
            # Open camera
            camera.open()
            
            # Test automatic exposure adjustment
            success = camera.auto_adjust_exposure()
            self.assertTrue(success)
            
            # Test automatic white balance adjustment
            success = camera.auto_adjust_white_balance()
            self.assertTrue(success)
            
            # Test automatic focus adjustment
            success = camera.auto_adjust_focus()
            self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()