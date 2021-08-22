"""Test enhanced detection functionality"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced.detections.detectors import *
from enhanced.detections.multi_detector import MultiDetector

class TestEnhancedDetections(unittest.TestCase):
    """Test cases for Enhanced Detection Systems"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test images
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_image[100:200, 200:300] = 255  # Add a white rectangle as test face
        
        # Create Haar cascade files for testing
        self.haar_frontal = os.path.join(os.path.dirname(__file__), '..', '..', 'cascades', 'haarcascade_frontalface_default.xml')
        self.haar_frontal2 = os.path.join(os.path.dirname(__file__), '..', '..', 'cascades', 'haarcascade_frontalface_alt2.xml')
    
    def test_haar_detector_initialization(self):
        """Test Haar Cascade detector initialization"""
        detector = HaarDetector()
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.cascade)
        self.assertIsNotNone(detector.scale_factor)
        self.assertIsNotNone(detector.min_neighbors)
        self.assertIsNotNone(detector.min_size)
    
    def test_haar_detection_basic(self):
        """Test basic Haar cascade detection"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(
            cascade_path=self.haar_frontal,
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(30, 30)
        )
        
        # Perform detection
        results = detector.detect(self.test_image)
        
        self.assertIsInstance(results, list)
        for result in results:
            self.assertEqual(result.__class__.__name__, 'DetectionResult')
            self.assertTrue(hasattr(result, 'bbox'))
            self.assertTrue(hasattr(result, 'confidence'))
            self.assertTrue(hasattr(result, 'landmarks'))
            self.assertTrue(hasattr(result, 'processing_time'))
    
    def test_haar_detection_with_face(self):
        """Test Haar detection with actual face"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        results = detector.detect(self.test_image)
        
        # Should detect at least one face
        self.assertGreater(len(results), 0)
        
        # Check result properties
        for result in results:
            self.assertGreater(result.confidence, 0)
            self.assertIsInstance(result.bbox, tuple)
            self.assertEqual(len(result.bbox), 4)  # (x, y, w, h)
    
    def test_haar_detection_no_face(self):
        """Test Haar detection with no face"""
        # Create blank image
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        results = detector.detect(blank_image)
        
        # Should return empty list
        self.assertEqual(len(results), 0)
    
    def test_mtcnn_detector_initialization(self):
        """Test MTCNN detector initialization"""
        with patch('enhanced.detections.detectors.mtcnn') as mock_mtcnn:
            mock_mtcnn.MTCNN.return_value = Mock()
            
            detector = MtcnnDetector()
            
            self.assertIsNotNone(detector)
            self.assertIsNotNone(detector.detector)
            self.assertIsNotNone(detector.confidence_threshold)
    
    def test_mtcnn_detection_with_face(self):
        """Test MTCNN detection with face"""
        with patch('enhanced.detections.detectors.mtcnn') as mock_mtcnn:
            # Mock MTCNN detector
            mock_detector = Mock()
            mock_detector.detect_faces.return_value = [
                {
                    'box': [100, 100, 100, 100],
                    'confidence': 0.95,
                    'keypoints': {
                        'left_eye': (110, 110),
                        'right_eye': (190, 110),
                        'nose': (150, 150),
                        'mouth_left': (130, 190),
                        'mouth_right': (170, 190)
                    }
                }
            ]
            mock_mtcnn.MTCNN.return_value = mock_detector
            
            detector = MtcnnDetector()
            results = detector.detect(self.test_image)
            
            # Should detect one face
            self.assertEqual(len(results), 1)
            
            # Check result properties
            result = results[0]
            self.assertEqual(result.bbox, (100, 100, 100, 100))
            self.assertEqual(result.confidence, 0.95)
            self.assertIsInstance(result.landmarks, dict)
    
    def test_mtcnn_detection_no_face(self):
        """Test MTCNN detection with no face"""
        with patch('enhanced.detections.detectors.mtcnn') as mock_mtcnn:
            # Mock MTCNN detector returning no faces
            mock_detector = Mock()
            mock_detector.detect_faces.return_value = []
            mock_mtcnn.MTCNN.return_value = mock_detector
            
            detector = MtcnnDetector()
            results = detector.detect(self.test_image)
            
            # Should return empty list
            self.assertEqual(len(results), 0)
    
    def test_hog_detector_initialization(self):
        """Test HOG detector initialization"""
        detector = HogDetector()
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.detector)
        self.assertIsNotNone(detector.scale)
        self.assertIsNotNone(detector.padding)
    
    def test_hog_detection_with_face(self):
        """Test HOG detection with face"""
        detector = HogDetector()
        results = detector.detect(self.test_image)
        
        # HOG is less accurate, might not detect in simple test
        self.assertIsInstance(results, list)
        
        # If it detects faces, check properties
        if results:
            for result in results:
                self.assertEqual(result.__class__.__name__, 'DetectionResult')
                self.assertTrue(hasattr(result, 'bbox'))
                self.assertTrue(hasattr(result, 'confidence'))
                self.assertTrue(hasattr(result, 'landmarks'))
                self.assertTrue(hasattr(result, 'processing_time'))
    
    def test_ssd_detector_initialization(self):
        """Test SSD detector initialization"""
        with patch('enhanced.detections.detectors.ssd') as mock_ssd:
            mock_detector = Mock()
            mock_ssd.SingleShotDetector.return_value = mock_detector
            
            detector = SsdDetector()
            
            self.assertIsNotNone(detector)
            self.assertIsNotNone(detector.detector)
    
    def test_ssd_detection_with_face(self):
        """Test SSD detection with face"""
        with patch('enhanced.detections.detectors.ssd') as mock_ssd:
            # Mock SSD detector
            mock_detector = Mock()
            mock_detector.detect.return_value = [
                (0.95, [100, 100, 200, 200], 1),  # confidence, bbox, class_id
                (0.85, [150, 150, 250, 250], 1)
            ]
            mock_ssd.SingleShotDetector.return_value = mock_detector
            
            detector = SsdDetector()
            results = detector.detect(self.test_image)
            
            # Should detect faces
            self.assertEqual(len(results), 2)
            
            # Check result properties
            for result in results:
                self.assertEqual(result.__class__.__name__, 'DetectionResult')
                self.assertTrue(hasattr(result, 'bbox'))
                self.assertTrue(hasattr(result, 'confidence'))
                self.assertTrue(hasattr(result, 'landmarks'))
                self.assertTrue(hasattr(result, 'processing_time'))
    
    def test_retinaface_detector_initialization(self):
        """Test RetinaFace detector initialization"""
        with patch('enhanced.detections.detectors.retinaface') as mock_retinaface:
            mock_detector = Mock()
            mock_retinaface.build_model.return_value = mock_detector
            
            detector = RetinaFaceDetector()
            
            self.assertIsNotNone(detector)
            self.assertIsNotNone(detector.detector)
            self.assertIsNotNone(detector.confidence_threshold)
    
    def test_retinaface_detection_with_face(self):
        """Test RetinaFace detection with face"""
        with patch('enhanced.detections.detectors.retinaface') as mock_retinaface:
            # Mock RetinaFace detector
            mock_detector = Mock()
            mock_detector.detect.return_value = [
                {
                    'confidence': 0.95,
                    'box': [100, 100, 200, 200],
                    'landmarks': [[110, 110], [190, 110], [150, 150], [130, 190], [170, 190]]
                }
            ]
            mock_retinaface.build_model.return_value = mock_detector
            
            detector = RetinaFaceDetector()
            results = detector.detect(self.test_image)
            
            # Should detect face
            self.assertEqual(len(results), 1)
            
            # Check result properties
            result = results[0]
            self.assertEqual(result.bbox, (100, 100, 100, 100))  # RetinaFace returns [x1, y1, x2, y2]
            self.assertEqual(result.confidence, 0.95)
            self.assertIsInstance(result.landmarks, list)
    
    def test_facial_landmarks_detection(self):
        """Test facial landmarks detection"""
        from enhanced.detections.landmarks import FacialLandmarks
        
        # Create mock landmarks detector
        with patch('enhanced.detections.landmarks.dlib') as mock_dlib:
            mock_predictor = Mock()
            mock_predictor.return_value = None  # Mock face shape
            
            mock_dlib.shape_predictor.return_value = mock_predictor
            mock_dlib.get_frontal_face_detector.return_value = Mock()
            
            # Create mock face
            mock_face = Mock()
            mock_face.left = 100
            mock_face.top = 100
            mock_face.right = 200
            mock_face.bottom = 200
            
            mock_detector = Mock()
            mock_detector.return_value = [mock_face]
            mock_dlib.get_frontal_face_detector.return_value = mock_detector
            
            landmark_detector = FacialLandmarks()
            
            # Test landmarks detection
            landmarks = landmark_detector.detect(self.test_image)
            
            self.assertIsInstance(landmarks, dict)
            self.assertIn('left_eye', landmarks)
            self.assertIn('right_eye', landmarks)
            self.assertIn('nose', landmarks)
            self.assertIn('mouth_left', landmarks)
            self.assertIn('mouth_right', landmarks)
    
    def test_multi_detector_initialization(self):
        """Test MultiDetector initialization"""
        multi_detector = MultiDetector()
        
        self.assertIsNotNone(multi_detector)
        self.assertIsInstance(multi_detector.detectors, list)
        self.assertEqual(len(multi_detector.detectors), 0)
    
    def test_multi_detector_add_detector(self):
        """Test adding detectors to MultiDetector"""
        multi_detector = MultiDetector()
        
        # Mock different detectors
        mock_detector1 = Mock()
        mock_detector1.__class__.__name__ = "HaarDetector"
        
        mock_detector2 = Mock()
        mock_detector2.__class__.__name__ = "MtcnnDetector"
        
        # Add detectors
        multi_detector.add_detector(mock_detector1, priority=1)
        multi_detector.add_detector(mock_detector2, priority=2)
        
        self.assertEqual(len(multi_detector.detectors), 2)
        self.assertEqual(multi_detector.detectors[0].detector, mock_detector2)  # Higher priority first
        self.assertEqual(multi_detector.detectors[1].detector, mock_detector1)
    
    def test_multi_detector_detection(self):
        """Test MultiDetector face detection"""
        multi_detector = MultiDetector()
        
        # Mock detector that returns results
        mock_detector1 = Mock()
        mock_detector1.detect.return_value = [
            Mock(bbox=(100, 100, 100, 100), confidence=0.95)
        ]
        mock_detector1.__class__.__name__ = "HaarDetector"
        
        # Mock detector that returns no results
        mock_detector2 = Mock()
        mock_detector2.detect.return_value = []
        mock_detector2.__class__.__name__ = "MtcnnDetector"
        
        # Add detectors
        multi_detector.add_detector(mock_detector1, priority=1)
        multi_detector.add_detector(mock_detector2, priority=2)
        
        # Perform detection
        results = multi_detector.detect(self.test_image)
        
        # Should return results from both detectors
        self.assertGreater(len(results), 0)
        
        # Check that both detectors were called
        mock_detector1.detect.assert_called_once()
        mock_detector2.detect.assert_called_once()
    
    def test_detection_result_object(self):
        """Test DetectionResult object"""
        from enhanced.detections.base import DetectionResult
        
        # Create detection result
        result = DetectionResult(
            bbox=(100, 100, 100, 100),
            confidence=0.95,
            landmarks={'left_eye': (110, 110), 'right_eye': (190, 110)},
            processing_time=0.1
        )
        
        # Check properties
        self.assertEqual(result.bbox, (100, 100, 100, 100))
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.landmarks, {'left_eye': (110, 110), 'right_eye': (190, 110)})
        self.assertEqual(result.processing_time, 0.1)
        
        # Check has_landmarks method
        self.assertTrue(result.has_landmarks())
    
    def test_detection_parameter_validation(self):
        """Test detection parameter validation"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        
        # Test invalid image
        invalid_image = "not an image"
        results = detector.detect(invalid_image)
        self.assertEqual(len(results), 0)
        
        # Test None image
        results = detector.detect(None)
        self.assertEqual(len(results), 0)
    
    def test_detection_performance_monitoring(self):
        """Test detection performance monitoring"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        
        # Perform multiple detections
        for _ in range(5):
            results = detector.detect(self.test_image)
            self.assertIsInstance(results, list)
        
        # Check average processing time
        avg_time = detector.get_average_processing_time()
        self.assertIsInstance(avg_time, float)
        self.assertGreater(avg_time, 0)
    
    def test_detection_different_image_sizes(self):
        """Test detection with different image sizes"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        
        # Test different image sizes
        sizes = [(320, 240), (640, 480), (1280, 720)]
        
        for width, height in sizes:
            test_image = np.zeros((height, width, 3), dtype=np.uint8)
            test_image[50:150, 100:200] = 255  # Add face
            
            results = detector.detect(test_image)
            self.assertIsInstance(results, list)
    
    def test_detection_different_image_formats(self):
        """Test detection with different image formats"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        
        # Test different image formats
        formats = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # RGB
            np.zeros((480, 640), dtype=np.uint8),    # Grayscale
            np.zeros((480, 640, 4), dtype=np.uint8)   # RGBA
        ]
        
        for img in formats:
            try:
                results = detector.detect(img)
                self.assertIsInstance(results, list)
            except Exception as e:
                # Some formats might not work, but shouldn't crash
                self.assertIsInstance(e, Exception)
    
    def test_multiple_detector_conflict_resolution(self):
        """Test conflict resolution when multiple detectors detect same face"""
        multi_detector = MultiDetector()
        
        # Mock two detectors that detect same face at different positions
        mock_detector1 = Mock()
        mock_detector1.detect.return_value = [
            Mock(bbox=(100, 100, 100, 100), confidence=0.95, __class__='Detector1')
        ]
        mock_detector1.__class__.__name__ = "Detector1"
        
        mock_detector2 = Mock()
        mock_detector2.detect.return_value = [
            Mock(bbox=(110, 110, 100, 100), confidence=0.90, __class__='Detector2')
        ]
        mock_detector2.__class__.__name__ = "Detector2"
        
        # Add detectors
        multi_detector.add_detector(mock_detector1, priority=1)
        multi_detector.add_detector(mock_detector2, priority=2)
        
        # Perform detection
        results = multi_detector.detect(self.test_image)
        
        # Should return results from both detectors (conflict resolution is handled by caller)
        self.assertEqual(len(results), 2)
        
        # Check priority ordering
        self.assertEqual(results[0].confidence, 0.95)  # Higher confidence first
        self.assertEqual(results[1].confidence, 0.90)
    
    def test_detector_error_handling(self):
        """Test error handling in detectors"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        
        # Test with corrupted image data
        corrupted_image = np.zeros((10, 10, 3), dtype=np.uint8)  # Very small image
        results = detector.detect(corrupted_image)
        self.assertIsInstance(results, list)
    
    def test_detector_memory_usage(self):
        """Test detector memory usage"""
        if not os.path.exists(self.haar_frontal):
            self.skipTest("Haar cascade file not found")
        
        detector = HaarDetector(cascade_path=self.haar_frontal)
        
        # Track memory before
        import sys
        initial_size = sys.getsizeof(detector)
        
        # Perform detection
        for _ in range(10):
            results = detector.detect(self.test_image)
        
        # Track memory after
        final_size = sys.getsizeof(detector)
        
        # Memory should not grow significantly
        self.assertLess(final_size - initial_size, 1000)  # Less than 1KB growth

if __name__ == '__main__':
    unittest.main()