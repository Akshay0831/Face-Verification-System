"""Test enhanced face detection functionality"""

import unittest
import os
import sys
import json
import tempfile
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_detection.detectors.enhanced_detector import EnhancedDetector, HOGDetector, MultiDetector

class TestEnhancedDetector(unittest.TestCase):
    """Test cases for Enhanced Face Detector"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        self.test_images = []
        for i in range(3):
            # Create simple test images (faces in center)
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            # Draw a simple face in the center
            cv2.circle(image, (50, 50), 20, (255, 255, 255), -1)  # Face
            cv2.circle(image, (40, 40), 5, (0, 0, 0), -1)  # Left eye
            cv2.circle(image, (60, 40), 5, (0, 0, 0), -1)  # Right eye
            cv2.circle(image, (50, 55), 5, (0, 0, 0), -1)  # Nose
            cv2.circle(image, (45, 65), 8, (0, 255, 0), 2)  # Mouth
            
            image_path = os.path.join(self.temp_dir, f'test_face_{i}.jpg')
            cv2.imwrite(image_path, image)
            self.test_images.append(image_path)
        
        # Test configuration
        self.test_config = {
            'confidence_threshold': 0.85,
            'enable_gpu': True,
            'max_workers': 4,
            'batch_size': 32,
            'detection_methods': ['hog', 'dlib', 'cnn'],
            'hog_config': {
                'orientations': 9,
                'pixels_per_cell': (8, 8),
                'cells_per_block': (2, 2),
                'scale_factor': 1.1
            },
            'dlib_config': {
                'upsample_factor': 1,
                'min_face_size': 50
            },
            'cnn_config': {
                'model_path': 'mmod_human_face_detector.dat',
                'confidence_threshold': 0.9
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_detector_initialization(self):
        """Test EnhancedDetector initialization"""
        detector = EnhancedDetector(self.test_config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.detectors)
        self.assertIsNotNone(detector.results_cache)
        self.assertEqual(len(detector.detectors), 3)  # HOG, dlib, CNN
        
        # Check configurations
        self.assertEqual(detector.config['confidence_threshold'], 0.85)
        self.assertTrue(detector.config['enable_gpu'])
        self.assertEqual(detector.config['max_workers'], 4)
    
    def test_enhanced_detector_initialization_default_config(self):
        """Test EnhancedDetector initialization with default config"""
        detector = EnhancedDetector()
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertEqual(len(detector.detectors), 3)  # Default detectors
        
        # Check default values
        self.assertEqual(detector.config['confidence_threshold'], 0.8)
        self.assertFalse(detector.config['enable_gpu'])  # GPU disabled by default
        self.assertEqual(detector.config['max_workers'], 2)
    
    def test_enhanced_detector_load_detector(self):
        """Test detector loading"""
        detector = EnhancedDetector(self.test_config)
        
        # Test loading individual detectors
        hog_detector = detector._load_detector('hog')
        self.assertIsNotNone(hog_detector)
        self.assertIsInstance(hog_detector, HOGDetector)
        
        dlib_detector = detector._load_detector('dlib')
        self.assertIsNotNone(dlib_detector)
        # dlib detector would be imported here
        
        cnn_detector = detector._load_detector('cnn')
        self.assertIsNotNone(cnn_detector)
        # CNN detector would be imported here
    
    def test_enhanced_detector_detect_faces(self):
        """Test face detection"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector responses
        mock_hog_response = [
            {
                'face_id': 'hog_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.92,
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog'
            }
        ]
        
        mock_dlib_response = [
            {
                'face_id': 'dlib_001',
                'bbox': [28, 32, 72, 68],
                'confidence': 0.88,
                'landmarks': {'left_eye': (38, 42), 'right_eye': (62, 42)},
                'method': 'dlib'
            }
        ]
        
        mock_cnn_response = [
            {
                'face_id': 'cnn_001',
                'bbox': [32, 28, 68, 72],
                'confidence': 0.95,
                'landmarks': {'left_eye': (42, 38), 'right_eye': (58, 38)},
                'method': 'cnn'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_hog_response
                    mock_dlib.return_value = mock_dlib_response
                    mock_cnn.return_value = mock_cnn_response
                    
                    # Detect faces
                    result = detector.detect_faces(self.test_images[0])
                    
                    self.assertIsNotNone(result)
                    self.assertIn('success', result)
                    self.assertIn('detections', result)
                    self.assertIn('method_scores', result)
                    self.assertIn('best_method', result)
                    self.assertIn('processing_time', result)
                    self.assertIn('method_details', result)
                    
                    # Check result values
                    self.assertTrue(result['success'])
                    self.assertIsInstance(result['detections'], list)
                    self.assertEqual(len(result['detections']), 1)  # One face detected
                    
                    # Check method scores
                    self.assertIn('hog', result['method_scores'])
                    self.assertIn('dlib', result['method_scores'])
                    self.assertIn('cnn', result['method_scores'])
                    
                    # Check best method
                    self.assertIn(result['best_method'], ['hog', 'dlib', 'cnn'])
                    
                    # Check processing time
                    self.assertIsInstance(result['processing_time'], float)
                    self.assertGreater(result['processing_time'], 0)
                    
                    # Check method details
                    self.assertIsInstance(result['method_details'], dict)
                    self.assertIn('hog', result['method_details'])
                    self.assertIn('dlib', result['method_details'])
                    self.assertIn('cnn', result['method_details'])
                    
                    # Verify detectors were called
                    mock_hog.assert_called_once()
                    mock_dlib.assert_called_once()
                    mock_cnn.assert_called_once()
    
    def test_enhanced_detector_detect_faces_multiple_faces(self):
        """Test face detection with multiple faces"""
        # Create test image with multiple faces
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (50, 50), 20, (255, 255, 255), -1)  # Face 1
        cv2.circle(image, (150, 50), 20, (255, 255, 255), -1)  # Face 2
        cv2.circle(image, (50, 150), 20, (255, 255, 255), -1)  # Face 3
        
        image_path = os.path.join(self.temp_dir, 'multiple_faces.jpg')
        cv2.imwrite(image_path, image)
        
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector responses with multiple faces
        mock_response = [
            {
                'face_id': f'face_{i}',
                'bbox': [30 + i*50, 30, 70 + i*50, 70],
                'confidence': 0.9,
                'landmarks': {'left_eye': (40 + i*50, 40), 'right_eye': (60 + i*50, 40)},
                'method': 'hog'
            }
            for i in range(3)
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_response
                    mock_dlib.return_value = mock_response
                    mock_cnn.return_value = mock_response
                    
                    # Detect faces
                    result = detector.detect_faces(image_path)
                    
                    self.assertTrue(result['success'])
                    self.assertEqual(len(result['detections']), 3)
    
    def test_enhanced_detector_no_faces_detected(self):
        """Test face detection when no faces are detected"""
        # Create test image without faces
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = os.path.join(self.temp_dir, 'no_faces.jpg')
        cv2.imwrite(image_path, image)
        
        detector = EnhancedDetector(self.test_config)
        
        # Mock empty responses
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = []
                    mock_dlib.return_value = []
                    mock_cnn.return_value = []
                    
                    # Detect faces
                    result = detector.detect_faces(image_path)
                    
                    self.assertIsNotNone(result)
                    self.assertTrue(result['success'])
                    self.assertEqual(len(result['detections']), 0)
                    self.assertEqual(result['best_method'], None)
    
    def test_enhanced_detector_error_handling(self):
        """Test error handling in face detection"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector error
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            mock_hog.side_effect = Exception("Detection failed")
            
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                mock_dlib.return_value = []
                
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_cnn.return_value = []
                    
                    # Detect faces with error
                    result = detector.detect_faces(self.test_images[0])
                    
                    self.assertIsNotNone(result)
                    self.assertIn('success', result)
                    self.assertIn('error', result)
                    self.assertFalse(result['success'])
                    self.assertIn('Detection failed', result['error'])
    
    def test_enhanced_detector_confidence_threshold(self):
        """Test confidence threshold filtering"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector responses with different confidence scores
        mock_response = [
            {
                'face_id': 'face_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.92,  # Above threshold
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog'
            },
            {
                'face_id': 'face_002',
                'bbox': [40, 40, 80, 80],
                'confidence': 0.75,  # Below threshold
                'landmarks': {'left_eye': (50, 50), 'right_eye': (70, 50)},
                'method': 'hog'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_response
                    mock_dlib.return_value = []
                    mock_cnn.return_value = []
                    
                    # Detect faces
                    result = detector.detect_faces(self.test_images[0])
                    
                    self.assertTrue(result['success'])
                    # Only faces above confidence threshold should be returned
                    self.assertEqual(len(result['detections']), 1)
                    self.assertEqual(result['detections'][0]['confidence'], 0.92)
    
    def test_enhanced_detector_batch_detection(self):
        """Test batch face detection"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector responses
        mock_response = [
            {
                'face_id': 'face_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.92,
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_response
                    mock_dlib.return_value = mock_response
                    mock_cnn.return_value = mock_response
                    
                    # Batch detect faces
                    results = detector.detect_faces_batch(self.test_images)
                    
                    self.assertIsNotNone(results)
                    self.assertEqual(len(results), len(self.test_images))
                    
                    # Check each result
                    for i, result in enumerate(results):
                        self.assertIn('success', result)
                        self.assertIn('detections', result)
                        self.assertIn('processing_time', result)
                        
                        # Verify detectors were called for each image
                        mock_hog.assert_any_call(self.test_images[i])
                        mock_dlib.assert_any_call(self.test_images[i])
                        mock_cnn.assert_any_call(self.test_images[i])
    
    def test_enhanced_detector_method_scoring(self):
        """Test method scoring system"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock different method performances
        mock_hog_response = [
            {
                'face_id': 'hog_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.85,
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog'
            }
        ]
        
        mock_dlib_response = [
            {
                'face_id': 'dlib_001',
                'bbox': [28, 32, 72, 68],
                'confidence': 0.92,
                'landmarks': {'left_eye': (38, 42), 'right_eye': (62, 42)},
                'method': 'dlib'
            }
        ]
        
        mock_cnn_response = [
            {
                'face_id': 'cnn_001',
                'bbox': [32, 28, 68, 72],
                'confidence': 0.95,
                'landmarks': {'left_eye': (42, 38), 'right_eye': (58, 38)},
                'method': 'cnn'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_hog_response
                    mock_dlib.return_value = mock_dlib_response
                    mock_cnn.return_value = mock_cnn_response
                    
                    # Detect faces
                    result = detector.detect_faces(self.test_images[0])
                    
                    # Check method scores
                    self.assertIn('hog', result['method_scores'])
                    self.assertIn('dlib', result['method_scores'])
                    self.assertIn('cnn', result['method_scores'])
                    
                    # Scores should be based on confidence and performance
                    self.assertGreater(result['method_scores']['cnn'], 
                                     result['method_scores']['dlib'])
                    self.assertGreater(result['method_scores']['dlib'], 
                                     result['method_scores']['hog'])
                    
                    # Best method should be CNN
                    self.assertEqual(result['best_method'], 'cnn')
    
    def test_enhanced_detector_method_details(self):
        """Test method details tracking"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detailed responses
        mock_hog_response = [
            {
                'face_id': 'hog_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.85,
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog',
                'processing_time': 0.1,
                'detection_features': ['feature1', 'feature2']
            }
        ]
        
        mock_dlib_response = [
            {
                'face_id': 'dlib_001',
                'bbox': [28, 32, 72, 68],
                'confidence': 0.92,
                'landmarks': {'left_eye': (38, 42), 'right_eye': (62, 42)},
                'method': 'dlib',
                'processing_time': 0.15,
                'detection_features': ['feature3', 'feature4']
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                mock_hog.return_value = mock_hog_response
                mock_dlib.return_value = mock_dlib_response
                
                # Detect faces
                result = detector.detect_faces(self.test_images[0])
                
                # Check method details
                self.assertIsInstance(result['method_details'], dict)
                self.assertIn('hog', result['method_details'])
                self.assertIn('dlib', result['method_details'])
                
                # Check each method detail
                hog_detail = result['method_details']['hog']
                self.assertEqual(hog_detail['detection_count'], 1)
                self.assertEqual(hog_detail['avg_confidence'], 0.85)
                self.assertEqual(hog_detail['processing_time'], 0.1)
                self.assertIn('detection_features', hog_detail)
                
                dlib_detail = result['method_details']['dlib']
                self.assertEqual(dlib_detail['detection_count'], 1)
                self.assertEqual(dlib_detail['avg_confidence'], 0.92)
                self.assertEqual(dlib_detail['processing_time'], 0.15)
                self.assertIn('detection_features', dlib_detail)
    
    def test_enhanced_detector_cache_functionality(self):
        """Test detection caching"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector response
        mock_response = [
            {
                'face_id': 'face_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.92,
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_response
                    mock_dlib.return_value = []
                    mock_cnn.return_value = []
                    
                    # First detection (should hit detector)
                    result1 = detector.detect_faces(self.test_images[0])
                    
                    # Second detection (should use cache)
                    result2 = detector.detect_faces(self.test_images[0])
                    
                    # Verify cache was used
                    self.assertTrue(hasattr(detector, 'results_cache'))
                    self.assertTrue(len(detector.results_cache) > 0)
                    
                    # Results should be the same
                    self.assertEqual(result1['detections'], result2['detections'])
                    
                    # Detector should be called only once (second call uses cache)
                    mock_hog.assert_called_once()
    
    def test_hog_detector_initialization(self):
        """Test HOGDetector initialization"""
        hog_config = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'scale_factor': 1.1
        }
        
        detector = HOGDetector(hog_config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.hog)
        self.assertIsNotNone(detector.face_cascade)
        
        # Check configuration
        self.assertEqual(detector.config['orientations'], 9)
        self.assertEqual(detector.config['pixels_per_cell'], (8, 8))
        self.assertEqual(detector.config['cells_per_block'], (2, 2))
        self.assertEqual(detector.config['scale_factor'], 1.1)
    
    def test_hog_detector_detect(self):
        """Test HOG face detection"""
        hog_config = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'scale_factor': 1.1
        }
        
        detector = HOGDetector(hog_config)
        
        # Mock OpenCV read
        with patch('cv2.imread') as mock_imread:
            with patch('cv2.cvtColor') as mock_cvtcolor:
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_imread.return_value = test_image
                mock_cvtcolor.return_value = test_image
                
                # Mock hog.detectMultiScale response
                with patch.object(detector.hog, 'detectMultiScale') as mock_detect:
                    mock_detect.return_value = [(30, 30, 40, 40)]  # One face
                    
                    # Detect faces
                    result = detector.detect(self.test_images[0])
                    
                    self.assertIsNotNone(result)
                    self.assertEqual(len(result), 1)
                    self.assertEqual(result[0]['face_id'], 'hog_001')
                    self.assertEqual(result[0]['bbox'], [30, 30, 70, 70])  # x, y, w, h
                    self.assertEqual(result[0]['method'], 'hog')
                    
                    # Verify OpenCV calls
                    mock_imread.assert_called_once_with(self.test_images[0])
                    mock_cvtcolor.assert_called_once()
                    mock_detect.assert_called_once()
    
    def test_multi_detector_initialization(self):
        """Test MultiDetector initialization"""
        multi_config = {
            'detectors': ['hog', 'cascade'],
            'voting_threshold': 0.6,
            'confidence_threshold': 0.8
        }
        
        detector = MultiDetector(multi_config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.detectors)
        self.assertEqual(len(detector.detectors), 2)
        
        # Check configuration
        self.assertEqual(detector.config['voting_threshold'], 0.6)
        self.assertEqual(detector.config['confidence_threshold'], 0.8)
    
    def test_multi_detector_voting_mechanism(self):
        """Test MultiDetector voting mechanism"""
        multi_config = {
            'detectors': ['hog', 'cascade'],
            'voting_threshold': 0.6,
            'confidence_threshold': 0.8
        }
        
        detector = MultiDetector(multi_config)
        
        # Mock detector responses
        mock_hog_response = [
            {
                'face_id': 'hog_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.92,
                'method': 'hog'
            }
        ]
        
        mock_cascade_response = [
            {
                'face_id': 'cascade_001',
                'bbox': [28, 32, 72, 68],
                'confidence': 0.88,
                'method': 'cascade'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['cascade'], 'detect') as mock_cascade:
                mock_hog.return_value = mock_hog_response
                mock_cascade.return_value = mock_cascade_response
                
                # Detect faces
                result = detector.detect(self.test_images[0])
                
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 1)
                
                # Check voting result
                detection = result[0]
                self.assertIn('vote_score', detection)
                self.assertGreater(detection['vote_score'], 0.6)  # Above threshold
                self.assertEqual(detection['method'], 'multi_detector')
    
    def test_multi_detector_low_voting_score(self):
        """Test MultiDetector with low voting score"""
        multi_config = {
            'detectors': ['hog', 'cascade'],
            'voting_threshold': 0.8,  # High threshold
            'confidence_threshold': 0.8
        }
        
        detector = MultiDetector(multi_config)
        
        # Mock detector responses with low confidence
        mock_hog_response = [
            {
                'face_id': 'hog_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.75,  # Below threshold
                'method': 'hog'
            }
        ]
        
        mock_cascade_response = [
            {
                'face_id': 'cascade_001',
                'bbox': [28, 32, 72, 68],
                'confidence': 0.70,  # Below threshold
                'method': 'cascade'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['cascade'], 'detect') as mock_cascade:
                mock_hog.return_value = mock_hog_response
                mock_cascade.return_value = mock_cascade_response
                
                # Detect faces
                result = detector.detect(self.test_images[0])
                
                # No detection should be returned due to low confidence
                self.assertEqual(len(result), 0)
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        detector = EnhancedDetector(self.test_config)
        
        # Mock detector response
        mock_response = [
            {
                'face_id': 'face_001',
                'bbox': [30, 30, 70, 70],
                'confidence': 0.92,
                'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
                'method': 'hog'
            }
        ]
        
        with patch.object(detector.detectors['hog'], 'detect') as mock_hog:
            with patch.object(detector.detectors['dlib'], 'detect') as mock_dlib:
                with patch.object(detector.detectors['cnn'], 'detect') as mock_cnn:
                    mock_hog.return_value = mock_response
                    mock_dlib.return_value = mock_response
                    mock_cnn.return_value = mock_response
                    
                    # Run benchmark
                    benchmark_results = detector.benchmark_detection(self.test_images[0])
                    
                    self.assertIsNotNone(benchmark_results)
                    self.assertIn('total_processing_time', benchmark_results)
                    self.assertIn('method_performance', benchmark_results)
                    self.assertIn('throughput', benchmark_results)
                    self.assertIn('memory_usage', benchmark_results)
                    
                    # Check benchmark values
                    self.assertIsInstance(benchmark_results['total_processing_time'], float)
                    self.assertIsInstance(benchmark_results['throughput'], float)
                    self.assertIsInstance(benchmark_results['memory_usage'], float)
                    
                    # Check method performance
                    self.assertIsInstance(benchmark_results['method_performance'], dict)
                    self.assertIn('hog', benchmark_results['method_performance'])
                    self.assertIn('dlib', benchmark_results['method_performance'])
                    self.assertIn('cnn', benchmark_results['method_performance'])

if __name__ == '__main__':
    unittest.main()