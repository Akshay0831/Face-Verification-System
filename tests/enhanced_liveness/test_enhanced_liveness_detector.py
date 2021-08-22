"""Test enhanced liveness detection functionality"""

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

from enhanced_liveness.detectors.enhanced_liveness_detector import (
    EnhancedLivenessDetector, 
    MotionAnalysisDetector, 
    TextureAnalysisDetector,
    ThermalDetector
)

class TestEnhancedLivenessDetector(unittest.TestCase):
    """Test cases for Enhanced Liveness Detector"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test face detection result
        self.test_face_detection = {
            'face_id': 'detect_001',
            'bbox': [30, 30, 70, 70],
            'confidence': 0.95,
            'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
            'timestamp': datetime.now()
        }
        
        # Create test image sequence
        self.test_image_sequence = []
        for i in range(5):
            # Create simple animated face
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(image, (50, 50), 20, (255, 255, 255), -1)  # Face
            cv2.circle(image, (40 + i*2, 40), 5, (0, 0, 0), -1)  # Moving left eye
            cv2.circle(image, (60 - i*2, 40), 5, (0, 0, 0), -1)  # Moving right eye
            cv2.circle(image, (50, 55), 5, (0, 0, 0), -1)  # Nose
            
            image_path = os.path.join(self.temp_dir, f'test_frame_{i}.jpg')
            cv2.imwrite(image_path, image)
            self.test_image_sequence.append(image_path)
        
        # Test configuration
        self.test_config = {
            'detection_methods': [
                'motion_analysis',
                'texture_analysis', 
                'thermal',
                'depth_map',
                'iris_response'
            ],
            'confidence_threshold': 0.9,
            'enable_gpu': True,
            'batch_size': 16,
            'max_workers': 4,
            'motion_analysis': {
                'frame_window': 10,
                'motion_threshold': 0.05,
                'eye_blink_threshold': 0.3,
                'head_movement_threshold': 0.2,
                'required_actions': ['eye_blink', 'head_movement']
            },
            'texture_analysis': {
                'gabor_kernels': 8,
                'texture_threshold': 0.7,
                'spoof_threshold': 0.8
            },
            'thermal': {
                'thermal_threshold': 35.0,  # Celsius
                'max_temp_diff': 5.0,
                'thermal_map_resolution': 32
            },
            'depth_map': {
                'depth_threshold': 0.1,
                'max_depth_variance': 0.05,
                'surface_smoothing': True
            },
            'iris_response': {
                'iris_threshold': 0.6,
                'pupil_response_threshold': 0.4,
                'blink_response_threshold': 0.3
            },
            'fusion': {
                'weighted_voting': True,
                'weights': {
                    'motion_analysis': 0.3,
                    'texture_analysis': 0.2,
                    'thermal': 0.2,
                    'depth_map': 0.15,
                    'iris_response': 0.15
                },
                'consensus_threshold': 0.75,
                'require_majority': True
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_liveness_detector_initialization(self):
        """Test EnhancedLivenessDetector initialization"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.detectors)
        self.assertIsNotNone(detector.fusion_engine)
        self.assertIsNotNone(detector.results_cache)
        self.assertEqual(len(detector.detectors), 5)  # All 5 methods
        
        # Check configurations
        self.assertEqual(detector.config['confidence_threshold'], 0.9)
        self.assertTrue(detector.config['enable_gpu'])
        self.assertEqual(detector.config['batch_size'], 16)
    
    def test_enhanced_liveness_detector_initialization_default_config(self):
        """Test EnhancedLivenessDetector initialization with default config"""
        detector = EnhancedLivenessDetector()
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertEqual(len(detector.detectors), 3)  # Default methods
        
        # Check default values
        self.assertEqual(detector.config['confidence_threshold'], 0.85)
        self.assertFalse(detector.config['enable_gpu'])  # GPU disabled by default
        self.assertEqual(detector.config['batch_size'], 8)
    
    def test_enhanced_liveness_detector_load_detector(self):
        """Test detector loading"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Test loading individual detectors
        motion_detector = detector._load_detector('motion_analysis')
        self.assertIsNotNone(motion_detector)
        self.assertIsInstance(motion_detector, MotionAnalysisDetector)
        
        texture_detector = detector._load_detector('texture_analysis')
        self.assertIsNotNone(texture_detector)
        self.assertIsInstance(texture_detector, TextureAnalysisDetector)
        
        thermal_detector = detector._load_detector('thermal')
        self.assertIsNotNone(thermal_detector)
        self.assertIsInstance(thermal_detector, ThermalDetector)
    
    def test_enhanced_liveness_detector_detect_liveness(self):
        """Test liveness detection"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock detector responses
        mock_motion_response = {
            'is_live': True,
            'confidence': 0.92,
            'liveness_score': 0.95,
            'method': 'motion_analysis',
            'actions_detected': ['eye_blink', 'head_movement'],
            'motion_details': {
                'eye_movement': 0.15,
                'head_movement': 0.25,
                'frame_count': 5,
                'processing_time': 0.1
            }
        }
        
        mock_texture_response = {
            'is_live': True,
            'confidence': 0.88,
            'liveness_score': 0.90,
            'method': 'texture_analysis',
            'texture_score': 0.85,
            'spoof_detection': 'none',
            'processing_time': 0.08
        }
        
        mock_thermal_response = {
            'is_live': True,
            'confidence': 0.95,
            'liveness_score': 0.97,
            'method': 'thermal',
            'temperature_range': (36.5, 37.5),
            'thermal_consistency': 0.92,
            'processing_time': 0.12
        }
        
        mock_depth_response = {
            'is_live': True,
            'confidence': 0.87,
            'liveness_score': 0.89,
            'method': 'depth_map',
            'depth_consistency': 0.85,
            'surface_smoothness': 0.88,
            'processing_time': 0.09
        }
        
        mock_iris_response = {
            'is_live': True,
            'confidence': 0.90,
            'liveness_score': 0.92,
            'method': 'iris_response',
            'iris_response_score': 0.88,
            'pupil_response': 0.85,
            'processing_time': 0.07
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_motion:
            with patch.object(detector.detectors['texture_analysis'], 'detect') as mock_texture:
                with patch.object(detector.detectors['thermal'], 'detect') as mock_thermal:
                    with patch.object(detector.detectors['depth_map'], 'detect') as mock_depth:
                        with patch.object(detector.detectors['iris_response'], 'detect') as mock_iris:
                            mock_motion.return_value = mock_motion_response
                            mock_texture.return_value = mock_texture_response
                            mock_thermal.return_value = mock_thermal_response
                            mock_depth.return_value = mock_depth_response
                            mock_iris.return_value = mock_iris_response
                            
                            # Detect liveness
                            result = detector.detect_liveness(self.test_face_detection)
                            
                            self.assertIsNotNone(result)
                            self.assertIn('success', result)
                            self.assertIn('is_live', result)
                            self.assertIn('confidence', result)
                            self.assertIn('liveness_score', result)
                            self.assertIn('method_scores', result)
                            self.assertIn('best_method', result)
                            self.assertIn('processing_time', result)
                            self.assertIn('method_details', result)
                            self.assertIn('fusion_details', result)
                            
                            # Check result values
                            self.assertTrue(result['success'])
                            self.assertTrue(result['is_live'])
                            self.assertEqual(result['confidence'], 0.95)  # Best method
                            self.assertEqual(result['liveness_score'], 0.947)  # Weighted average
                            self.assertEqual(result['best_method'], 'thermal')
                            
                            # Check method scores
                            self.assertIn('motion_analysis', result['method_scores'])
                            self.assertIn('texture_analysis', result['method_scores'])
                            self.assertIn('thermal', result['method_scores'])
                            self.assertIn('depth_map', result['method_scores'])
                            self.assertIn('iris_response', result['method_scores'])
                            
                            # Check processing time
                            self.assertIsInstance(result['processing_time'], float)
                            self.assertGreater(result['processing_time'], 0)
                            
                            # Check fusion details
                            self.assertIsInstance(result['fusion_details'], dict)
                            self.assertIn('fusion_method', result['fusion_details'])
                            self.assertIn('consensus_reached', result['fusion_details'])
                            self.assertIn('vote_breakdown', result['fusion_details'])
                            
                            # Verify detectors were called
                            mock_motion.assert_called_once()
                            mock_texture.assert_called_once()
                            mock_thermal.assert_called_once()
                            mock_depth.assert_called_once()
                            mock_iris.assert_called_once()
    
    def test_enhanced_liveness_detector_detect_liveness_spoof_detected(self):
        """Test liveness detection when spoof is detected"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock motion detector detects spoof
        mock_spoof_response = {
            'is_live': False,
            'confidence': 0.92,
            'liveness_score': 0.2,
            'method': 'motion_analysis',
            'spoof_type': 'photo_attack',
            'confidence_in_spoof': 0.85,
            'actions_detected': []  # No motion detected
        }
        
        # Other detectors might still think it's live
        mock_thermal_response = {
            'is_live': True,
            'confidence': 0.95,
            'liveness_score': 0.97,
            'method': 'thermal',
            'temperature_range': (36.5, 37.5),
            'thermal_consistency': 0.92
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_motion:
            with patch.object(detector.detectors['thermal'], 'detect') as mock_thermal:
                mock_motion.return_value = mock_spoof_response
                mock_thermal.return_value = mock_thermal_response
                
                # Detect liveness
                result = detector.detect_liveness(self.test_face_detection)
                
                self.assertIsNotNone(result)
                self.assertTrue(result['success'])
                self.assertFalse(result['is_live'])
                self.assertIn('spoof_detected', result)
                self.assertIn('spoof_type', result)
                self.assertIn('spoof_confidence', result)
                self.assertIn('spoof_details', result)
                
                # Check spoof information
                self.assertEqual(result['spoof_type'], 'photo_attack')
                self.assertEqual(result['spoof_confidence'], 0.85)
                self.assertIsInstance(result['spoof_details'], dict)
    
    def test_enhanced_liveness_detector_detect_liveness_no_consensus(self):
        """Test liveness detection when no consensus is reached"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mixed responses - some methods detect as live, some as not
        mock_motion_response = {
            'is_live': True,
            'confidence': 0.85,
            'liveness_score': 0.88,
            'method': 'motion_analysis'
        }
        
        mock_texture_response = {
            'is_live': False,
            'confidence': 0.82,
            'liveness_score': 0.25,
            'method': 'texture_analysis'
        }
        
        mock_thermal_response = {
            'is_live': True,
            'confidence': 0.90,
            'liveness_score': 0.92,
            'method': 'thermal'
        }
        
        mock_depth_response = {
            'is_live': False,
            'confidence': 0.78,
            'liveness_score': 0.30,
            'method': 'depth_map'
        }
        
        mock_iris_response = {
            'is_live': True,
            'confidence': 0.88,
            'liveness_score': 0.90,
            'method': 'iris_response'
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_motion:
            with patch.object(detector.detectors['texture_analysis'], 'detect') as mock_texture:
                with patch.object(detector.detectors['thermal'], 'detect') as mock_thermal:
                    with patch.object(detector.detectors['depth_map'], 'detect') as mock_depth:
                        with patch.object(detector.detectors['iris_response'], 'detect') as mock_iris:
                            mock_motion.return_value = mock_motion_response
                            mock_texture.return_value = mock_texture_response
                            mock_thermal.return_value = mock_thermal_response
                            mock_depth.return_value = mock_depth_response
                            mock_iris.return_value = mock_iris_response
                            
                            # Detect liveness
                            result = detector.detect_liveness(self.test_face_detection)
                            
                            self.assertIsNotNone(result)
                            self.assertIn('success', result)
                            self.assertIn('is_live', result)
                            self.assertIn('consensus_reached', result)
                            self.assertIn('method_votes', result)
                            self.assertIn('final_decision', result)
                            
                            # Check consensus
                            self.assertFalse(result['consensus_reached'])
                            self.assertIn('method_votes', result)
                            self.assertEqual(result['final_decision'], 'inconclusive')
    
    def test_enhanced_liveness_detector_detect_liveness_single_method(self):
        """Test liveness detection with single method enabled"""
        config = self.test_config.copy()
        config['detection_methods'] = ['motion_analysis']  # Only motion analysis
        
        detector = EnhancedLivenessDetector(config)
        
        mock_response = {
            'is_live': True,
            'confidence': 0.92,
            'liveness_score': 0.95,
            'method': 'motion_analysis'
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_detect:
            mock_detect.return_value = mock_response
            
            # Detect liveness
            result = detector.detect_liveness(self.test_face_detection)
            
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertTrue(result['is_live'])
            self.assertEqual(result['confidence'], 0.92)
            self.assertEqual(result['liveness_score'], 0.95)
            self.assertEqual(result['best_method'], 'motion_analysis')
    
    def test_enhanced_liveness_detector_batch_detection(self):
        """Test batch liveness detection"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock detector responses
        mock_response = {
            'is_live': True,
            'confidence': 0.92,
            'liveness_score': 0.95,
            'method': 'motion_analysis'
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_detect:
            mock_detect.return_value = mock_response
            
            # Batch detect liveness
            results = detector.detect_liveness_batch([self.test_face_detection] * 3)
            
            self.assertIsNotNone(results)
            self.assertEqual(len(results), 3)
            
            # Check each result
            for i, result in enumerate(results):
                self.assertIn('success', result)
                self.assertIn('is_live', result)
                self.assertIn('confidence', result)
                self.assertIn('processing_time', result)
                
                # Verify detectors were called for each face
                mock_detect.assert_any_call(self.test_face_detection)
    
    def test_enhanced_liveness_detector_method_scoring(self):
        """Test method scoring system"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock different method performances
        mock_high_conf_response = {
            'is_live': True,
            'confidence': 0.95,
            'liveness_score': 0.97,
            'method': 'thermal',
            'processing_time': 0.1
        }
        
        mock_low_conf_response = {
            'is_live': True,
            'confidence': 0.75,
            'liveness_score': 0.78,
            'method': 'motion_analysis',
            'processing_time': 0.2
        }
        
        with patch.object(detector.detectors['thermal'], 'detect') as mock_thermal:
            with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_motion:
                mock_thermal.return_value = mock_high_conf_response
                mock_motion.return_value = mock_low_conf_response
                
                # Detect liveness
                result = detector.detect_liveness(self.test_face_detection)
                
                # Check method scores
                self.assertIn('thermal', result['method_scores'])
                self.assertIn('motion_analysis', result['method_scores'])
                
                # Scores should be based on confidence and performance
                self.assertGreater(result['method_scores']['thermal'], 
                                 result['method_scores']['motion_analysis'])
                
                # Check best method
                self.assertEqual(result['best_method'], 'thermal')
    
    def test_enhanced_liveness_detector_method_details(self):
        """Test method details tracking"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock detailed responses
        mock_motion_response = {
            'is_live': True,
            'confidence': 0.92,
            'liveness_score': 0.95,
            'method': 'motion_analysis',
            'actions_detected': ['eye_blink', 'head_movement'],
            'motion_details': {
                'eye_movement': 0.15,
                'head_movement': 0.25,
                'frame_count': 5,
                'processing_time': 0.1
            }
        }
        
        mock_texture_response = {
            'is_live': True,
            'confidence': 0.88,
            'liveness_score': 0.90,
            'method': 'texture_analysis',
            'texture_score': 0.85,
            'spoof_detection': 'none',
            'gabor_features': 256,
            'processing_time': 0.08
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_motion:
            with patch.object(detector.detectors['texture_analysis'], 'detect') as mock_texture:
                mock_motion.return_value = mock_motion_response
                mock_texture.return_value = mock_texture_response
                
                # Detect liveness
                result = detector.detect_liveness(self.test_face_detection)
                
                # Check method details
                self.assertIsInstance(result['method_details'], dict)
                self.assertIn('motion_analysis', result['method_details'])
                self.assertIn('texture_analysis', result['method_details'])
                
                # Check each method detail
                motion_detail = result['method_details']['motion_analysis']
                self.assertEqual(motion_detail['detection_count'], 1)
                self.assertEqual(motion_detail['avg_confidence'], 0.92)
                self.assertEqual(motion_detail['avg_liveness_score'], 0.95)
                self.assertEqual(motion_detail['avg_processing_time'], 0.1)
                self.assertEqual(motion_detail['actions_detected'], ['eye_blink', 'head_movement'])
                
                texture_detail = result['method_details']['texture_analysis']
                self.assertEqual(texture_detail['detection_count'], 1)
                self.assertEqual(texture_detail['avg_confidence'], 0.88)
                self.assertEqual(texture_detail['avg_liveness_score'], 0.90)
                self.assertEqual(texture_detail['avg_processing_time'], 0.08)
                self.assertEqual(texture_detail['avg_texture_score'], 0.85)
    
    def test_enhanced_liveness_detector_sequence_detection(self):
        """Test sequence-based liveness detection"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock sequence response
        mock_sequence_response = {
            'is_live': True,
            'confidence': 0.94,
            'liveness_score': 0.96,
            'method': 'motion_analysis',
            'sequence_analysis': {
                'motion_consistency': 0.88,
                'temporal_pattern': 'natural',
                'frame_quality': 0.92,
                'frame_count': 5,
                'average_processing_time': 0.08
            }
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect_sequence') as mock_detect:
            mock_detect.return_value = mock_sequence_response
            
            # Detect liveness from sequence
            result = detector.detect_liveness_sequence(self.test_image_sequence)
            
            self.assertIsNotNone(result)
            self.assertIn('success', result)
            self.assertIn('is_live', result)
            self.assertIn('confidence', result)
            self.assertIn('sequence_analysis', result)
            self.assertIn('processing_time', result)
            
            # Check result values
            self.assertTrue(result['success'])
            self.assertTrue(result['is_live'])
            self.assertEqual(result['confidence'], 0.94)
            self.assertIsInstance(result['sequence_analysis'], dict)
            self.assertIsInstance(result['processing_time'], float)
            
            # Verify sequence detection was called
            mock_detect.assert_called_once()
    
    def test_motion_analysis_detector_initialization(self):
        """Test MotionAnalysisDetector initialization"""
        config = {
            'frame_window': 10,
            'motion_threshold': 0.05,
            'eye_blink_threshold': 0.3,
            'head_movement_threshold': 0.2,
            'required_actions': ['eye_blink', 'head_movement']
        }
        
        detector = MotionAnalysisDetector(config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.face_detector)
        self.assertIsNotNone(detector.motion_analyzer)
        self.assertIsNotNone(detector.eye_tracker)
        
        # Check configuration
        self.assertEqual(detector.config['frame_window'], 10)
        self.assertEqual(detector.config['motion_threshold'], 0.05)
        self.assertEqual(detector.config['eye_blink_threshold'], 0.3)
        self.assertEqual(detector.config['required_actions'], ['eye_blink', 'head_movement'])
    
    def test_motion_analysis_detector_detect(self):
        """Test motion-based liveness detection"""
        config = {
            'frame_window': 5,
            'motion_threshold': 0.05,
            'eye_blink_threshold': 0.3,
            'head_movement_threshold': 0.2
        }
        
        detector = MotionAnalysisDetector(config)
        
        # Mock face detection
        with patch.object(detector.face_detector, 'detect') as mock_detect:
            mock_detect.return_value = [self.test_face_detection]
            
            # Mock motion analysis
            with patch.object(detector.motion_analyzer, 'analyze_motion') as mock_analyze:
                mock_analyze.return_value = {
                    'eye_movement': 0.15,
                    'head_movement': 0.25,
                    'motion_consistency': 0.85
                }
                
                # Mock eye tracking
                with patch.object(detector.eye_tracker, 'track_eyes') as mock_track:
                    mock_track.return_value = {
                        'left_eye_open': 1.0,
                        'right_eye_open': 1.0,
                        'eye_blink_detected': False,
                        'eye_movement': 0.15
                    }
                    
                    # Detect liveness
                    result = detector.detect(self.test_image_sequence)
                    
                    self.assertIsNotNone(result)
                    self.assertTrue(result['is_live'])
                    self.assertEqual(result['confidence'], 0.92)
                    self.assertEqual(result['liveness_score'], 0.95)
                    self.assertEqual(result['method'], 'motion_analysis')
                    self.assertIn('actions_detected', result)
                    self.assertIn('motion_details', result)
                    
                    # Verify calls were made
                    mock_detect.assert_called()
                    mock_analyze.assert_called()
                    mock_track.assert_called()
    
    def test_texture_analysis_detector_initialization(self):
        """Test TextureAnalysisDetector initialization"""
        config = {
            'gabor_kernels': 8,
            'texture_threshold': 0.7,
            'spoof_threshold': 0.8
        }
        
        detector = TextureAnalysisDetector(config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.texture_analyzer)
        self.assertIsNotNone(detector.gabor_extractor)
        
        # Check configuration
        self.assertEqual(detector.config['gabor_kernels'], 8)
        self.assertEqual(detector.config['texture_threshold'], 0.7)
        self.assertEqual(detector.config['spoof_threshold'], 0.8)
    
    def test_thermal_detector_initialization(self):
        """Test ThermalDetector initialization"""
        config = {
            'thermal_threshold': 35.0,
            'max_temp_diff': 5.0,
            'thermal_map_resolution': 32
        }
        
        detector = ThermalDetector(config)
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
        self.assertIsNotNone(detector.thermal_analyzer)
        
        # Check configuration
        self.assertEqual(detector.config['thermal_threshold'], 35.0)
        self.assertEqual(detector.config['max_temp_diff'], 5.0)
        self.assertEqual(detector.config['thermal_map_resolution'], 32)
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock detector responses
        mock_response = {
            'is_live': True,
            'confidence': 0.92,
            'liveness_score': 0.95,
            'method': 'motion_analysis'
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_detect:
            mock_detect.return_value = mock_response
            
            # Run benchmark
            benchmark_results = detector.benchmark_liveness_detection(self.test_face_detection)
            
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
            self.assertIn('motion_analysis', benchmark_results['method_performance'])
    
    def test_error_handling(self):
        """Test error handling"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Test detector error
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_detect:
            mock_detect.side_effect = Exception("Motion analysis failed")
            
            # Detect liveness with error
            result = detector.detect_liveness(self.test_face_detection)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Motion analysis failed', result['error'])
        
        # Test sequence processing error
        with patch.object(detector.detectors['motion_analysis'], 'detect_sequence') as mock_detect:
            mock_detect.side_effect = Exception("Sequence processing failed")
            
            # Detect liveness from sequence with error
            result = detector.detect_liveness_sequence(self.test_image_sequence)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Sequence processing failed', result['error'])
    
    def test_cache_functionality(self):
        """Test liveness detection caching"""
        detector = EnhancedLivenessDetector(self.test_config)
        
        # Mock detector response
        mock_response = {
            'is_live': True,
            'confidence': 0.92,
            'liveness_score': 0.95,
            'method': 'motion_analysis'
        }
        
        with patch.object(detector.detectors['motion_analysis'], 'detect') as mock_detect:
            mock_detect.return_value = mock_response
            
            # First detection (should hit detector)
            result1 = detector.detect_liveness(self.test_face_detection)
            
            # Second detection (should use cache)
            result2 = detector.detect_liveness(self.test_face_detection)
            
            # Verify cache was used
            self.assertTrue(hasattr(detector, 'results_cache'))
            self.assertTrue(len(detector.results_cache) > 0)
            
            # Results should be the same
            self.assertEqual(result1['is_live'], result2['is_live'])
            self.assertEqual(result1['confidence'], result2['confidence'])
            
            # Detector should be called only once (second call uses cache)
            mock_detect.assert_called_once()

if __name__ == '__main__':
    unittest.main()