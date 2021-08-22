"""Test enhanced liveness detection functionality"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced.liveness.detectors import *
from enhanced.liveness.analysis import MotionAnalysis, BlinkAnalysis

class TestEnhancedLiveness(unittest.TestCase):
    """Test cases for Enhanced Liveness Detection Systems"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test face images
        self.test_face = np.zeros((224, 224, 3), dtype=np.uint8)
        self.test_face[50:174, 50:174] = 255  # White rectangle as face
        
        # Create test video frames
        self.test_frames = []
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[100:200, 200:300] = 255  # Moving face
            self.test_frames.append(frame)
        
        # Create test face landmarks
        self.test_landmarks = {
            'left_eye': (110, 110),
            'right_eye': (190, 110),
            'nose': (150, 150),
            'mouth_left': (130, 190),
            'mouth_right': (170, 190)
        }
    
    def test_motion_analysis_initialization(self):
        """Test MotionAnalysis initialization"""
        motion_analyzer = MotionAnalysis()
        
        self.assertIsNotNone(motion_analyzer)
        self.assertEqual(motion_analyzer.threshold, 0.3)
        self.assertEqual(motion_analyzer.min_motion_frames, 3)
        self.assertEqual(motion_analyzer.max_consecutive_static, 5)
    
    def test_motion_analysis_calculation(self):
        """Test motion analysis calculation"""
        motion_analyzer = MotionAnalysis()
        
        # Test with static frames (no motion)
        static_frames = [self.test_face.copy() for _ in range(5)]
        motion_result = motion_analyzer.calculate_motion(static_frames)
        
        self.assertIsInstance(motion_result, dict)
        self.assertIn('total_motion', motion_result)
        self.assertIn('motion_direction', motion_result)
        self.assertIn('motion_quality', motion_result)
        
        # Static frames should have minimal motion
        self.assertLess(motion_result['total_motion'], 0.1)
        self.assertEqual(motion_result['motion_quality'], 'low')
        
        # Test with motion frames
        motion_frames = []
        for i in range(5):
            frame = self.test_face.copy()
            # Add movement
            if i < 3:
                frame[50:174, 50:174] = 200  # Different intensity
            motion_frames.append(frame)
        
        motion_result = motion_analyzer.calculate_motion(motion_frames)
        
        # Motion frames should have higher motion
        self.assertGreater(motion_result['total_motion'], 0.1)
        self.assertIn(motion_result['motion_quality'], ['medium', 'high'])
    
    def test_blink_analysis_initialization(self):
        """Test BlinkAnalysis initialization"""
        blink_analyzer = BlinkAnalysis()
        
        self.assertIsNotNone(blink_analyzer)
        self.assertEqual(blink_analyzer.eye_open_threshold, 0.2)
        self.assertEqual(blink_analyzer.blink_duration_frames, 3)
        self.assertEqual(blink_analyzer.min_blinks_required, 1)
    
    def test_blink_analysis_eye_aspect_ratio(self):
        """Test eye aspect ratio calculation"""
        blink_analyzer = BlinkAnalysis()
        
        # Create test eye landmarks
        left_eye_points = [(110, 100), (120, 105), (130, 100), (125, 110), (115, 110)]
        right_eye_points = [(170, 100), (180, 105), (190, 100), (185, 110), (175, 110)]
        
        # Calculate eye aspect ratio for open eye
        left_ear = blink_analyzer.calculate_eye_aspect_ratio(left_eye_points)
        right_ear = blink_analyzer.calculate_eye_aspect_ratio(right_eye_points)
        
        # Eye aspect ratio should be reasonable for open eyes
        self.assertGreater(left_ear, 0.2)
        self.assertGreater(right_ear, 0.2)
        
        # Create closed eye landmarks (vertically aligned)
        closed_left_eye = [(110, 105), (120, 105), (130, 105), (125, 105), (115, 105)]
        closed_right_eye = [(170, 105), (180, 105), (190, 105), (185, 105), (175, 105)]
        
        # Calculate eye aspect ratio for closed eyes
        closed_left_ear = blink_analyzer.calculate_eye_aspect_ratio(closed_left_eye)
        closed_right_ear = blink_analyzer.calculate_eye_aspect_ratio(closed_right_eye)
        
        # Closed eyes should have lower aspect ratio
        self.assertLess(closed_left_ear, 0.1)
        self.assertLess(closed_right_ear, 0.1)
    
    def test_blink_detection(self):
        """Test blink detection"""
        blink_analyzer = BlinkAnalysis()
        
        # Create test blink sequence
        blink_sequence = []
        for i in range(10):
            if 3 <= i <= 5:  # Blink frames
                # Simulate closed eye
                ear = 0.05
            else:  # Open eye
                ear = 0.3
            blink_sequence.append(ear)
        
        # Detect blinks
        blink_result = blink_analyzer.detect_blinks(blink_sequence)
        
        self.assertIsInstance(blink_result, dict)
        self.assertIn('blink_count', blink_result)
        self.assertIn('blink_quality', blink_result)
        self.assertIn('blink_consistency', blink_result)
        
        # Should detect at least one blink
        self.assertGreater(blink_result['blink_count'], 0)
    
    def test_static_liveness_detector_initialization(self):
        """Test StaticLivenessDetector initialization"""
        detector = StaticLivenessDetector()
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.motion_analyzer)
        self.assertIsNotNone(detector.blink_analyzer)
        self.assertEqual(detector.confidence_threshold, 0.7)
    
    def test_static_liveness_detection(self):
        """Test static liveness detection"""
        detector = StaticLivenessDetector()
        
        # Test with real face image (should pass basic checks)
        result = detector.detect(self.test_face, landmarks=self.test_landmarks)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        self.assertIn('analysis', result)
        self.assertIn('timestamp', result)
        
        # Should be liveness (though confidence may be low due to simplicity)
        self.assertIsInstance(result['is_liveness'], bool)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_static_liveness_detection_no_landmarks(self):
        """Test static liveness detection without landmarks"""
        detector = StaticLivenessDetector()
        
        # Test with no landmarks
        result = detector.detect(self.test_face)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        self.assertIn('analysis', result)
        
        # Should have lower confidence without landmarks
        self.assertLess(result['confidence'], 0.5)
    
    def test_static_liveness_detection_no_face(self):
        """Test static liveness detection with no face"""
        detector = StaticLivenessDetector()
        
        # Test with blank image
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(blank_image)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        
        # Should not be liveness
        self.assertFalse(result['is_liveness'])
        self.assertLess(result['confidence'], 0.3)
    
    def test_video_liveness_detector_initialization(self):
        """Test VideoLivenessDetector initialization"""
        detector = VideoLivenessDetector()
        
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.motion_analyzer)
        self.assertIsNotNone(detector.blink_analyzer)
        self.assertEqual(detector.min_frames, 5)
        self.assertEqual(detector.confidence_threshold, 0.8)
    
    def test_video_liveness_detection(self):
        """Test video liveness detection"""
        detector = VideoLivenessDetector()
        
        # Test with motion sequence
        result = detector.detect_video(self.test_frames)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        self.assertIn('analysis', result)
        self.assertIn('motion_analysis', result)
        self.assertIn('blink_analysis', result)
        
        # Should have basic structure
        self.assertIsInstance(result['is_liveness'], bool)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_video_liveness_detection_insufficient_frames(self):
        """Test video liveness detection with insufficient frames"""
        detector = VideoLivenessDetector(min_frames=10)
        
        # Test with fewer frames than required
        insufficient_frames = self.test_frames[:5]
        result = detector.detect_video(insufficient_frames)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        self.assertIn('error', result)
        
        # Should have error
        self.assertIn('error', result)
        self.assertLess(result['confidence'], 0.3)
    
    def test_3d_structure_analysis_initialization(self):
        """Test 3DStructureAnalyzer initialization"""
        analyzer = MotionAnalysis()  # Reuse motion analyzer for 3D structure
        self.assertIsNotNone(analyzer)
    
    def test_3d_structure_analysis(self):
        """Test 3D structure analysis"""
        analyzer = MotionAnalysis()
        
        # Create test depth map (simulated)
        depth_map = np.random.rand(224, 224) * 10  # Random depth values
        
        # Analyze 3D structure
        structure_result = analyzer.analyze_3d_structure(depth_map)
        
        self.assertIsInstance(structure_result, dict)
        self.assertIn('depth_variance', structure_result)
        self.assertIn('surface_normality', structure_result)
        self.assertIn('is_3d_like', structure_result)
        
        # Should have reasonable values
        self.assertGreater(structure_result['depth_variance'], 0)
        self.assertGreaterEqual(structure_result['surface_normality'], 0)
        self.assertLessEqual(structure_result['surface_normality'], 1)
        self.assertIsInstance(structure_result['is_3d_like'], bool)
    
    def test_texture_analysis_initialization(self):
        """Test texture analysis initialization"""
        from enhanced.liveness.analysis import TextureAnalysis
        
        texture_analyzer = TextureAnalysis()
        self.assertIsNotNone(texture_analyzer)
    
    def test_texture_analysis(self):
        """Test texture analysis"""
        from enhanced.liveness.analysis import TextureAnalysis
        
        texture_analyzer = TextureAnalysis()
        
        # Test with real face image
        texture_result = texture_analyzer.analyze_texture(self.test_face)
        
        self.assertIsInstance(texture_result, dict)
        self.assertIn('texture_score', texture_result)
        self.assertIn('is_natural_texture', texture_result)
        self.assertIn('edge_count', texture_result)
        
        # Should have reasonable values
        self.assertGreaterEqual(texture_result['texture_score'], 0)
        self.assertLessEqual(texture_result['texture_score'], 1)
        self.assertIsInstance(texture_result['is_natural_texture'], bool)
        self.assertGreater(texture_result['edge_count'], 0)
    
    def test_liveness_detector_error_handling(self):
        """Test liveness detector error handling"""
        detector = StaticLivenessDetector()
        
        # Test with invalid image
        invalid_image = "not an image"
        result = detector.detect(invalid_image)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        self.assertIn('error', result)
        
        # Should have error and low confidence
        self.assertIn('error', result)
        self.assertLess(result['confidence'], 0.1)
        self.assertFalse(result['is_liveness'])
        
        # Test with None
        result = detector.detect(None)
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertLess(result['confidence'], 0.1)
    
    def test_liveness_detector_performance(self):
        """Test liveness detector performance"""
        import time
        
        # Test static detector performance
        detector = StaticLivenessDetector()
        
        start_time = time.time()
        for _ in range(10):
            result = detector.detect(self.test_face)
        end_time = time.time()
        
        # Should be fast (less than 5 seconds for 10 detections)
        self.assertLess(end_time - start_time, 5.0)
        
        # Test video detector performance
        video_detector = VideoLivenessDetector()
        
        start_time = time.time()
        result = video_detector.detect_video(self.test_frames)
        end_time = time.time()
        
        # Should be reasonable (less than 10 seconds for 10 frames)
        self.assertLess(end_time - start_time, 10.0)
    
    def test_liveness_detector_memory_usage(self):
        """Test liveness detector memory usage"""
        import sys
        
        # Create detector
        detector = StaticLivenessDetector()
        
        # Track memory before
        initial_size = sys.getsizeof(detector)
        
        # Perform multiple detections
        for _ in range(100):
            result = detector.detect(self.test_face)
        
        # Track memory after
        final_size = sys.getsizeof(detector)
        
        # Memory should not grow significantly
        self.assertLess(final_size - initial_size, 1000)  # Less than 1KB growth
    
    def test_combined_liveness_analysis(self):
        """Test combined liveness analysis"""
        detector = StaticLivenessDetector()
        
        # Perform comprehensive analysis
        result = detector.detect(self.test_face, landmarks=self.test_landmarks)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('is_liveness', result)
        self.assertIn('confidence', result)
        self.assertIn('analysis', result)
        self.assertIn('timestamp', result)
        
        # Check analysis components
        analysis = result['analysis']
        self.assertIn('motion_analysis', analysis)
        self.assertIn('blink_analysis', analysis)
        self.assertIn('texture_analysis', analysis)
        self.assertIn('structure_analysis', analysis)
        
        # All should be valid
        self.assertIsInstance(analysis['motion_analysis'], dict)
        self.assertIsInstance(analysis['blink_analysis'], dict)
        self.assertIsInstance(analysis['texture_analysis'], dict)
        self.assertIsInstance(analysis['structure_analysis'], dict)
    
    def test_liveness_detector_threshold_adjustment(self):
        """Test liveness detector threshold adjustment"""
        detector = StaticLivenessDetector()
        detector.confidence_threshold = 0.9  # High threshold
        
        # Test with basic face
        result = detector.detect(self.test_face)
        
        # Should have lower confidence due to high threshold
        self.assertLess(result['confidence'], 0.9)
        self.assertFalse(result['is_liveness'])
        
        # Lower threshold
        detector.confidence_threshold = 0.3
        
        result = detector.detect(self.test_face)
        
        # Should have higher confidence
        self.assertGreater(result['confidence'], 0.3)
        self.assertTrue(result['is_liveness'])
    
    def test_liveness_detector_result_serialization(self):
        """Test liveness detector result serialization"""
        import json
        
        detector = StaticLivenessDetector()
        result = detector.detect(self.test_face, landmarks=self.test_landmarks)
        
        # Try to serialize to JSON
        try:
            json_str = json.dumps(result, default=str)
            self.assertIsInstance(json_str, str)
            
            # Deserialize back
            deserialized = json.loads(json_str)
            self.assertIsInstance(deserialized, dict)
            
            # Check key components
            self.assertIn('is_liveness', deserialized)
            self.assertIn('confidence', deserialized)
            self.assertIn('analysis', deserialized)
            
        except Exception as e:
            self.fail(f"Result serialization failed: {e}")

if __name__ == '__main__':
    unittest.main()