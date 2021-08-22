"""Test enhanced face recognition functionality"""

import unittest
import os
import sys
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_recognition.recognizers.enhanced_recognizer import EnhancedRecognizer, VGGFaceRecognizer, LBPHRecognizer

class TestEnhancedRecognizer(unittest.TestCase):
    """Test cases for Enhanced Face Recognizer"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test face data
        self.test_face_data = {
            'person_id': 'person_001',
            'name': 'John Doe',
            'faces': [
                {
                    'image_id': 'face_001',
                    'encoding': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'timestamp': datetime.now(),
                    'location': 'entrance'
                }
            ],
            'metadata': {
                'department': 'Engineering',
                'access_level': 'employee',
                'registration_date': datetime.now(),
                'last_seen': datetime.now()
            }
        }
        
        # Test face detection result
        self.test_face_detection = {
            'face_id': 'detect_001',
            'bbox': [30, 30, 70, 70],
            'confidence': 0.95,
            'landmarks': {'left_eye': (40, 40), 'right_eye': (60, 40)},
            'timestamp': datetime.now()
        }
        
        # Test configuration
        self.test_config = {
            'models': {
                'vgg_face': {
                    'model_path': 'models/vgg_face.h5',
                    'input_shape': (224, 224),
                    'preprocessing': 'vgg'
                },
                'lbph': {
                    'radius': 1,
                    'neighbors': 8,
                    'grid_x': 8,
                    'grid_y': 8
                }
            },
            'confidence_threshold': 0.85,
            'enable_gpu': True,
            'batch_size': 32,
            'max_workers': 4,
            'feature_extraction': {
                'enable_pca': True,
                'pca_components': 128,
                'enable_l2_normalization': True
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_recognizer_initialization(self):
        """Test EnhancedRecognizer initialization"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        self.assertIsNotNone(recognizer)
        self.assertIsNotNone(recognizer.config)
        self.assertIsNotNone(recognizer.recognizers)
        self.assertIsNotNone(recognizer.feature_extractor)
        self.assertIsNotNone(recognizer.results_cache)
        self.assertEqual(len(recognizer.recognizers), 2)  # VGG and LBPH
        
        # Check configurations
        self.assertEqual(recognizer.config['confidence_threshold'], 0.85)
        self.assertTrue(recognizer.config['enable_gpu'])
        self.assertEqual(recognizer.config['batch_size'], 32)
    
    def test_enhanced_recognizer_initialization_default_config(self):
        """Test EnhancedRecognizer initialization with default config"""
        recognizer = EnhancedRecognizer()
        
        self.assertIsNotNone(recognizer)
        self.assertIsNotNone(recognizer.config)
        self.assertEqual(len(recognizer.recognizers), 2)  # Default recognizers
        
        # Check default values
        self.assertEqual(recognizer.config['confidence_threshold'], 0.8)
        self.assertFalse(recognizer.config['enable_gpu'])  # GPU disabled by default
        self.assertEqual(recognizer.config['batch_size'], 16)
    
    def test_enhanced_recognizer_load_recognizer(self):
        """Test recognizer loading"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Test loading individual recognizers
        vgg_recognizer = recognizer._load_recognizer('vgg_face')
        self.assertIsNotNone(vgg_recognizer)
        self.assertIsInstance(vgg_recognizer, VGGFaceRecognizer)
        
        lbph_recognizer = recognizer._load_recognizer('lbph')
        self.assertIsNotNone(lbph_recognizer)
        self.assertIsInstance(lbph_recognizer, LBPHRecognizer)
    
    def test_enhanced_recognizer_register_person(self):
        """Test person registration"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock database
        with patch.object(recognizer.database, 'save_person') as mock_save:
            mock_save.return_value = True
            
            # Register person
            result = recognizer.register_person(self.test_face_data)
            
            self.assertIsNotNone(result)
            self.assertIn('success', result)
            self.assertIn('person_id', result)
            self.assertIn('encoding_count', result)
            self.assertIn('registration_time', result)
            
            # Check result values
            self.assertTrue(result['success'])
            self.assertEqual(result['person_id'], 'person_001')
            self.assertEqual(result['encoding_count'], 1)
            self.assertIsInstance(result['registration_time'], datetime)
            
            # Verify database was called
            mock_save.assert_called_once_with(self.test_face_data)
    
    def test_enhanced_recognizer_register_person_multiple_faces(self):
        """Test person registration with multiple faces"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Create person with multiple faces
        person_data = self.test_face_data.copy()
        person_data['faces'] = [
            {
                'image_id': f'face_{i}',
                'encoding': [0.1 * (i + 1) + j * 0.1 for j in range(10)],
                'timestamp': datetime.now(),
                'location': 'entrance'
            }
            for i in range(3)  # 3 faces
        ]
        
        # Mock database
        with patch.object(recognizer.database, 'save_person') as mock_save:
            mock_save.return_value = True
            
            # Register person
            result = recognizer.register_person(person_data)
            
            self.assertTrue(result['success'])
            self.assertEqual(result['person_id'], 'person_001')
            self.assertEqual(result['encoding_count'], 3)
    
    def test_enhanced_recognizer_recognize_face(self):
        """Test face recognition"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock recognizer responses
            mock_vgg_response = {
                'person_id': 'person_001',
                'confidence': 0.92,
                'match': True,
                'distance': 0.15,
                'method': 'vgg_face'
            }
            
            mock_lbph_response = {
                'person_id': 'person_001',
                'confidence': 0.88,
                'match': True,
                'distance': 0.25,
                'method': 'lbph'
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_vgg_response
                    mock_lbph.return_value = mock_lbph_response
                    
                    # Recognize face
                    result = recognizer.recognize_face(self.test_face_detection)
                    
                    self.assertIsNotNone(result)
                    self.assertIn('success', result)
                    self.assertIn('person_id', result)
                    self.assertIn('name', result)
                    self.assertIn('confidence', result)
                    self.assertIn('match', result)
                    self.assertIn('method_scores', result)
                    self.assertIn('best_method', result)
                    self.assertIn('processing_time', result)
                    self.assertIn('method_details', result)
                    
                    # Check result values
                    self.assertTrue(result['success'])
                    self.assertEqual(result['person_id'], 'person_001')
                    self.assertEqual(result['name'], 'John Doe')
                    self.assertEqual(result['confidence'], 0.92)  # VGG has higher confidence
                    self.assertTrue(result['match'])
                    
                    # Check method scores
                    self.assertIn('vgg_face', result['method_scores'])
                    self.assertIn('lbph', result['method_scores'])
                    
                    # Check best method
                    self.assertEqual(result['best_method'], 'vgg_face')
                    
                    # Check processing time
                    self.assertIsInstance(result['processing_time'], float)
                    self.assertGreater(result['processing_time'], 0)
                    
                    # Check method details
                    self.assertIsInstance(result['method_details'], dict)
                    self.assertIn('vgg_face', result['method_details'])
                    self.assertIn('lbph', result['method_details'])
                    
                    # Verify recognizers were called
                    mock_vgg.assert_called_once()
                    mock_lbph.assert_called_once()
    
    def test_enhanced_recognizer_recognize_face_no_match(self):
        """Test face recognition when no match is found"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval (returns None for unknown person)
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = None
            
            # Mock recognizer responses (no match)
            mock_vgg_response = {
                'person_id': None,
                'confidence': 0.3,
                'match': False,
                'distance': 0.8,
                'method': 'vgg_face'
            }
            
            mock_lbph_response = {
                'person_id': None,
                'confidence': 0.25,
                'match': False,
                'distance': 0.9,
                'method': 'lbph'
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_vgg_response
                    mock_lbph.return_value = mock_lbph_response
                    
                    # Recognize face
                    result = recognizer.recognize_face(self.test_face_detection)
                    
                    self.assertIsNotNone(result)
                    self.assertFalse(result['success'])
                    self.assertEqual(result['person_id'], None)
                    self.assertEqual(result['confidence'], 0.3)  # Best among no matches
                    self.assertFalse(result['match'])
                    self.assertEqual(result['best_method'], None)
    
    def test_enhanced_recognizer_recognize_face_low_confidence(self):
        """Test face recognition with low confidence results"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Set low confidence threshold for this test
        recognizer.config['confidence_threshold'] = 0.9
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock low confidence responses
            mock_vgg_response = {
                'person_id': 'person_001',
                'confidence': 0.85,  # Below threshold
                'match': False,  # Even though ID matches, confidence is low
                'distance': 0.35,
                'method': 'vgg_face'
            }
            
            mock_lbph_response = {
                'person_id': 'person_001',
                'confidence': 0.82,  # Below threshold
                'match': False,
                'distance': 0.45,
                'method': 'lbph'
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_vgg_response
                    mock_lbph.return_value = mock_lbph_response
                    
                    # Recognize face
                    result = recognizer.recognize_face(self.test_face_detection)
                    
                    self.assertIsNotNone(result)
                    self.assertFalse(result['success'])
                    self.assertEqual(result['match'], False)
                    # Confidence should be the best result but below threshold
                    self.assertEqual(result['confidence'], 0.85)
    
    def test_enhanced_recognizer_update_person_encoding(self):
        """Test updating person encoding"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock database
        with patch.object(recognizer.database, 'update_person_encoding') as mock_update:
            mock_update.return_value = True
            
            # Update encoding
            result = recognizer.update_person_encoding(
                person_id='person_001',
                new_encoding=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                metadata={'location': 'reception', 'timestamp': datetime.now()}
            )
            
            self.assertIsNotNone(result)
            self.assertIn('success', result)
            self.assertIn('person_id', result)
            self.assertIn('encoding_id', result)
            self.assertIn('update_time', result)
            
            # Check result values
            self.assertTrue(result['success'])
            self.assertEqual(result['person_id'], 'person_001')
            self.assertIsInstance(result['encoding_id'], str)
            self.assertIsInstance(result['update_time'], datetime)
            
            # Verify database was called
            mock_update.assert_called_once()
    
    def test_enhanced_recognizer_batch_recognition(self):
        """Test batch face recognition"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock recognizer responses
            mock_vgg_response = {
                'person_id': 'person_001',
                'confidence': 0.92,
                'match': True,
                'distance': 0.15,
                'method': 'vgg_face'
            }
            
            mock_lbph_response = {
                'person_id': 'person_001',
                'confidence': 0.88,
                'match': True,
                'distance': 0.25,
                'method': 'lbph'
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_vgg_response
                    mock_lbph.return_value = mock_lbph_response
                    
                    # Batch recognize faces
                    results = recognizer.recognize_faces_batch([self.test_face_detection] * 3)
                    
                    self.assertIsNotNone(results)
                    self.assertEqual(len(results), 3)
                    
                    # Check each result
                    for i, result in enumerate(results):
                        self.assertIn('success', result)
                        self.assertIn('person_id', result)
                        self.assertIn('confidence', result)
                        self.assertIn('processing_time', result)
                        
                        # Verify recognizers were called for each face
                        mock_vgg.assert_any_call(self.test_face_detection)
                        mock_lbph.assert_any_call(self.test_face_detection)
    
    def test_enhanced_recognizer_method_scoring(self):
        """Test method scoring system"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock different method performances
            mock_vgg_response = {
                'person_id': 'person_001',
                'confidence': 0.92,
                'match': True,
                'distance': 0.15,
                'method': 'vgg_face',
                'processing_time': 0.1
            }
            
            mock_lbph_response = {
                'person_id': 'person_001',
                'confidence': 0.88,
                'match': True,
                'distance': 0.25,
                'method': 'lbph',
                'processing_time': 0.05
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_vgg_response
                    mock_lbph.return_value = mock_lbph_response
                    
                    # Recognize face
                    result = recognizer.recognize_face(self.test_face_detection)
                    
                    # Check method scores
                    self.assertIn('vgg_face', result['method_scores'])
                    self.assertIn('lbph', result['method_scores'])
                    
                    # Scores should be based on confidence and processing time
                    self.assertGreater(result['method_scores']['vgg_face'], 
                                     result['method_scores']['lbph'])
                    
                    # Check best method
                    self.assertEqual(result['best_method'], 'vgg_face')
    
    def test_enhanced_recognizer_method_details(self):
        """Test method details tracking"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock detailed responses
            mock_vgg_response = {
                'person_id': 'person_001',
                'confidence': 0.92,
                'match': True,
                'distance': 0.15,
                'method': 'vgg_face',
                'processing_time': 0.1,
                'feature_count': 2048
            }
            
            mock_lbph_response = {
                'person_id': 'person_001',
                'confidence': 0.88,
                'match': True,
                'distance': 0.25,
                'method': 'lbph',
                'processing_time': 0.05,
                'feature_count': 576
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_vgg_response
                    mock_lbph.return_value = mock_lbph_response
                    
                    # Recognize face
                    result = recognizer.recognize_face(self.test_face_detection)
                    
                    # Check method details
                    self.assertIsInstance(result['method_details'], dict)
                    self.assertIn('vgg_face', result['method_details'])
                    self.assertIn('lbph', result['method_details'])
                    
                    # Check each method detail
                    vgg_detail = result['method_details']['vgg_face']
                    self.assertEqual(vgg_detail['recognition_count'], 1)
                    self.assertEqual(vgg_detail['avg_confidence'], 0.92)
                    self.assertEqual(vgg_detail['avg_distance'], 0.15)
                    self.assertEqual(vgg_detail['processing_time'], 0.1)
                    self.assertEqual(vgg_detail['feature_count'], 2048)
                    
                    lbph_detail = result['method_details']['lbph']
                    self.assertEqual(lbph_detail['recognition_count'], 1)
                    self.assertEqual(lbph_detail['avg_confidence'], 0.88)
                    self.assertEqual(lbph_detail['avg_distance'], 0.25)
                    self.assertEqual(lbph_detail['processing_time'], 0.05)
                    self.assertEqual(lbph_detail['feature_count'], 576)
    
    def test_enhanced_recognizer_feature_extraction(self):
        """Test feature extraction functionality"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock raw face image
        raw_encoding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Test feature extraction with PCA
        extracted_features = recognizer._extract_features(raw_encoding)
        
        self.assertIsNotNone(extracted_features)
        self.assertIsInstance(extracted_features, list)
        self.assertEqual(len(extracted_features), 128)  # PCA components
        
        # Test L2 normalization
        normalized_features = recognizer._apply_l2_normalization(extracted_features)
        
        self.assertIsNotNone(normalized_features)
        self.assertIsInstance(normalized_features, list)
        self.assertEqual(len(normalized_features), 128)
        
        # Check that features are normalized (L2 norm should be 1)
        l2_norm = np.linalg.norm(normalized_features)
        self.assertAlmostEqual(l2_norm, 1.0, places=5)
    
    def test_enhanced_recognizer_cache_functionality(self):
        """Test recognition caching"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock recognizer responses
            mock_response = {
                'person_id': 'person_001',
                'confidence': 0.92,
                'match': True,
                'distance': 0.15,
                'method': 'vgg_face'
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                with patch.object(recognizer.recognizers['lbph'], 'recognize') as mock_lbph:
                    mock_vgg.return_value = mock_response
                    mock_lbph.return_value = mock_response
                    
                    # First recognition (should hit recognizers)
                    result1 = recognizer.recognize_face(self.test_face_detection)
                    
                    # Second recognition (should use cache)
                    result2 = recognizer.recognize_face(self.test_face_detection)
                    
                    # Verify cache was used
                    self.assertTrue(hasattr(recognizer, 'results_cache'))
                    self.assertTrue(len(recognizer.results_cache) > 0)
                    
                    # Results should be the same
                    self.assertEqual(result1['person_id'], result2['person_id'])
                    self.assertEqual(result1['confidence'], result2['confidence'])
                    
                    # Recognizers should be called only once (second call uses cache)
                    mock_vgg.assert_called_once()
    
    def test_vgg_face_recognizer_initialization(self):
        """Test VGGFaceRecognizer initialization"""
        vgg_config = {
            'model_path': 'models/vgg_face.h5',
            'input_shape': (224, 224),
            'preprocessing': 'vgg'
        }
        
        recognizer = VGGFaceRecognizer(vgg_config)
        
        self.assertIsNotNone(recognizer)
        self.assertIsNotNone(recognizer.config)
        self.assertIsNotNone(recognizer.model)
        self.assertIsNotNone(recognizer.preprocessor)
        
        # Check configuration
        self.assertEqual(recognizer.config['model_path'], 'models/vgg_face.h5')
        self.assertEqual(recognizer.config['input_shape'], (224, 224))
        self.assertEqual(recognizer.config['preprocessing'], 'vgg')
    
    def test_vgg_face_recognizer_recognize(self):
        """Test VGG face recognition"""
        vgg_config = {
            'model_path': 'models/vgg_face.h5',
            'input_shape': (224, 224),
            'preprocessing': 'vgg'
        }
        
        recognizer = VGGFaceRecognizer(vgg_config)
        
        # Mock model prediction
        mock_encoding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_distances = np.array([0.15, 0.25, 0.35])
        
        with patch.object(recognizer.model, 'predict') as mock_predict:
            with patch.object(recognizer.database, 'find_best_match') as mock_find:
                mock_predict.return_value = mock_encoding
                mock_find.return_value = ('person_001', 0.15)
                
                # Recognize face
                result = recognizer.recognize(self.test_face_detection)
                
                self.assertIsNotNone(result)
                self.assertEqual(result['person_id'], 'person_001')
                self.assertEqual(result['confidence'], 0.85)  # Based on distance threshold
                self.assertEqual(result['match'], True)
                self.assertEqual(result['distance'], 0.15)
                self.assertEqual(result['method'], 'vgg_face')
                
                # Verify model and database calls
                mock_predict.assert_called_once()
                mock_find.assert_called_once()
    
    def test_lbph_recognizer_initialization(self):
        """Test LBPHRecognizer initialization"""
        lbph_config = {
            'radius': 1,
            'neighbors': 8,
            'grid_x': 8,
            'grid_y': 8
        }
        
        recognizer = LBPHRecognizer(lbph_config)
        
        self.assertIsNotNone(recognizer)
        self.assertIsNotNone(recognizer.config)
        self.assertIsNotNone(recognizer.recognizer)
        
        # Check configuration
        self.assertEqual(recognizer.config['radius'], 1)
        self.assertEqual(recognizer.config['neighbors'], 8)
        self.assertEqual(recognizer.config['grid_x'], 8)
        self.assertEqual(recognizer.config['grid_y'], 8)
    
    def test_lbph_recognizer_recognize(self):
        """Test LBPH face recognition"""
        lbph_config = {
            'radius': 1,
            'neighbors': 8,
            'grid_x': 8,
            'grid_y': 8
        }
        
        recognizer = LBPHRecognizer(lbph_config)
        
        # Mock face image extraction
        mock_face_image = np.random.rand(50, 50)
        
        # Mock recognizer training and prediction
        with patch.object(recognizer.recognizer, 'predict') as mock_predict:
            mock_predict.return_value = (0, 0.75)  # person_id, confidence
            
            # Recognize face
            result = recognizer.recognize(self.test_face_detection)
            
            self.assertIsNotNone(result)
            self.assertEqual(result['person_id'], 0)
            self.assertEqual(result['confidence'], 0.75)
            self.assertEqual(result['match'], True)
            self.assertEqual(result['distance'], 0.25)  # Computed from confidence
            self.assertEqual(result['method'], 'lbph')
            
            # Verify recognizer call
            mock_predict.assert_called_once()
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Mock face data retrieval
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            # Mock recognizer responses
            mock_response = {
                'person_id': 'person_001',
                'confidence': 0.92,
                'match': True,
                'distance': 0.15,
                'method': 'vgg_face'
            }
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                mock_vgg.return_value = mock_response
                
                # Run benchmark
                benchmark_results = recognizer.benchmark_recognition(self.test_face_detection)
                
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
                self.assertIn('vgg_face', benchmark_results['method_performance'])
    
    def test_error_handling(self):
        """Test error handling"""
        recognizer = EnhancedRecognizer(self.test_config)
        
        # Test database error during registration
        with patch.object(recognizer.database, 'save_person') as mock_save:
            mock_save.side_effect = Exception("Database connection failed")
            
            # Register person with error
            result = recognizer.register_person(self.test_face_data)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Database connection failed', result['error'])
        
        # Test database error during recognition
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.side_effect = Exception("Database query failed")
            
            # Recognize face with error
            result = recognizer.recognize_face(self.test_face_detection)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Database query failed', result['error'])
        
        # Test model error during recognition
        with patch.object(recognizer.database, 'get_person') as mock_get:
            mock_get.return_value = self.test_face_data
            
            with patch.object(recognizer.recognizers['vgg_face'], 'recognize') as mock_vgg:
                mock_vgg.side_effect = Exception("Model inference failed")
                
                # Recognize face with model error
                result = recognizer.recognize_face(self.test_face_detection)
                
                self.assertFalse(result['success'])
                self.assertIn('error', result)
                self.assertIn('Model inference failed', result['error'])

if __name__ == '__main__':
    unittest.main()