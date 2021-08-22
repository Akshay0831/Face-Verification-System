"""Test enhanced face recognition functionality"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced.recognition.recognizers import *
from enhanced.recognition.vgg_face import VGGFaceRecognizer
from enhanced.recognition.face_net import FaceNetRecognizer
from enhanced.recognition.arcface import ArcFaceRecognizer

class TestEnhancedRecognition(unittest.TestCase):
    """Test cases for Enhanced Recognition Systems"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test embeddings
        self.test_embedding = np.random.randn(512).astype(np.float32)
        self.test_embedding2 = np.random.randn(512).astype(np.float32)
        
        # Create test face image
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.test_image[50:174, 50:174] = 255  # White rectangle as test face
        
        # Create temporary directory for face database
        self.temp_dir = tempfile.mkdtemp()
        self.face_db_path = os.path.join(self.temp_dir, 'face_database')
        os.makedirs(self.face_db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_euclidean_distance_calculator(self):
        """Test Euclidean distance calculation"""
        from enhanced.recognition.distance import EuclideanDistance
        
        calculator = EuclideanDistance()
        
        # Test same embeddings
        distance = calculator.calculate(self.test_embedding, self.test_embedding)
        self.assertEqual(distance, 0.0)
        
        # Test different embeddings
        distance = calculator.calculate(self.test_embedding, self.test_embedding2)
        self.assertGreater(distance, 0.0)
        
        # Test threshold comparison
        self.assertTrue(calculator.is_match(distance, threshold=0.5))
        self.assertFalse(calculator.is_match(distance, threshold=0.1))
    
    def test_cosine_similarity_calculator(self):
        """Test Cosine similarity calculation"""
        from enhanced.recognition.distance import CosineSimilarity
        
        calculator = CosineSimilarity()
        
        # Test same embeddings (should be close to 1.0)
        similarity = calculator.calculate(self.test_embedding, self.test_embedding)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # Test different embeddings (should be less than 1.0)
        similarity = calculator.calculate(self.test_embedding, self.test_embedding2)
        self.assertLess(similarity, 1.0)
        self.assertGreater(similarity, -1.0)
        
        # Test threshold comparison
        self.assertTrue(calculator.is_match(similarity, threshold=0.7))
        self.assertFalse(calculator.is_match(similarity, threshold=0.9))
    
    def test_face_recognition_base(self):
        """Test base face recognition functionality"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        # Create base recognizer
        recognizer = FaceRecognitionBase()
        
        # Should have basic properties
        self.assertIsNotNone(recognizer)
        self.assertEqual(recognizer.threshold, 0.6)
        self.assertIsInstance(recognizer.face_db, dict)
        self.assertEqual(len(recognizer.face_db), 0)
    
    def test_face_recognition_database_operations(self):
        """Test face database operations"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        
        # Add face to database
        person_id = "test_person"
        name = "Test Person"
        embedding = self.test_embedding.copy()
        
        success = recognizer.add_to_database(person_id, name, embedding)
        self.assertTrue(success)
        self.assertEqual(len(recognizer.face_db), 1)
        self.assertIn(person_id, recognizer.face_db)
        
        # Check stored data
        stored_data = recognizer.face_db[person_id]
        self.assertEqual(stored_data['name'], name)
        self.assertEqual(stored_data['embedding'], embedding)
        
        # Get person name
        retrieved_name = recognizer.get_person_name(person_id)
        self.assertEqual(retrieved_name, name)
        
        # Get all persons
        persons = recognizer.get_all_persons()
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0]['person_id'], person_id)
        self.assertEqual(persons[0]['name'], name)
        
        # Remove person from database
        success = recognizer.remove_from_database(person_id)
        self.assertTrue(success)
        self.assertEqual(len(recognizer.face_db), 0)
        
        # Try to remove non-existent person
        success = recognizer.remove_from_database("nonexistent")
        self.assertFalse(success)
    
    def test_face_recognition_matching(self):
        """Test face recognition matching"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        
        # Add known faces
        recognizer.add_to_database("person1", "Person 1", self.test_embedding)
        recognizer.add_to_database("person2", "Person 2", self.test_embedding2)
        
        # Test matching with same person
        result = recognizer.recognize(self.test_embedding)
        self.assertEqual(result['person_id'], "person1")
        self.assertEqual(result['name'], "Person 1")
        self.assertGreater(result['confidence'], 0.9)
        
        # Test matching with different person
        result = recognizer.recognize(self.test_embedding2)
        self.assertEqual(result['person_id'], "person2")
        self.assertEqual(result['name'], "Person 2")
        self.assertGreater(result['confidence'], 0.9)
        
        # Test unknown face
        unknown_embedding = np.random.randn(512).astype(np.float32)
        result = recognizer.recognize(unknown_embedding)
        self.assertIsNone(result)
    
    def test_face_recognition_threshold_adjustment(self):
        """Test threshold adjustment"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        recognizer.add_to_database("person1", "Person 1", self.test_embedding)
        
        # Test with different thresholds
        recognizer.threshold = 0.5
        result = recognizer.recognize(self.test_embedding)
        self.assertIsNotNone(result)
        
        recognizer.threshold = 0.99  # Very high threshold
        result = recognizer.recognize(self.test_embedding)
        self.assertIsNone(result)
    
    def test_vgg_face_initialization(self):
        """Test VGGFace recognizer initialization"""
        with patch('enhanced.recognition.vgg_face.VGGFace') as mock_vgg:
            mock_vgg.return_value = Mock()
            
            recognizer = VGGFaceRecognizer()
            
            self.assertIsNotNone(recognizer)
            self.assertIsNotNone(recognizer.model)
            self.assertIsNotNone(recognizer.input_shape)
    
    def test_vgg_face_extract_embedding(self):
        """Test VGGFace embedding extraction"""
        with patch('enhanced.recognition.vgg_face.VGGFace') as mock_vgg:
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_vgg.return_value = mock_model
            
            recognizer = VGGFaceRecognizer()
            embedding = recognizer.extract_embedding(self.test_image)
            
            self.assertIsNotNone(embedding)
            self.assertEqual(embedding.shape, (512,))  # VGGFace typically produces 512-dim embeddings
            
            # Verify model was called with correct preprocessing
            mock_model.predict.assert_called_once()
    
    def test_vgg_face_recognition_workflow(self):
        """Test complete VGGFace recognition workflow"""
        with patch('enhanced.recognition.vgg_face.VGGFace') as mock_vgg:
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_vgg.return_value = mock_model
            
            recognizer = VGGFaceRecognizer()
            
            # Add face to database
            recognizer.add_to_database("person1", "Person 1", self.test_embedding)
            
            # Recognize face
            result = recognizer.recognize(self.test_image)
            
            self.assertIsNotNone(result)
            self.assertEqual(result['person_id'], "person1")
            self.assertEqual(result['name'], "Person 1")
            self.assertIsInstance(result['confidence'], float)
    
    def test_facenet_initialization(self):
        """Test FaceNet recognizer initialization"""
        with patch('enhanced.recognition.face_net.KeyPointExtractor') as mock_keypoint:
            with patch('enhanced.recognition.face_net.InceptionResNetV2') as mock_inception:
                mock_keypoint.return_value = Mock()
                mock_inception.return_value = Mock()
                
                recognizer = FaceNetRecognizer()
                
                self.assertIsNotNone(recognizer)
                self.assertIsNotNone(recognizer.keypoint_extractor)
                self.assertIsNotNone(recognizer.model)
                self.assertIsNotNone(recognizer.input_shape)
    
    def test_facenet_extract_embedding(self):
        """Test FaceNet embedding extraction"""
        with patch('enhanced.recognition.face_net.KeyPointExtractor') as mock_keypoint:
            with patch('enhanced.recognition.face_net.InceptionResNetV2') as mock_inception:
                # Mock components
                mock_keypoint.return_value = Mock()
                mock_inception.return_value = Mock()
                mock_inception.return_value.predict.return_value = np.array([[0.1, 0.2, 0.3]])
                
                recognizer = FaceNetRecognizer()
                embedding = recognizer.extract_embedding(self.test_image)
                
                self.assertIsNotNone(embedding)
                self.assertEqual(embedding.shape, (192,))  # FaceNet typically produces 192-dim embeddings
                
                # Verify model was called
                mock_inception.return_value.predict.assert_called_once()
    
    def test_arcface_initialization(self):
        """Test ArcFace recognizer initialization"""
        with patch('enhanced.recognition.arcface.ArcFace') as mock_arcface:
            mock_arcface.return_value = Mock()
            
            recognizer = ArcFaceRecognizer()
            
            self.assertIsNotNone(recognizer)
            self.assertIsNotNone(recognizer.model)
            self.assertIsNotNone(recognizer.input_shape)
    
    def test_arcface_extract_embedding(self):
        """Test ArcFace embedding extraction"""
        with patch('enhanced.recognition.arcface.ArcFace') as mock_arcface:
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_arcface.return_value = mock_model
            
            recognizer = ArcFaceRecognizer()
            embedding = recognizer.extract_embedding(self.test_image)
            
            self.assertIsNotNone(embedding)
            self.assertEqual(embedding.shape, (512,))  # ArcFace typically produces 512-dim embeddings
            
            # Verify model was called with correct preprocessing
            mock_model.predict.assert_called_once()
    
    def test_database_persistence(self):
        """Test database persistence and loading"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        # Create and populate recognizer
        recognizer1 = FaceRecognitionBase()
        recognizer1.add_to_database("person1", "Person 1", self.test_embedding)
        recognizer1.add_to_database("person2", "Person 2", self.test_embedding2)
        
        # Save database
        db_file = os.path.join(self.temp_dir, 'face_db.json')
        success = recognizer1.save_database(db_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(db_file))
        
        # Create new recognizer and load database
        recognizer2 = FaceRecognitionBase()
        success = recognizer2.load_database(db_file)
        self.assertTrue(success)
        
        # Verify data was loaded correctly
        self.assertEqual(len(recognizer2.face_db), 2)
        self.assertIn("person1", recognizer2.face_db)
        self.assertIn("person2", recognizer2.face_db)
        self.assertEqual(recognizer2.get_person_name("person1"), "Person 1")
        self.assertEqual(recognizer2.get_person_name("person2"), "Person 2")
        
        # Test recognition with loaded data
        result = recognizer2.recognize(self.test_embedding)
        self.assertEqual(result['person_id'], "person1")
        self.assertEqual(result['name'], "Person 1")
    
    def test_database_error_handling(self):
        """Test database error handling"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        
        # Save to invalid path
        success = recognizer.save_database("/invalid/path/db.json")
        self.assertFalse(success)
        
        # Load from invalid path
        success = recognizer.load_database("/invalid/path/db.json")
        self.assertFalse(success)
        
        # Load from non-existent file
        success = recognizer.load_database("nonexistent.json")
        self.assertFalse(success)
    
    def test_face_recognition_performance(self):
        """Test face recognition performance"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        
        # Add multiple faces to database
        for i in range(10):
            embedding = np.random.randn(512).astype(np.float32)
            recognizer.add_to_database(f"person{i}", f"Person {i}", embedding)
        
        # Test recognition performance
        test_embedding = np.random.randn(512).astype(np.float32)
        
        import time
        start_time = time.time()
        result = recognizer.recognize(test_embedding)
        end_time = time.time()
        
        # Recognition should be fast (less than 1 second for 10 entries)
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsNotNone(result)
    
    def test_face_recognition_edge_cases(self):
        """Test face recognition edge cases"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        
        # Test with empty embedding
        empty_embedding = np.array([])
        success = recognizer.add_to_database("empty", "Empty Face", empty_embedding)
        self.assertFalse(success)  # Should fail validation
        
        # Test with None embedding
        success = recognizer.add_to_database("none", "None Face", None)
        self.assertFalse(success)  # Should fail validation
        
        # Test recognition with None image
        result = recognizer.recognize(None)
        self.assertIsNone(result)
        
        # Test recognition with empty embedding
        result = recognizer.recognize(np.array([]))
        self.assertIsNone(result)
    
    def test_face_recognition_validation(self):
        """Test face recognition validation"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        
        # Test duplicate person ID
        recognizer.add_to_database("person1", "Person 1", self.test_embedding)
        success = recognizer.add_to_database("person1", "Person 1 Duplicate", self.test_embedding)
        self.assertFalse(success)  # Should fail
        
        # Test empty person ID
        success = recognizer.add_to_database("", "Empty ID", self.test_embedding)
        self.assertFalse(success)  # Should fail
        
        # Test empty person name
        success = recognizer.add_to_database("person2", "", self.test_embedding)
        self.assertFalse(success)  # Should fail
    
    def test_face_recognition_confidence_scoring(self):
        """Test confidence scoring"""
        from enhanced.recognition.base import FaceRecognitionBase
        
        recognizer = FaceRecognitionBase()
        recognizer.threshold = 0.5  # Lower threshold for testing
        
        # Add face
        recognizer.add_to_database("person1", "Person 1", self.test_embedding)
        
        # Test with similar embedding
        similar_embedding = self.test_embedding + np.random.randn(512).astype(np.float32) * 0.1
        result = recognizer.recognize(similar_embedding)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['person_id'], "person1")
        self.assertGreater(result['confidence'], 0.5)  # Should be above threshold
        self.assertLess(result['confidence'], 1.0)  # Should be less than perfect
        
        # Test with dissimilar embedding
        dissimilar_embedding = self.test_embedding + np.random.randn(512).astype(np.float32) * 1.0
        result = recognizer.recognize(dissimilar_embedding)
        
        # Might still match due to threshold, but confidence should be lower
        if result is not None:
            self.assertLess(result['confidence'], 0.8)

if __name__ == '__main__':
    unittest.main()