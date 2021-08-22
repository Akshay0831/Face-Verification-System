"""Test system performance and scalability"""

import unittest
import os
import sys
import json
import tempfile
import time
import threading
import multiprocessing
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import psutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from system import FaceVerificationSystem
from enterprise.enterprise_scalability import EnterpriseScalability

class TestPerformance(unittest.TestCase):
    """Test cases for System Performance"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'performance': {
                'max_workers': 4,
                'batch_size': 32,
                'cache_size': '1GB',
                'enable_gpu': True,
                'enable_optimization': True
            },
            'scalability': {
                'auto_scaling': True,
                'min_instances': 1,
                'max_instances': 10,
                'scaling_threshold': 0.8
            }
        }
        
        # Create test image files
        self.test_images = []
        for i in range(10):
            image_path = os.path.join(self.temp_dir, f'performance_test_{i}.jpg')
            # Create a simple mock image file
            with open(image_path, 'wb') as f:
                f.write(b'fake_image_data_' * 100)  # Larger test data
            self.test_images.append(image_path)
        
        # Create test batch sizes
        self.batch_sizes = [1, 5, 10, 25, 50, 100]
        
        # Create test worker counts
        self.worker_counts = [1, 2, 4, 8, 16]
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_system_initialization_performance(self):
        """Test system initialization performance"""
        config_file = os.path.join(self.temp_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # Test initialization time
        start_time = time.time()
        system = FaceVerificationSystem(config_file=config_file)
        initialization_time = time.time() - start_time
        
        self.assertIsNotNone(system)
        self.assertLess(initialization_time, 5.0)  # Should initialize in less than 5 seconds
        
        print(f"System initialization time: {initialization_time:.2f} seconds")
    
    def test_face_detection_performance(self):
        """Test face detection performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock detector
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': f'face_{i}',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
                for i in range(3)  # 3 faces detected
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Test single image performance
                start_time = time.time()
                result = system.detect_faces(self.test_images[0])
                single_image_time = time.time() - start_time
                
                self.assertTrue(result['success'])
                self.assertLess(single_image_time, 2.0)  # Should process in less than 2 seconds
                
                print(f"Single image detection time: {single_image_time:.2f} seconds")
                
                # Test batch processing performance
                batch_times = []
                for batch_size in self.batch_sizes:
                    if batch_size > len(self.test_images):
                        break
                    
                    start_time = time.time()
                    batch_results = []
                    
                    # Process images in batch
                    for i in range(batch_size):
                        result = system.detect_faces(self.test_images[i])
                        batch_results.append(result)
                    
                    batch_time = time.time() - start_time
                    batch_times.append((batch_size, batch_time))
                    
                    self.assertTrue(all(r['success'] for r in batch_results))
                    self.assertLess(batch_time, 10.0)  # Should complete in less than 10 seconds
                    
                    print(f"Batch size {batch_size}: {batch_time:.2f} seconds "
                          f"(avg: {batch_time/batch_size:.2f} seconds per image)")
    
    def test_face_recognition_performance(self):
        """Test face recognition performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock face data
        face_data = {
            'person_id': 'test_person_001',
            'name': 'Test Person',
            'faces': [
                {
                    'image_id': 'img_001',
                    'encoding': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Mock encoding
                    'timestamp': datetime.now(),
                    'location': 'entrance'
                }
            ],
            'metadata': {
                'department': 'IT',
                'access_level': 'employee',
                'last_seen': datetime.now()
            }
        }
        
        # Mock face detection
        mock_detection = {
            'face_id': 'face_001',
            'bbox': [10, 20, 100, 120],
            'confidence': 0.95,
            'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
            'timestamp': datetime.now()
        }
        
        # Mock recognizer
        with patch.object(system.recognizer, 'recognize_face') as mock_recognize:
            mock_recognize.return_value = {
                'person_id': 'test_person_001',
                'confidence': 0.92,
                'match': True,
                'name': 'Test Person'
            }
            
            with patch.object(system.database, 'get_face_data') as mock_get:
                mock_get.return_value = face_data
                
                # Test single recognition performance
                start_time = time.time()
                result = system.recognize_face(mock_detection)
                recognition_time = time.time() - start_time
                
                self.assertTrue(result['success'])
                self.assertLess(recognition_time, 1.0)  # Should process in less than 1 second
                
                print(f"Single face recognition time: {recognition_time:.2f} seconds")
                
                # Test batch recognition performance
                batch_times = []
                for batch_size in self.batch_sizes:
                    if batch_size > len(self.test_images):
                        break
                    
                    start_time = time.time()
                    batch_results = []
                    
                    # Process recognitions in batch
                    for i in range(batch_size):
                        result = system.recognize_face(mock_detection)
                        batch_results.append(result)
                    
                    batch_time = time.time() - start_time
                    batch_times.append((batch_size, batch_time))
                    
                    self.assertTrue(all(r['success'] for r in batch_results))
                    self.assertLess(batch_time, 5.0)  # Should complete in less than 5 seconds
                    
                    print(f"Batch recognition size {batch_size}: {batch_time:.2f} seconds "
                          f"(avg: {batch_time/batch_size:.2f} seconds per recognition)")
    
    def test_liveness_detection_performance(self):
        """Test liveness detection performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock face detection
        mock_detection = {
            'face_id': 'face_001',
            'bbox': [10, 20, 100, 120],
            'confidence': 0.95,
            'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
            'timestamp': datetime.now()
        }
        
        # Mock liveness detector
        with patch.object(system.liveness_detector, 'detect_liveness') as mock_liveness:
            mock_liveness.return_value = {
                'is_live': True,
                'confidence': 0.88,
                'liveness_score': 0.92,
                'method_used': 'motion_analysis',
                'timestamp': datetime.now()
            }
            
            with patch.object(system.database, 'save_liveness_result') as mock_save:
                mock_save.return_value = True
                
                # Test single liveness detection performance
                start_time = time.time()
                result = system.detect_liveness(mock_detection)
                liveness_time = time.time() - start_time
                
                self.assertTrue(result['success'])
                self.assertLess(liveness_time, 2.0)  # Should process in less than 2 seconds
                
                print(f"Single liveness detection time: {liveness_time:.2f} seconds")
                
                # Test batch liveness detection performance
                batch_times = []
                for batch_size in self.batch_sizes:
                    if batch_size > len(self.test_images):
                        break
                    
                    start_time = time.time()
                    batch_results = []
                    
                    # Process liveness detections in batch
                    for i in range(batch_size):
                        result = system.detect_liveness(mock_detection)
                        batch_results.append(result)
                    
                    batch_time = time.time() - start_time
                    batch_times.append((batch_size, batch_time))
                    
                    self.assertTrue(all(r['success'] for r in batch_results))
                    self.assertLess(batch_time, 8.0)  # Should complete in less than 8 seconds
                    
                    print(f"Batch liveness detection size {batch_size}: {batch_time:.2f} seconds "
                          f"(avg: {batch_time/batch_size:.2f} seconds per detection)")
    
    def test_complete_verification_pipeline_performance(self):
        """Test complete verification pipeline performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock face detection
        mock_detection = {
            'face_id': 'face_001',
            'bbox': [10, 20, 100, 120],
            'confidence': 0.95,
            'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
            'timestamp': datetime.now()
        }
        
        # Mock face recognition
        mock_recognition = {
            'person_id': 'test_person_001',
            'confidence': 0.92,
            'match': True,
            'name': 'Test Person'
        }
        
        # Mock liveness detection
        mock_liveness = {
            'is_live': True,
            'confidence': 0.88,
            'liveness_score': 0.92,
            'method_used': 'motion_analysis',
            'timestamp': datetime.now()
        }
        
        # Mock database
        with patch.object(system.database, 'get_face_data') as mock_get:
            mock_get.return_value = {
                'person_id': 'test_person_001',
                'name': 'Test Person',
                'faces': [],
                'metadata': {}
            }
            
            with patch.object(system.database, 'save_verification') as mock_save:
                mock_save.return_value = True
                
                with patch.object(system.notifier, 'send_verification_notification') as mock_notify:
                    mock_notify.return_value = True
                    
                    # Test single verification performance
                    start_time = time.time()
                    result = system.verify_face(self.test_images[0])
                    verification_time = time.time() - start_time
                    
                    self.assertTrue(result['success'])
                    self.assertLess(verification_time, 5.0)  # Should complete in less than 5 seconds
                    
                    print(f"Single verification time: {verification_time:.2f} seconds")
                    
                    # Test batch verification performance
                    batch_times = []
                    for batch_size in self.batch_sizes:
                        if batch_size > len(self.test_images):
                            break
                        
                        start_time = time.time()
                        batch_results = []
                        
                        # Process verifications in batch
                        for i in range(batch_size):
                            result = system.verify_face(self.test_images[i])
                            batch_results.append(result)
                        
                        batch_time = time.time() - start_time
                        batch_times.append((batch_size, batch_time))
                        
                        self.assertTrue(all(r['success'] for r in batch_results))
                        self.assertLess(batch_time, 15.0)  # Should complete in less than 15 seconds
                        
                        print(f"Batch verification size {batch_size}: {batch_time:.2f} seconds "
                              f"(avg: {batch_time/batch_size:.2f} seconds per verification)")
    
    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock detector
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': f'face_{i}',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
                for i in range(3)
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Test different worker counts
                worker_times = []
                for worker_count in self.worker_counts:
                    if worker_count > multiprocessing.cpu_count():
                        continue
                    
                    start_time = time.time()
                    
                    # Process images with multiple workers
                    with multiprocessing.Pool(processes=worker_count) as pool:
                        results = pool.map(system.detect_faces, self.test_images)
                    
                    processing_time = time.time() - start_time
                    worker_times.append((worker_count, processing_time))
                    
                    self.assertTrue(all(r['success'] for r in results))
                    self.assertLess(processing_time, 10.0)  # Should complete in less than 10 seconds
                    
                    print(f"Workers {worker_count}: {processing_time:.2f} seconds "
                          f"(avg: {processing_time/len(self.test_images):.2f} seconds per image)")
    
    def test_memory_usage_performance(self):
        """Test memory usage performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock detector
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': f'face_{i}',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
                for i in range(3)
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Track memory usage during processing
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Process multiple images
                for _ in range(10):
                    for image_path in self.test_images:
                        system.detect_faces(image_path)
                
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                self.assertLess(memory_increase, 100)  # Memory increase should be less than 100MB
                print(f"Memory increase: {memory_increase:.2f} MB")
    
    def test_cache_performance(self):
        """Test cache performance"""
        # Initialize system with cache
        config = self.test_config.copy()
        config['performance']['cache_size'] = '10MB'
        
        config_file = os.path.join(self.temp_dir, 'cache_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        system = FaceVerificationSystem(config_file=config_file)
        
        # Mock detector with caching behavior
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Test cache hit vs miss performance
                cache_times = []
                
                # First pass - cache miss
                start_time = time.time()
                for image_path in self.test_images:
                    system.detect_faces(image_path)
                first_pass_time = time.time() - start_time
                cache_times.append(('miss', first_pass_time))
                
                # Second pass - cache hit
                start_time = time.time()
                for image_path in self.test_images:
                    system.detect_faces(image_path)
                second_pass_time = time.time() - start_time
                cache_times.append(('hit', second_pass_time))
                
                # Calculate performance improvement
                improvement = (first_pass_time - second_pass_time) / first_pass_time * 100
                self.assertGreater(improvement, 0)  # Cache should provide improvement
                
                print(f"Cache miss time: {first_pass_time:.2f} seconds")
                print(f"Cache hit time: {second_pass_time:.2f} seconds")
                print(f"Performance improvement: {improvement:.1f}%")
    
    def test_scalability_performance(self):
        """Test system scalability performance"""
        # Initialize scalability engine
        scalability = EnterpriseScalability()
        
        # Mock load data
        load_data = [
            {
                'timestamp': datetime.now(),
                'cpu_usage': 0.45,
                'memory_usage': 0.67,
                'request_rate': 100,
                'response_time': 0.25
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'cpu_usage': 0.52,
                'memory_usage': 0.71,
                'request_rate': 120,
                'response_time': 0.28
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=10),
                'cpu_usage': 0.48,
                'memory_usage': 0.69,
                'request_rate': 110,
                'response_time': 0.26
            }
        ]
        
        # Test scalability analysis performance
        start_time = time.time()
        analysis = scalability.engine.analyze_load_patterns(load_data)
        analysis_time = time.time() - start_time
        
        self.assertIsNotNone(analysis)
        self.assertLess(analysis_time, 1.0)  # Should complete in less than 1 second
        
        print(f"Scalability analysis time: {analysis_time:.2f} seconds")
        
        # Test load prediction performance
        start_time = time.time()
        prediction = scalability.engine.predict_load_demand(load_data, forecast_period=24)
        prediction_time = time.time() - start_time
        
        self.assertIsNotNone(prediction)
        self.assertLess(prediction_time, 2.0)  # Should complete in less than 2 seconds
        
        print(f"Load prediction time: {prediction_time:.2f} seconds")
    
    def test_load_balancing_performance(self):
        """Test load balancing performance"""
        # Initialize system with load balancing
        system = FaceVerificationSystem()
        
        # Mock device optimizer
        with patch.object(system.device_optimizer, 'get_available_devices') as mock_get_devices:
            mock_get_devices.return_value = [
                {'id': 'device_001', 'status': 'active', 'load': 0.3},
                {'id': 'device_002', 'status': 'active', 'load': 0.7},
                {'id': 'device_003', 'status': 'active', 'load': 0.5}
            ]
            
            with patch.object(system.device_optimizer, 'select_optimal_device') as mock_select:
                mock_select.return_value = 'device_001'
                
                # Test load balancing performance with different load patterns
                load_patterns = [
                    0.1, 0.3, 0.5, 0.7, 0.9  # Different load levels
                ]
                
                selection_times = []
                for load in load_patterns:
                    # Mock load values
                    with patch('system.device_optimizer.DeviceOptimizer.get_device_load') as mock_load:
                        mock_load.return_value = load
                        
                        start_time = time.time()
                        selected_device = system.select_device_for_processing()
                        selection_time = time.time() - start_time
                        
                        selection_times.append((load, selection_time))
                        self.assertEqual(selected_device, 'device_001')
                        self.assertLess(selection_time, 0.1)  # Should complete in less than 0.1 second
                
                # Print results
                print("Load balancing performance:")
                for load, time_taken in selection_times:
                    print(f"Load {load}: {time_taken:.4f} seconds")
    
    def test_error_handling_performance(self):
        """Test error handling performance"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Test database error performance
        with patch.object(system.database, 'save_detection') as mock_save:
            mock_save.side_effect = Exception("Database connection failed")
            
            start_time = time.time()
            result = system.detect_faces(self.test_images[0])
            error_handling_time = time.time() - start_time
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertLess(error_handling_time, 1.0)  # Should complete in less than 1 second
            
            print(f"Database error handling time: {error_handling_time:.2f} seconds")
        
        # Test detector error performance
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.side_effect = Exception("Detection failed")
            
            start_time = time.time()
            result = system.detect_faces(self.test_images[0])
            error_handling_time = time.time() - start_time
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertLess(error_handling_time, 1.0)  # Should complete in less than 1 second
            
            print(f"Detector error handling time: {error_handling_time:.2f} seconds")
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Run some operations to generate performance data
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': 'face_001',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Run multiple operations
                for _ in range(10):
                    for image_path in self.test_images:
                        system.detect_faces(image_path)
        
        # Get performance metrics
        metrics = system.get_performance_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertIn('total_detections', metrics)
        self.assertIn('average_processing_time', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('throughput', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('cpu_usage', metrics)
        
        # Check that metrics are reasonable
        self.assertEqual(metrics['total_detections'], 100)  # 10 operations × 10 images
        self.assertGreater(metrics['success_rate'], 0.95)  # Should have high success rate
        self.assertGreater(metrics['throughput'], 0)  # Should have positive throughput
        self.assertGreater(metrics['average_processing_time'], 0)
        self.assertLess(metrics['average_processing_time'], 5.0)  # Should be reasonable
        
        print(f"Performance metrics:")
        print(f"Total detections: {metrics['total_detections']}")
        print(f"Average processing time: {metrics['average_processing_time']:.2f} seconds")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        print(f"Throughput: {metrics['throughput']:.2f} operations/second")
        print(f"Memory usage: {metrics['memory_usage']:.2f} MB")
        print(f"CPU usage: {metrics['cpu_usage']:.2f}%")
    
    def test_stress_test(self):
        """Test system under stress conditions"""
        # Initialize system
        system = FaceVerificationSystem()
        
        # Mock detector
        with patch.object(system.detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [
                {
                    'face_id': f'face_{i}',
                    'bbox': [10, 20, 100, 120],
                    'confidence': 0.95,
                    'landmarks': {'left_eye': [30, 40], 'right_eye': [80, 40]},
                    'timestamp': datetime.now()
                }
                for i in range(5)
            ]
            
            with patch.object(system.database, 'save_detection') as mock_save:
                mock_save.return_value = True
                
                # Create high load scenario
                high_load_start = time.time()
                
                # Process large number of images
                results = []
                for _ in range(100):  # 100 iterations
                    batch_results = []
                    for image_path in self.test_images:
                        result = system.detect_faces(image_path)
                        batch_results.append(result)
                    results.append(batch_results)
                
                high_load_time = time.time() - high_load_start
                
                # Check results
                self.assertEqual(len(results), 100)
                for batch in results:
                    for result in batch:
                        self.assertTrue(result['success'])
                
                # Performance should scale reasonably
                total_images = 100 * len(self.test_images)
                avg_time_per_image = high_load_time / total_images
                
                self.assertLess(high_load_time, 60.0)  # Should complete in less than 60 seconds
                self.assertLess(avg_time_per_image, 0.5)  # Average should be less than 0.5 seconds per image
                
                print(f"Stress test results:")
                print(f"Total images processed: {total_images}")
                print(f"Total time: {high_load_time:.2f} seconds")
                print(f"Average time per image: {avg_time_per_image:.4f} seconds")
                print(f"Throughput: {total_images/high_load_time:.2f} images/second")
                
                # Get performance metrics under load
                metrics = system.get_performance_metrics()
                self.assertGreater(metrics['total_detections'], 0)
                self.assertGreater(metrics['throughput'], 0)
                self.assertLess(metrics['average_processing_time'], 1.0)

if __name__ == '__main__':
    unittest.main()