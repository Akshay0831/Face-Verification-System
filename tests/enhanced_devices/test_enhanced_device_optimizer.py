"""Test enhanced device optimization functionality"""

import unittest
import os
import sys
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import psutil
import platform

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_devices.optimizers.enhanced_device_optimizer import (
    EnhancedDeviceOptimizer, 
    PerformanceOptimizer,
    MemoryOptimizer,
    CpuOptimizer,
    GpuOptimizer,
    ResourceScheduler
)

class TestEnhancedDeviceOptimizer(unittest.TestCase):
    """Test cases for Enhanced Device Optimizer"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Test device information
        self.test_device_info = {
            'device_id': 'device_001',
            'device_type': 'edge_device',
            'location': 'main_entrance',
            'capabilities': {
                'cpu': {
                    'cores': 4,
                    'threads': 8,
                    'frequency': 2.4,
                    'architecture': 'x86_64'
                },
                'gpu': {
                    'available': True,
                    'memory': 4096,
                    'cores': 192,
                    'frequency': 1.5
                },
                'memory': {
                    'total': 8192,
                    'available': 6144,
                    'used': 2048
                },
                'storage': {
                    'total': 500000,
                    'available': 350000,
                    'used': 150000
                }
            },
            'current_usage': {
                'cpu': 45.0,
                'gpu': 20.0,
                'memory': 25.0,
                'storage': 30.0
            },
            'temperature': {
                'cpu': 65.0,
                'gpu': 55.0,
                'system': 45.0
            },
            'power_usage': 150.0,  # Watts
            'network': {
                'latency': 5.0,
                'bandwidth': 1000.0,
                'status': 'connected'
            },
            'environment': {
                'temperature': 25.0,
                'humidity': 45.0,
                'lighting': 'normal'
            },
            'timestamp': datetime.now()
        }
        
        # Test optimization task
        self.test_optimization_task = {
            'task_id': 'task_001',
            'type': 'performance_optimization',
            'priority': 'high',
            'description': 'Optimize face recognition performance',
            'target_metric': 'fps',
            'target_value': 30.0,
            'current_value': 15.0,
            'parameters': {
                'batch_size': 16,
                'use_gpu': True,
                'parallel_workers': 4,
                'cache_size': 1024
            },
            'deadline': datetime.now() + timedelta(minutes=30),
            'resource_requirements': {
                'cpu': 2,
                'gpu': 1,
                'memory': 2048
            }
        }
        
        # Test configuration
        self.test_config = {
            'optimization_modes': [
                'performance',
                'memory',
                'power',
                'temperature',
                'quality'
            ],
            'performance_settings': {
                'target_fps': 30,
                'max_latency': 10.0,
                'min_confidence': 0.85,
                'gpu_utilization_target': 80.0,
                'cpu_utilization_target': 70.0,
                'auto_batch_size': True,
                'batch_size_min': 1,
                'batch_size_max': 64,
                'parallel_workers': 4,
                'cache_enabled': True,
                'cache_size': 2048,
                'preload_models': True
            },
            'memory_settings': {
                'memory_target': 0.7,  # 70% of total memory
                'min_free_memory': 1024,
                'cache_cleanup_threshold': 0.85,
                'enable_memory_profiling': True,
                'memory_optimization_level': 'aggressive',
                'prevent_swap': True,
                'memory_preallocation': True
            },
            'cpu_settings': {
                'cpu_target': 0.75,  # 75% CPU utilization
                'min_free_cores': 1,
                'enable_cpu_scaling': True,
                'cpu_scaling_threshold': 0.8,
                'enable_cpu_affinity': True,
                'enable_cpu_isolation': True,
                'priority': 'normal'
            },
            'gpu_settings': {
                'gpu_target': 0.8,  # 80% GPU utilization
                'memory_target': 0.85,  # 85% GPU memory
                'enable_gpu_scaling': True,
                'gpu_scaling_threshold': 0.9,
                'enable_gpu_memory_optimization': True,
                'enable_power_management': True,
                'gpu_temperature_threshold': 85.0
            },
            'power_settings': {
                'power_target': 100.0,  # Watts
                'enable_power_scaling': True,
                'power_scaling_threshold': 0.85,
                'enable_dynamic_power': True,
                'power_profile': 'balanced',
                'battery_threshold': 0.2
            },
            'temperature_settings': {
                'cpu_temp_max': 85.0,
                'gpu_temp_max': 85.0,
                'system_temp_max': 75.0,
                'enable_cooling': True,
                'enable_thermal_throttling': True,
                'thermal_threshold': 80.0,
                'cooling_method': 'passive',
                'monitoring_interval': 5.0
            },
            'quality_settings': {
                'detection_threshold': 0.7,
                'recognition_threshold': 0.8,
                'liveness_threshold': 0.9,
                'max_resolution': 1080,
                'min_resolution': 640,
                'enable_multi_resolution': True,
                'quality_adjustment_interval': 30.0
            },
            'scheduling': {
                'enable_priority_based': True,
                'enable_time_based': True,
                'enable_load_based': True,
                'max_concurrent_tasks': 5,
                'task_timeout': 300,
                'enable_preemption': True,
                'priority_levels': ['critical', 'high', 'medium', 'low']
            },
            'monitoring': {
                'monitoring_interval': 5.0,
                'enable_metrics_collection': True,
                'enable_alerting': True,
                'enable_logging': True,
                'metrics_retention_days': 7,
                'alert_thresholds': {
                    'cpu_usage': 90.0,
                    'memory_usage': 90.0,
                    'gpu_usage': 90.0,
                    'temperature': 85.0,
                    'power_usage': 200.0,
                    'latency': 50.0
                }
            }
        }
        
        # Test current system state
        self.test_system_state = {
            'timestamp': datetime.now(),
            'cpu': {
                'usage': 65.0,
                'temperature': 70.0,
                'frequency': 2.2,
                'cores': [45.0, 50.0, 60.0, 55.0],
                'processes': [
                    {'pid': 1234, 'name': 'face_detection', 'cpu': 30.0},
                    {'pid': 5678, 'name': 'face_recognition', 'cpu': 25.0}
                ]
            },
            'gpu': {
                'usage': 75.0,
                'memory_usage': 60.0,
                'temperature': 65.0,
                'processes': [
                    {'pid': 1234, 'name': 'face_detection', 'gpu': 40.0},
                    {'pid': 5678, 'name': 'face_recognition', 'gpu': 35.0}
                ]
            },
            'memory': {
                'total': 8192,
                'available': 4096,
                'used': 4096,
                'cache': 1024,
                'swap_used': 0
            },
            'storage': {
                'total': 500000,
                'available': 350000,
                'used': 150000
            },
            'power': {
                'current': 180.0,
                'average': 160.0,
                'peak': 200.0
            },
            'network': {
                'latency': 8.0,
                'bandwidth': 800.0,
                'connections': 5
            },
            'temperature': {
                'cpu': 70.0,
                'gpu': 65.0,
                'system': 50.0
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_device_optimizer_initialization(self):
        """Test EnhancedDeviceOptimizer initialization"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.optimizer_engine)
        self.assertIsNotNone(optimizer.scheduler)
        self.assertIsNotNone(optimizer.monitor)
        self.assertIsNotNone(optimizer.performance_optimizer)
        self.assertIsNotNone(optimizer.memory_optimizer)
        self.assertIsNotNone(optimizer.cpu_optimizer)
        self.assertIsNotNone(optimizer.gpu_optimizer)
        
        # Check configurations
        self.assertEqual(optimizer.config['performance_settings']['target_fps'], 30)
        self.assertEqual(optimizer.config['memory_settings']['memory_target'], 0.7)
        self.assertEqual(optimizer.config['cpu_settings']['cpu_target'], 0.75)
        self.assertEqual(optimizer.config['gpu_settings']['gpu_target'], 0.8)
        self.assertEqual(optimizer.config['monitoring']['monitoring_interval'], 5.0)
    
    def test_enhanced_device_optimizer_initialization_default_config(self):
        """Test EnhancedDeviceOptimizer initialization with default config"""
        optimizer = EnhancedDeviceOptimizer()
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        
        # Check default values
        self.assertEqual(optimizer.config['performance_settings']['target_fps'], 25)
        self.assertEqual(optimizer.config['memory_settings']['memory_target'], 0.6)
        self.assertEqual(optimizer.config['cpu_settings']['cpu_target'], 0.7)
        self.assertEqual(optimizer.config['gpu_settings']['gpu_target'], 0.75)
        self.assertEqual(optimizer.config['monitoring']['monitoring_interval'], 10.0)
    
    def test_enhanced_device_optimizer_optimize_device(self):
        """Test device optimization"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        # Mock optimizer responses
        mock_performance_response = {
            'success': True,
            'metric': 'fps',
            'before': 15.0,
            'after': 32.0,
            'improvement': 113.3,
            'optimizations': [
                {'type': 'batch_size', 'from': 8, 'to': 16},
                {'type': 'gpu_acceleration', 'enabled': True},
                {'type': 'parallel_processing', 'workers': 4}
            ],
            'processing_time': 0.5,
            'resource_usage': {
                'cpu': 75.0,
                'gpu': 80.0,
                'memory': 4096
            }
        }
        
        mock_memory_response = {
            'success': True,
            'metric': 'memory_usage',
            'before': 4096,
            'after': 3072,
            'freed': 1024,
            'optimizations': [
                {'type': 'cache_cleanup', 'freed': 512},
                {'type': 'memory_defragmentation', 'improvement': 20.0}
            ],
            'processing_time': 0.3,
            'resource_usage': {
                'cpu': 30.0,
                'memory': 3072
            }
        }
        
        mock_cpu_response = {
            'success': True,
            'metric': 'cpu_utilization',
            'before': 65.0,
            'after': 70.0,
            'optimizations': [
                {'type': 'cpu_scaling', 'frequency': 2.4},
                {'type': 'process_priority', 'priority': 'high'}
            ],
            'processing_time': 0.2,
            'resource_usage': {
                'cpu': 70.0,
                'memory': 2048
            }
        }
        
        mock_gpu_response = {
            'success': True,
            'metric': 'gpu_utilization',
            'before': 75.0,
            'after': 85.0,
            'optimizations': [
                {'type': 'gpu_scaling', 'power_limit': 150},
                {'type': 'memory_optimization', 'memory_target': 85.0}
            ],
            'processing_time': 0.4,
            'resource_usage': {
                'gpu': 85.0,
                'memory': 3482
            }
        }
        
        with patch.object(optimizer.performance_optimizer, 'optimize') as mock_perf:
            with patch.object(optimizer.memory_optimizer, 'optimize') as mock_mem:
                with patch.object(optimizer.cpu_optimizer, 'optimize') as mock_cpu:
                    with patch.object(optimizer.gpu_optimizer, 'optimize') as mock_gpu:
                        mock_perf.return_value = mock_performance_response
                        mock_mem.return_value = mock_memory_response
                        mock_cpu.return_value = mock_cpu_response
                        mock_gpu.return_value = mock_gpu_response
                        
                        # Optimize device
                        result = optimizer.optimize_device(
                            self.test_device_info,
                            self.test_optimization_task
                        )
                        
                        self.assertIsNotNone(result)
                        self.assertIn('success', result)
                        self.assertIn('device_id', result)
                        self.assertIn('optimizations', result)
                        self.assertIn('metrics', result)
                        self.assertIn('resource_usage', result)
                        self.assertIn('processing_time', result)
                        self.assertIn('recommendations', result)
                        
                        # Check result values
                        self.assertTrue(result['success'])
                        self.assertEqual(result['device_id'], 'device_001')
                        self.assertIsInstance(result['processing_time'], float)
                        self.assertEqual(len(result['optimizations']), 4)  # 4 optimizers
                        
                        # Check metrics
                        self.assertIsInstance(result['metrics'], dict)
                        self.assertIn('fps', result['metrics'])
                        self.assertIn('memory_usage', result['metrics'])
                        self.assertIn('cpu_utilization', result['metrics'])
                        self.assertIn('gpu_utilization', result['metrics'])
                        
                        # Check recommendations
                        self.assertIsInstance(result['recommendations'], list)
                        
                        # Verify optimizers were called
                        mock_perf.assert_called_once()
                        mock_mem.assert_called_once()
                        mock_cpu.assert_called_once()
                        mock_gpu.assert_called_once()
    
    def test_enhanced_device_optimizer_optimize_device_partial_failure(self):
        """Test device optimization with partial failures"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        # Mock partial failures
        mock_performance_response = {
            'success': True,
            'metric': 'fps',
            'before': 15.0,
            'after': 25.0,
            'improvement': 66.7
        }
        
        mock_memory_response = {
            'success': False,
            'metric': 'memory_usage',
            'error': 'Insufficient memory available'
        }
        
        mock_cpu_response = {
            'success': True,
            'metric': 'cpu_utilization',
            'before': 65.0,
            'after': 70.0
        }
        
        mock_gpu_response = {
            'success': False,
            'metric': 'gpu_utilization',
            'error': 'GPU temperature too high'
        }
        
        with patch.object(optimizer.performance_optimizer, 'optimize') as mock_perf:
            with patch.object(optimizer.memory_optimizer, 'optimize') as mock_mem:
                with patch.object(optimizer.cpu_optimizer, 'optimize') as mock_cpu:
                    with patch.object(optimizer.gpu_optimizer, 'optimize') as mock_gpu:
                        mock_perf.return_value = mock_performance_response
                        mock_mem.return_value = mock_memory_response
                        mock_cpu.return_value = mock_cpu_response
                        mock_gpu.return_value = mock_gpu_response
                        
                        # Optimize device
                        result = optimizer.optimize_device(
                            self.test_device_info,
                            self.test_optimization_task
                        )
                        
                        self.assertIsNotNone(result)
                        self.assertTrue(result['success'])
                        self.assertEqual(len(result['optimizations']), 4)
                        
                        # Check that failures are recorded
                        self.assertIn('failed_optimizations', result)
                        self.assertEqual(len(result['failed_optimizations']), 2)
                        self.assertIn('memory_usage', result['failed_optimizations'])
                        self.assertIn('gpu_utilization', result['failed_optimizations'])
    
    def test_enhanced_device_optimizer_batch_optimization(self):
        """Test batch device optimization"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        # Mock optimization response
        mock_response = {
            'success': True,
            'metric': 'fps',
            'before': 15.0,
            'after': 32.0,
            'improvement': 113.3
        }
        
        with patch.object(optimizer.performance_optimizer, 'optimize') as mock_perf:
            mock_perf.return_value = mock_response
            
            # Batch optimize devices
            devices = [self.test_device_info] * 3
            tasks = [self.test_optimization_task] * 3
            results = optimizer.optimize_devices_batch(devices, tasks)
            
            self.assertIsNotNone(results)
            self.assertEqual(len(results), 3)
            
            # Check each result
            for i, result in enumerate(results):
                self.assertIn('success', result)
                self.assertIn('device_id', result)
                self.assertIn('processing_time', result)
                
                # Verify optimizers were called for each device
                mock_perf.assert_any_call(devices[i], tasks[i])
    
    def test_performance_optimizer_initialization(self):
        """Test PerformanceOptimizer initialization"""
        config = {
            'target_fps': 30,
            'max_latency': 10.0,
            'min_confidence': 0.85,
            'gpu_utilization_target': 80.0,
            'cpu_utilization_target': 70.0
        }
        
        optimizer = PerformanceOptimizer(config)
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.gpu_manager)
        self.assertIsNotNone(optimizer.batch_manager)
        self.assertIsNotNone(optimizer.cache_manager)
        
        # Check configuration
        self.assertEqual(optimizer.config['target_fps'], 30)
        self.assertEqual(optimizer.config['max_latency'], 10.0)
        self.assertEqual(optimizer.config['min_confidence'], 0.85)
    
    def test_performance_optimizer_optimize(self):
        """Test performance optimization"""
        config = {
            'target_fps': 30,
            'max_latency': 10.0,
            'min_confidence': 0.85,
            'auto_batch_size': True,
            'batch_size_min': 1,
            'batch_size_max': 64,
            'parallel_workers': 4,
            'cache_enabled': True,
            'cache_size': 2048
        }
        
        optimizer = PerformanceOptimizer(config)
        
        # Mock device info
        device_info = self.test_device_info
        
        # Mock task
        task = self.test_optimization_task
        
        # Mock performance analysis
        mock_analysis = {
            'current_fps': 15.0,
            'current_latency': 25.0,
            'gpu_usage': 50.0,
            'cpu_usage': 60.0,
            'bottlenecks': ['batch_size', 'gpu_utilization'],
            'optimization_potential': 120.0
        }
        
        # Mock optimization suggestions
        mock_suggestions = [
            {'type': 'batch_size', 'value': 16, 'potential_improvement': 0.4},
            {'type': 'gpu_acceleration', 'enabled': True, 'potential_improvement': 0.3},
            {'type': 'parallel_workers', 'value': 4, 'potential_improvement': 0.2}
        ]
        
        with patch.object(optimizer, '_analyze_performance') as mock_analyze:
            with patch.object(optimizer, '_generate_optimization_suggestions') as mock_suggest:
                with patch.object(optimizer, '_apply_optimizations') as mock_apply:
                    mock_analyze.return_value = mock_analysis
                    mock_suggest.return_value = mock_suggestions
                    mock_apply.return_value = mock_suggestions[:2]  # Apply first 2 suggestions
                    
                    # Optimize performance
                    result = optimizer.optimize(device_info, task)
                    
                    self.assertIsNotNone(result)
                    self.assertTrue(result['success'])
                    self.assertEqual(result['metric'], 'fps')
                    self.assertEqual(result['before'], 15.0)
                    self.assertEqual(result['after'], 32.0)
                    self.assertEqual(result['improvement'], 113.3)
                    self.assertIn('optimizations', result)
                    self.assertIn('resource_usage', result)
                    self.assertIn('processing_time', result)
                    
                    # Verify calls were made
                    mock_analyze.assert_called_once()
                    mock_suggest.assert_called_once()
                    mock_apply.assert_called_once()
    
    def test_memory_optimizer_initialization(self):
        """Test MemoryOptimizer initialization"""
        config = {
            'memory_target': 0.7,
            'min_free_memory': 1024,
            'cache_cleanup_threshold': 0.85,
            'enable_memory_profiling': True
        }
        
        optimizer = MemoryOptimizer(config)
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.memory_monitor)
        self.assertIsNotNone(optimizer.cache_manager)
        self.assertIsNotNone(optimizer.memory_profiler)
        
        # Check configuration
        self.assertEqual(optimizer.config['memory_target'], 0.7)
        self.assertEqual(optimizer.config['min_free_memory'], 1024)
        self.assertEqual(optimizer.config['cache_cleanup_threshold'], 0.85)
    
    def test_memory_optimizer_optimize(self):
        """Test memory optimization"""
        config = {
            'memory_target': 0.7,
            'min_free_memory': 1024,
            'cache_cleanup_threshold': 0.85,
            'enable_memory_profiling': True,
            'memory_optimization_level': 'aggressive'
        }
        
        optimizer = MemoryOptimizer(config)
        
        # Mock device info
        device_info = self.test_device_info
        
        # Mock task
        task = self.test_optimization_task
        
        # Mock memory analysis
        mock_analysis = {
            'current_memory_usage': 4096,
            'target_memory_usage': 3072,
            'memory_pressure': 'high',
            'cache_usage': 1024,
            'memory_leaks': [],
            'optimization_potential': 1024
        }
        
        # Mock memory optimization
        mock_optimization = {
            'cache_cleanup': 512,
            'memory_defragmentation': 256,
            'process_memory_limit': 256
        }
        
        with patch.object(optimizer, '_analyze_memory') as mock_analyze:
            with patch.object(optimizer, '_perform_memory_optimization') as mock_opt:
                mock_analyze.return_value = mock_analysis
                mock_opt.return_value = mock_optimization
                
                # Optimize memory
                result = optimizer.optimize(device_info, task)
                
                self.assertIsNotNone(result)
                self.assertTrue(result['success'])
                self.assertEqual(result['metric'], 'memory_usage')
                self.assertEqual(result['before'], 4096)
                self.assertEqual(result['after'], 3328)
                self.assertEqual(result['freed'], 768)
                self.assertIn('optimizations', result)
                self.assertIn('resource_usage', result)
                self.assertIn('processing_time', result)
                
                # Verify calls were made
                mock_analyze.assert_called_once()
                mock_opt.assert_called_once()
    
    def test_cpu_optimizer_initialization(self):
        """Test CpuOptimizer initialization"""
        config = {
            'cpu_target': 0.75,
            'min_free_cores': 1,
            'enable_cpu_scaling': True,
            'enable_cpu_affinity': True,
            'priority': 'normal'
        }
        
        optimizer = CpuOptimizer(config)
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.cpu_monitor)
        self.assertIsNotNone(optimizer.cpu_scheduler)
        self.assertIsNotNone(optimizer.frequency_scaler)
        
        # Check configuration
        self.assertEqual(optimizer.config['cpu_target'], 0.75)
        self.assertEqual(optimizer.config['min_free_cores'], 1)
        self.assertTrue(optimizer.config['enable_cpu_scaling'])
        self.assertTrue(optimizer.config['enable_cpu_affinity'])
    
    def test_cpu_optimizer_optimize(self):
        """Test CPU optimization"""
        config = {
            'cpu_target': 0.75,
            'min_free_cores': 1,
            'enable_cpu_scaling': True,
            'enable_cpu_affinity': True,
            'cpu_scaling_threshold': 0.8
        }
        
        optimizer = CpuOptimizer(config)
        
        # Mock device info
        device_info = self.test_device_info
        
        # Mock task
        task = self.test_optimization_task
        
        # Mock CPU analysis
        mock_analysis = {
            'current_cpu_usage': 65.0,
            'target_cpu_usage': 75.0,
            'cpu_pressure': 'medium',
            'frequency': 2.2,
            'frequency_potential': 0.2,
            'bottlenecks': ['frequency_scaling'],
            'optimization_potential': 15.4
        }
        
        # Mock CPU optimization
        mock_optimization = {
            'frequency_scaling': 2.4,
            'cpu_affinity': [0, 2, 4, 6],
            'process_priority': 'high'
        }
        
        with patch.object(optimizer, '_analyze_cpu') as mock_analyze:
            with patch.object(optimizer, '_perform_cpu_optimization') as mock_opt:
                mock_analyze.return_value = mock_analysis
                mock_opt.return_value = mock_optimization
                
                # Optimize CPU
                result = optimizer.optimize(device_info, task)
                
                self.assertIsNotNone(result)
                self.assertTrue(result['success'])
                self.assertEqual(result['metric'], 'cpu_utilization')
                self.assertEqual(result['before'], 65.0)
                self.assertEqual(result['after'], 75.0)
                self.assertIn('optimizations', result)
                self.assertIn('resource_usage', result)
                self.assertIn('processing_time', result)
                
                # Verify calls were made
                mock_analyze.assert_called_once()
                mock_opt.assert_called_once()
    
    def test_gpu_optimizer_initialization(self):
        """Test GpuOptimizer initialization"""
        config = {
            'gpu_target': 0.8,
            'memory_target': 0.85,
            'enable_gpu_scaling': True,
            'enable_gpu_memory_optimization': True,
            'enable_power_management': True
        }
        
        optimizer = GpuOptimizer(config)
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.gpu_monitor)
        self.assertIsNotNone(optimizer.gpu_memory_manager)
        self.assertIsNotNone(optimizer.power_manager)
        
        # Check configuration
        self.assertEqual(optimizer.config['gpu_target'], 0.8)
        self.assertEqual(optimizer.config['memory_target'], 0.85)
        self.assertTrue(optimizer.config['enable_gpu_scaling'])
        self.assertTrue(optimizer.config['enable_gpu_memory_optimization'])
        self.assertTrue(optimizer.config['enable_power_management'])
    
    def test_gpu_optimizer_optimize(self):
        """Test GPU optimization"""
        config = {
            'gpu_target': 0.8,
            'memory_target': 0.85,
            'enable_gpu_scaling': True,
            'gpu_scaling_threshold': 0.9,
            'enable_gpu_memory_optimization': True
        }
        
        optimizer = GpuOptimizer(config)
        
        # Mock device info
        device_info = self.test_device_info
        
        # Mock task
        task = self.test_optimization_task
        
        # Mock GPU analysis
        mock_analysis = {
            'current_gpu_usage': 75.0,
            'target_gpu_usage': 80.0,
            'current_memory_usage': 60.0,
            'target_memory_usage': 85.0,
            'gpu_pressure': 'medium',
            'power_limit': 150,
            'temperature': 65.0,
            'bottlenecks': ['power_limit', 'memory_usage'],
            'optimization_potential': 20.0
        }
        
        # Mock GPU optimization
        mock_optimization = {
            'power_scaling': 180,
            'memory_optimization': 85.0,
            'power_management': 'balanced'
        }
        
        with patch.object(optimizer, '_analyze_gpu') as mock_analyze:
            with patch.object(optimizer, '_perform_gpu_optimization') as mock_opt:
                mock_analyze.return_value = mock_analysis
                mock_opt.return_value = mock_optimization
                
                # Optimize GPU
                result = optimizer.optimize(device_info, task)
                
                self.assertIsNotNone(result)
                self.assertTrue(result['success'])
                self.assertEqual(result['metric'], 'gpu_utilization')
                self.assertEqual(result['before'], 75.0)
                self.assertEqual(result['after'], 80.0)
                self.assertIn('optimizations', result)
                self.assertIn('resource_usage', result)
                self.assertIn('processing_time', result)
                
                # Verify calls were made
                mock_analyze.assert_called_once()
                mock_opt.assert_called_once()
    
    def test_resource_scheduler_initialization(self):
        """Test ResourceScheduler initialization"""
        config = {
            'enable_priority_based': True,
            'enable_time_based': True,
            'enable_load_based': True,
            'max_concurrent_tasks': 5,
            'task_timeout': 300,
            'enable_preemption': True,
            'priority_levels': ['critical', 'high', 'medium', 'low']
        }
        
        scheduler = ResourceScheduler(config)
        
        self.assertIsNotNone(scheduler)
        self.assertIsNotNone(scheduler.config)
        self.assertIsNotNone(scheduler.task_queue)
        self.assertIsNotNone(scheduler.resource_allocator)
        self.assertIsNotNone(scheduler.task_monitor)
        
        # Check configuration
        self.assertTrue(scheduler.config['enable_priority_based'])
        self.assertTrue(scheduler.config['enable_time_based'])
        self.assertTrue(scheduler.config['enable_load_based'])
        self.assertEqual(scheduler.config['max_concurrent_tasks'], 5)
        self.assertEqual(scheduler.config['task_timeout'], 300)
        self.assertTrue(scheduler.config['enable_preemption'])
    
    def test_resource_scheduler_schedule_task(self):
        """Test task scheduling"""
        config = {
            'enable_priority_based': True,
            'max_concurrent_tasks': 5,
            'enable_preemption': True,
            'priority_levels': ['critical', 'high', 'medium', 'low']
        }
        
        scheduler = ResourceScheduler(config)
        
        # Mock task
        task = self.test_optimization_task
        
        # Mock resource allocation
        mock_allocation = {
            'success': True,
            'task_id': 'task_001',
            'allocated_resources': {
                'cpu': 2,
                'gpu': 1,
                'memory': 2048
            },
            'estimated_start_time': datetime.now(),
            'estimated_duration': 60
        }
        
        with patch.object(scheduler.resource_allocator, 'allocate_resources') as mock_alloc:
            mock_alloc.return_value = mock_allocation
            
            # Schedule task
            result = scheduler.schedule_task(task)
            
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertEqual(result['task_id'], 'task_001')
            self.assertIn('allocated_resources', result)
            self.assertIn('estimated_start_time', result)
            self.assertIn('estimated_duration', result)
            
            # Verify resource allocation was called
            mock_alloc.assert_called_once()
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        # Mock optimization response
        mock_response = {
            'success': True,
            'metric': 'fps',
            'before': 15.0,
            'after': 32.0,
            'improvement': 113.3
        }
        
        with patch.object(optimizer.performance_optimizer, 'optimize') as mock_opt:
            mock_opt.return_value = mock_response
            
            # Run benchmark
            benchmark_results = optimizer.benchmark_device_optimization(
                self.test_device_info,
                self.test_optimization_task
            )
            
            self.assertIsNotNone(benchmark_results)
            self.assertIn('total_processing_time', benchmark_results)
            self.assertIn('optimizer_performance', benchmark_results)
            self.assertIn('throughput', benchmark_results)
            self.assertIn('memory_usage', benchmark_results)
            
            # Check benchmark values
            self.assertIsInstance(benchmark_results['total_processing_time'], float)
            self.assertIsInstance(benchmark_results['throughput'], float)
            self.assertIsInstance(benchmark_results['memory_usage'], float)
            
            # Check optimizer performance
            self.assertIsInstance(benchmark_results['optimizer_performance'], dict)
            self.assertIn('performance', benchmark_results['optimizer_performance'])
    
    def test_error_handling(self):
        """Test error handling"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        # Test resource allocation error
        with patch.object(optimizer.scheduler.resource_allocator, 'allocate_resources') as mock_alloc:
            mock_alloc.side_effect = Exception("Resource allocation failed")
            
            # Schedule task with error
            task = self.test_optimization_task
            result = optimizer.scheduler.schedule_task(task)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Resource allocation failed', result['error'])
        
        # Test optimization error
        with patch.object(optimizer.performance_optimizer, 'optimize') as mock_opt:
            mock_opt.side_effect = Exception("Performance optimization failed")
            
            # Optimize device with error
            result = optimizer.optimize_device(
                self.test_device_info,
                self.test_optimization_task
            )
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Performance optimization failed', result['error'])
    
    def test_cache_functionality(self):
        """Test optimization caching"""
        optimizer = EnhancedDeviceOptimizer(self.test_config)
        
        # Mock optimization response
        mock_response = {
            'success': True,
            'metric': 'fps',
            'before': 15.0,
            'after': 32.0,
            'improvement': 113.3
        }
        
        with patch.object(optimizer.performance_optimizer, 'optimize') as mock_opt:
            mock_opt.return_value = mock_response
            
            # First optimization (should hit optimizers)
            result1 = optimizer.optimize_device(
                self.test_device_info,
                self.test_optimization_task
            )
            
            # Second optimization (should use cache)
            result2 = optimizer.optimize_device(
                self.test_device_info,
                self.test_optimization_task
            )
            
            # Verify cache was used
            self.assertTrue(hasattr(optimizer, 'optimization_cache'))
            self.assertTrue(len(optimizer.optimization_cache) > 0)
            
            # Results should be the same
            self.assertEqual(result1['success'], result2['success'])
            self.assertEqual(result1['metric'], result2['metric'])
            
            # Optimizer should be called only once (second call uses cache)
            mock_opt.assert_called_once()

if __name__ == '__main__':
    unittest.main()