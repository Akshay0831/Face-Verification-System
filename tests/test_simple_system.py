"""Simple system validation tests"""

import unittest
import os
import sys
import tempfile
import yaml
import numpy as np
from unittest.mock import Mock, patch

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from system import FaceVerificationSystem
from core import (
    DetectionResult, RecognitionResult, LivenessResult
)


class TestSimpleSystem(unittest.TestCase):
    """Test basic system functionality without plugins"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal configuration file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_data = {
            'system': {
                'name': 'Simple Test System',
                'version': '1.0.0'
            },
            'plugins': {
                'detection': {
                    'plugin': None
                },
                'recognition': {
                    'plugin': None
                },
                'liveness': {
                    'plugin': None
                },
                'notifications': {
                    'plugin': None
                }
            },
            'performance': {
                'mode': 'standard'
            }
        }
        
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()
        
        # Create system instance
        self.system = FaceVerificationSystem(self.temp_config.name)
        
        # Initialize system (will fail gracefully with no plugins)
        try:
            self.system.initialize()
        except Exception:
            pass  # Expected to fail with no plugins
        
        # Manually set initialized to True to test process_frame
        self.system.initialized = True
        
        # Manually set initialized to True to test process_frame
        self.system.initialized = True
        
        # Manually set initialized to True to test process_frame
        self.system.initialized = True
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_system_initialization(self):
        """Test basic system initialization"""
        # Reset initialized flag to test actual initialization
        self.system.initialized = False
        result = self.system.initialize()
        self.assertFalse(result)  # Should fail due to no plugins
        
    def test_system_info(self):
        """Test system information retrieval"""
        info = self.system.get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn('initialized', info)
        self.assertIn('running', info)
        self.assertIn('plugins_loaded', info)
        
    def test_empty_process_frame(self):
        """Test frame processing with no plugins"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.system.process_frame(image)
        
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('recognitions', result)
        self.assertIn('liveness_results', result)
        self.assertIn('intruders', result)
        self.assertIn('processing_time', result)
        
        # Should be empty arrays due to no plugins
        self.assertEqual(len(result['detections']), 0)
        self.assertEqual(len(result['recognitions']), 0)
        self.assertEqual(len(result['liveness_results']), 0)
        self.assertEqual(len(result['intruders']), 0)
        
    def test_metrics_tracking(self):
        """Test metrics tracking functionality"""
        # Initial metrics
        metrics = self.system.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_detections', metrics)
        
        # Update metrics
        self.system.metrics['total_detections'] = 5
        self.system.metrics['successful_recognitions'] = 3
        
        updated_metrics = self.system.get_metrics()
        self.assertEqual(updated_metrics['total_detections'], 5)
        self.assertEqual(updated_metrics['successful_recognitions'], 3)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Reset initialized flag for error testing
        self.system.initialized = False
        
        # Test with None
        result = self.system.process_frame(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})  # Should return empty dict when not initialized


if __name__ == '__main__':
    unittest.main(verbosity=2)