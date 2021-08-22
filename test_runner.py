#!/usr/bin/env python3
"""
Simple test runner for the face verification system.
"""

import sys
import os
import unittest

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def run_core_tests():
    """Run tests for core modules only"""
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Import core test modules
    try:
        from tests.core.test_plugin_manager import TestPluginManager
        suite.addTest(unittest.makeSuite(TestPluginManager))
        print("✓ Plugin Manager tests loaded")
    except Exception as e:
        print(f"✗ Failed to load Plugin Manager tests: {e}")
    
    try:
        from tests.core.test_device_manager import TestDeviceManager  
        suite.addTest(unittest.makeSuite(TestDeviceManager))
        print("✓ Device Manager tests loaded")
    except Exception as e:
        print(f"✗ Failed to load Device Manager tests: {e}")
    
    # Add tests for enhanced modules if they exist
    test_modules = [
        ('tests.enhanced.test_recognition', 'Enhanced Recognition'),
        ('tests.enhanced.test_liveness', 'Enhanced Liveness'),
        ('tests.enhanced.test_notifications', 'Enhanced Notifications'),
        ('tests.enhanced.test_devices', 'Enhanced Devices'),
        ('tests.enhanced.test_detections', 'Enhanced Detections')
    ]
    
    for module_name, test_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[test_name])
            test_class = getattr(module, f'Test{test_name.replace(" ", "")}')
            suite.addTest(unittest.makeSuite(test_class))
            print(f"✓ {test_name} tests loaded")
        except Exception as e:
            print(f"✗ Failed to load {test_name} tests: {e}")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = run_core_tests()
    sys.exit(exit_code)