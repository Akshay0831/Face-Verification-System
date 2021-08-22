"""
Test suite for Face Verification System
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

__all__ = []

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Core functionality tests
    suite.addTest(unittest.makeSuite(TestPluginManager))
    suite.addTest(unittest.makeSuite(TestDeviceManager))
    
    # Enhanced features tests
    suite.addTest(unittest.makeSuite(TestEnhancedDetections))
    suite.addTest(unittest.makeSuite(TestEnhancedRecognition))
    suite.addTest(unittest.makeSuite(TestEnhancedLiveness))
    suite.addTest(unittest.makeSuite(TestEnhancedNotifications))
    suite.addTest(unittest.makeSuite(TestEnhancedDevices))
    
    # Interface tests
    suite.addTest(unittest.makeSuite(TestWebInterface))
    suite.addTest(unittest.makeSuite(TestMobileInterface))
    suite.addTest(unittest.makeSuite(TestDesktopInterface))
    
    # Enterprise features tests
    suite.addTest(unittest.makeSuite(TestAnalyticsDashboard))
    suite.addTest(unittest.makeSuite(TestBusinessIntelligence))
    suite.addTest(unittest.makeSuite(TestScalabilityLayer))
    
    # System integration tests
    suite.addTest(unittest.makeSuite(TestSystemIntegration))
    
    return suite

if __name__ == '__main__':
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)