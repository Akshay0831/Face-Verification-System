"""Command Line Interface for modular face verification system."""

import argparse
import os
import sys
import signal
import cv2
import time
import logging
from typing import Optional

from . import FaceVerificationSystem, DeviceType, PerformanceMode
from .utils import get_logger

logger = get_logger('cli')


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down...")
    sys.exit(0)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Modular Face Verification System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config/system.yaml
  %(prog)s --camera 0 --mode high_speed
  %(prog)s --enroll --user_id john_doe
  %(prog)s --info
  %(prog)s --test-image test.jpg
        """
    )
    
    # Basic options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='face_verification_system/config/system.yaml',
        help='Path to configuration file (default: face_verification_system/config/system.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level (default: INFO)'
    )
    
    # System operations
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument(
        '--info',
        action='store_true',
        help='Display system information'
    )
    
    group.add_argument(
        '--reload-plugins',
        action='store_true',
        help='Reload all plugins'
    )
    
    # Camera operations
    group.add_argument(
        '--camera', '-C',
        type=int,
        default=0,
        help='Camera source index or RTSP URL (default: 0)'
    )
    
    group.add_argument(
        '--test-image',
        type=str,
        help='Test face detection on an image file'
    )
    
    # System settings
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['standard', 'high_speed', 'ultra_high'],
        help='Performance mode'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'raspberry_pi', 'windows', 'linux', 'android'],
        help='Force specific device type'
    )
    
    # User operations
    parser.add_argument(
        '--enroll',
        action='store_true',
        help='Enroll a new user'
    )
    
    parser.add_argument(
        '--user-id',
        type=str,
        help='User ID for enrollment operations'
    )
    
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Directory containing user images for enrollment'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for output files (default: output)'
    )
    
    parser.add_argument(
        '--show-output',
        action='store_true',
        help='Show visual output in window'
    )
    
    # Testing options
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run system tests'
    )
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup logging
    if args.log_level:
        get_logger().setLevel(getattr(logging, args.log_level.upper()))
    
    try:
        # Initialize system
        logger.info("Initializing Face Verification System...")
        system = FaceVerificationSystem(args.config)
        
        if not system.initialize():
            logger.error("Failed to initialize system")
            sys.exit(1)
        
        logger.info("System initialized successfully")
        
        # Handle different commands
        if args.info:
            display_system_info(system)
        
        elif args.reload_plugins:
            logger.info("Reloading plugins...")
            if system.reload_plugins():
                logger.info("Plugins reloaded successfully")
            else:
                logger.error("Failed to reload plugins")
        
        elif args.enroll:
            handle_enrollment(system, args)
        
        elif args.test_image:
            test_image_processing(system, args.test_image, args.show_output)
        
        elif args.camera is not None:
            start_camera_capture(system, args.camera, args.show_output)
        
        elif args.test:
            run_tests(system)
        
        else:
            # Default: show help
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Shutting down due to user request")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        if 'system' in locals():
            system.shutdown()
            logger.info("System shutdown complete")


def display_system_info(system: FaceVerificationSystem):
    """Display comprehensive system information"""
    print("=== Face Verification System Information ===")
    print()
    
    # System info
    info = system.get_system_info()
    print("System Status:")
    print(f"  Initialized: {info['initialized']}")
    print(f"  Running: {info['running']}")
    print(f"  Device Type: {info['device_type']}")
    print(f"  Performance Mode: {info['performance_mode']}")
    print()
    
    # System resources
    resources = info['system_resources']
    if resources:
        print("System Resources:")
        for key, value in resources.items():
            print(f"  {key}: {value}")
        print()
    
    # Plugin information
    plugins_info = info['plugins_loaded']
    print(f"Plugins Loaded: {plugins_info['total_loaded']}")
    print("Plugin Details:")
    for plugin in plugins_info['plugins']:
        print(f"  - {plugin['name']} v{plugin['version']} ({plugin['type']})")
        if plugin.get('description'):
            print(f"    {plugin['description']}")
        if plugin.get('author'):
            print(f"    Author: {plugin['author']}")
    print()
    
    # Metrics
    metrics = system.get_metrics()
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def handle_enrollment(system: FaceVerificationSystem, args):
    """Handle user enrollment operations"""
    logger = logging.getLogger(__name__)
    
    if not args.user_id:
        logger.error("User ID required for enrollment")
        return
    
    print(f"Enrolling user: {args.user_id}")
    
    if args.images_dir:
        # Batch enrollment from directory
        if not os.path.exists(args.images_dir):
            logger.error(f"Images directory not found: {args.images_dir}")
            return
        
        enrolled_count = 0
        image_files = [f for f in os.listdir(args.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(image_files)} images for enrollment")
        
        for image_file in image_files:
            image_path = os.path.join(args.images_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Preprocess image
                if image.shape[0] < 224 or image.shape[1] < 224:
                    image = cv2.resize(image, (224, 224))
                
                if system.enroll_user(args.user_id, image):
                    enrolled_count += 1
                    print(f"  ✓ Enrolled from: {image_file}")
                else:
                    print(f"  ✗ Failed to enroll from: {image_file}")
        
        print(f"Enrollment complete: {enrolled_count}/{len(image_files)} images processed")
    
    else:
        # Single image enrollment (camera capture)
        print("Press SPACE to capture image, ESC to cancel")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Show frame
            cv2.imshow('Enrollment - Press SPACE to capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                # Capture and enroll
                if system.enroll_user(args.user_id, frame):
                    print("✓ User enrolled successfully")
                    captured = True
                else:
                    print("✗ Failed to enroll user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not captured:
            logger.info("Enrollment cancelled")


def test_image_processing(system: FaceVerificationSystem, image_path: str, show_output: bool):
    """Test face detection on an image file"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return
    
    print(f"Testing image processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Failed to load image")
        return
    
    # Process the image
    result = system.process_frame(image)
    
    # Display results
    print("\nProcessing Results:")
    print(f"  Detections: {len(result['detections'])}")
    print(f"  Recognitions: {len(result['recognitions'])}")
    print(f"  Intruders: {len(result['intruders'])}")
    print(f"  Processing Time: {result['processing_time']:.3f}s")
    
    # Show detection details
    for i, detection in enumerate(result['detections']):
        print(f"\nDetection {i+1}:")
        print(f"  BBox: {detection.bbox}")
        print(f"  Confidence: {detection.confidence:.3f}")
        
        # Draw detection box
        x, y, w, h = detection.bbox
        color = (0, 255, 0)  # Green for detection
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Show recognition result
        if i < len(result['recognitions']):
            recognition = result['recognitions'][i]
            text = f"User: {recognition.user_id}" if recognition.user_id else "Unknown"
            confidence_text = f" ({recognition.confidence:.2f})"
            cv2.putText(image, text + confidence_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Show intruder information
    if result['intruders']:
        for i, intruder in enumerate(result['intruders']):
            print(f"\nIntruder {i+1}:")
            print(f"  Confidence: {intruder['confidence']:.3f}")
            x, y, w, h = intruder['bbox']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for intruder
            cv2.putText(image, "INTRUDER", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save output image
    output_path = os.path.join('output', f'test_result_{os.path.basename(image_path)}')
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"\nOutput saved to: {output_path}")
    
    # Show image if requested
    if show_output:
        print("\nDisplaying result image (press any key to close)")
        cv2.imshow('Processing Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def start_camera_capture(system: FaceVerificationSystem, camera_source: int, show_output: bool):
    """Start real-time camera capture"""
    logger = logging.getLogger(__name__)
    
    print(f"Starting camera capture from source: {camera_source}")
    print("Press ESC to stop, R to reload plugins, SPACE to capture enrollment image")
    
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        logger.error(f"Failed to open camera source: {camera_source}")
        return
    
    last_notification_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            # Process the frame
            result = system.process_frame(frame)
            
            # Draw results on frame
            for detection in result['detections']:
                x, y, w, h = detection.bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show confidence
                conf_text = f"{detection.confidence:.2f}"
                cv2.putText(frame, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show intruder alerts
            current_time = time.time()
            if result['intruders'] and current_time - last_notification_time > 30:  # 30 second cooldown
                print("🚨 INTRUDER DETECTED! 🚨")
                last_notification_time = current_time
            
            # Show processing info
            processing_text = f"Time: {result['processing_time']:.3f}s"
            cv2.putText(frame, processing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show instructions
            instructions = "ESC: Stop | R: Reload | SPACE: Enroll"
            cv2.putText(frame, instructions, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            if show_output:
                cv2.imshow('Face Verification', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Stopping camera capture...")
                break
            elif key == ord('r') or key == ord('R'):  # R
                print("Reloading plugins...")
                system.reload_plugins()
            elif key == ord(' '):  # SPACE
                # Quick enrollment mode
                print("Quick enrollment mode activated")
                print("Press SPACE to capture image, ESC to cancel")
                enrollment_active = True
                while enrollment_active:
                    ret, enroll_frame = cap.read()
                    if ret:
                        cv2.imshow('Enrollment - Press SPACE to capture', enroll_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            print("Enrollment cancelled")
                            break
                        elif key == 32:
                            # Capture frame
                            system.enroll_user("quick_user", enroll_frame)
                            print("User enrolled from quick capture")
                            enrollment_active = False
                
                cv2.destroyWindow('Enrollment - Press SPACE to capture')
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_tests(system: FaceVerificationSystem):
    """Run system tests"""
    logger = logging.getLogger(__name__)
    print("Running system tests...")
    
    try:
        # Import test module
        import unittest
        from .tests.test_core_system import TestFaceVerificationSystem, TestCoreComponents, TestSystemIntegration
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestFaceVerificationSystem))
        suite.addTest(unittest.makeSuite(TestCoreComponents))
        suite.addTest(unittest.makeSuite(TestSystemIntegration))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        print(f"\nTests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
        
        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)
        
    except ImportError as e:
        logger.error(f"Failed to import tests: {e}")
        print("Tests not available. Run 'pip install -e .[dev]' to install test dependencies.")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()