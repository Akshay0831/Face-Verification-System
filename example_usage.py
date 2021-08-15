"""Face verification system usage example"""

import os
import sys
import cv2
import numpy as np
import logging

# Add the current directory to the path to import the face verification system
sys.path.append(os.path.dirname(__file__))

from face_verification_system import FaceVerificationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main example usage function"""
    print("=== Modular Face Verification System Example ===")
    print()
    
    # Initialize the system
    print("1. Initializing Face Verification System...")
    
    # Create system instance
    # Use the default configuration file
    system = FaceVerificationSystem()
    
    # Initialize the system
    if not system.initialize():
        print("Failed to initialize the system!")
        return
    
    print("✓ System initialized successfully")
    
    # Get system information
    print("\n2. System Information:")
    system_info = system.get_system_info()
    print(f"   Device Type: {system_info['device_type']}")
    print(f"   Performance Mode: {system_info['performance_mode']}")
    print(f"   Initialized: {system_info['initialized']}")
    print(f"   Plugins Loaded: {len(system_info['plugins_loaded']['plugins'])}")
    
    # Show loaded plugins
    print("\n3. Loaded Plugins:")
    for plugin in system_info['plugins_loaded']['plugins']:
        print(f"   - {plugin['name']} v{plugin['version']} ({plugin['type']})")
        if plugin.get('description'):
            print(f"     {plugin['description']}")
    
    # Load a test image if available
    test_image_path = "test.jpg"
    if os.path.exists(test_image_path):
        print(f"\n4. Testing with image: {test_image_path}")
        
        # Load and test the image
        image = cv2.imread(test_image_path)
        if image is not None:
            print("   Image loaded successfully")
            
            # Process the frame
            print("   Processing frame...")
            result = system.process_frame(image)
            
            # Display results
            print(f"   - Detections found: {len(result['detections'])}")
            print(f"   - Recognitions: {len(result['recognitions'])}")
            print(f"   - Intruders detected: {len(result['intruders'])}")
            print(f"   - Processing time: {result['processing_time']:.3f}s")
            
            # Show detection details
            for i, detection in enumerate(result['detections']):
                print(f"   Detection {i+1}:")
                print(f"     BBox: {detection.bbox}")
                print(f"     Confidence: {detection.confidence:.3f}")
            
            # Show recognition results
            for i, recognition in enumerate(result['recognitions']):
                if recognition.user_id:
                    print(f"   Recognized {i+1}: User {recognition.user_id} (confidence: {recognition.confidence:.3f})")
                else:
                    print(f"   Recognition {i+1}: Unknown person (confidence: {recognition.confidence:.3f})")
            
            # Show intruder information
            if result['intruders']:
                print("   Intruders detected!")
                for i, intruder in enumerate(result['intruders']):
                    print(f"   Intruder {i+1}: Confidence {intruder['confidence']:.3f}")
        else:
            print("   Failed to load test image")
    
    # Start camera capture if requested
    start_camera = input("\n5. Start camera capture? (y/n): ").lower().strip()
    if start_camera == 'y':
        print("Starting camera capture...")
        print("Press ESC to stop")
        system.start_camera_capture()
    
    # Display metrics
    print("\n6. System Metrics:")
    metrics = system.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Shutdown the system
    print("\n7. Shutting down...")
    system.shutdown()
    print("✓ System shutdown complete")

def demo_enrollment():
    """Demo user enrollment functionality"""
    print("\n=== User Enrollment Demo ===")
    
    # Initialize system
    system = FaceVerificationSystem()
    if not system.initialize():
        print("Failed to initialize system!")
        return
    
    # For demo purposes, create a dummy face image
    # In practice, this would come from camera or uploaded image
    dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_face.fill(128)  # Gray image for demo
    
    # Enroll a test user
    user_id = "demo_user_001"
    print(f"Enrolling user: {user_id}")
    
    success = system.enroll_user(user_id, dummy_face)
    if success:
        print("✓ User enrolled successfully")
    else:
        print("✗ Failed to enroll user")
    
    # Shutdown
    system.shutdown()

if __name__ == "__main__":
    try:
        # Run main example
        main()
        
        # Optionally run enrollment demo
        run_enrollment = input("\nRun enrollment demo? (y/n): ").lower().strip()
        if run_enrollment == 'y':
            demo_enrollment()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        logging.error(f"Example usage error: {e}", exc_info=True)