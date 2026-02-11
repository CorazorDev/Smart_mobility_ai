import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from ml_backend.orchestrator import ModelOrchestrator

def test_pipeline():
    print("Testing Pipeline Integration...")
    
    # Create dummy image (black image)
    dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
    
    # Draw a dummy face (white rectangle) to trigger face detection (Haar might not pick it up, but worth a try or just rely on dry run)
    # Actually Haar needs features, a black rect won't work.
    # But we can test that it runs without crashing.
    cv2.rectangle(dummy_image, (100, 100), (300, 300), (255, 255, 255), -1)
    
    # Create dummy sensor data
    dummy_sensor_data = {
        'imu': {
            'acc_x': 0.1, 'acc_y': 0.2, 'acc_z': 9.8,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0
        }
    }
    
    try:
        orchestrator = ModelOrchestrator()
        print("Orchestrator initialized successfully.")
        
        # Test inference
        results = orchestrator.process_frame(dummy_image, dummy_sensor_data)
        print("Inference successful.")
        print("Results:", results)
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
