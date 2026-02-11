import numpy as np
from typing import Dict, Any
import time

# Import models
from ml_backend.models.driver_safety.face_detection_optimized import OptimizedFaceDetector
from ml_backend.models.driver_safety.fatigue_detection_optimized import OptimizedFatigueDetector
from ml_backend.models.driver_safety.distraction_detection_optimized import OptimizedDistractionDetector
from ml_backend.models.driver_safety.drunk_detection_optimized import OptimizedDrunkDetector

from ml_backend.models.road_safety.pothole_detection_optimized import OptimizedPotholeDetector
from ml_backend.models.road_safety.road_crack_optimized import OptimizedRoadCrackDetector
from ml_backend.models.road_safety.road_roughness import OptimizedRoadRoughnessDetector

class ModelOrchestrator:
    """
    Manages the execution of all optimized models.
    Implements cascading logic to save CPU cycles.
    """
    
    def __init__(self):
        print("Initializing Optimized Models...")
        # Driver Safety
        self.face_detector = OptimizedFaceDetector()
        self.fatigue_detector = OptimizedFatigueDetector()
        self.distraction_detector = OptimizedDistractionDetector()
        self.drunk_detector = OptimizedDrunkDetector()
        
        # Road Safety
        self.pothole_detector = OptimizedPotholeDetector()
        self.crack_detector = OptimizedRoadCrackDetector()
        self.roughness_detector = OptimizedRoadRoughnessDetector()
        
        print("Models Initialized.")
        
    def process_frame(self, image: np.ndarray, sensor_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single frame through the pipeline.
        
        Cascading Logic:
        1. Face Detection -> If no face, skip fatigue/distraction
        2. Road models can run in parallel (or sequential on single thread)
        """
        results = {}
        start_time = time.time()
        
        # --- Driver Safety ---
        # 1. Face Detection
        faces = self.face_detector.detect(image)
        results['faces'] = faces
        
        driver_detected = len(faces) > 0
        
        if driver_detected:
            # Get primary driver (largest face)
            primary_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
            bbox = primary_face['bbox']
            
            # 2. Fatigue Detection (Only if face present)
            fatigue_result = self.fatigue_detector.detect(image, bbox)
            results['fatigue'] = fatigue_result
            
            # 3. Distraction Detection (Only if driver present)
            # Optimization: Run every N frames or if head pose suggests looking away
            distraction_result = self.distraction_detector.detect_objects(image)
            results['distraction'] = distraction_result
        else:
            results['fatigue'] = None
            results['distraction'] = None
            
        # 4. Drunk Detection (IMU based)
        if sensor_data and 'imu' in sensor_data:
            self.drunk_detector.add_imu_reading(sensor_data['imu'])
            # Predict only occasionally or on request
            results['drunk_detection'] = self.drunk_detector.predict()
            
        # --- Road Safety ---
        # Optimization: These can be run on a separate thread or process in production
        # For now, sequential
        
        # 5. Pothole Detection
        potholes = self.pothole_detector.detect(image)
        results['potholes'] = potholes
        
        # 6. Crack Detection
        # Optimization: Run only if road surface is detected (omitted for now)
        cracks = self.crack_detector.detect(image)
        results['cracks'] = cracks
        
        # 7. Roughness (IMU based)
        if sensor_data and 'imu' in sensor_data:
             self.roughness_detector.add_imu_reading(sensor_data['imu'])
             results['roughness'] = self.roughness_detector.predict()
             
        inference_time = (time.time() - start_time) * 1000
        results['inference_time_ms'] = inference_time
        
        return results
