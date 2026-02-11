import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class OptimizedRoadCrackDetector:
    """
    CPU-optimized road crack detection using ENet (Segmentation) or OpenCV
    
    Performance: 
    - ENet (INT8): 45-55ms
    - OpenCV: 8-10ms
    """
    
    def __init__(self, use_enet: bool = True):
        self.use_enet = use_enet
        self.enet_session = None
        
        if use_enet:
            model_path = Path("models/enet_road_crack_int8.onnx")
            if model_path.exists():
                try:
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = 4
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    self.enet_session = ort.InferenceSession(
                        str(model_path),
                        sess_options,
                        providers=['CPUExecutionProvider']
                    )
                    self.input_name = self.enet_session.get_inputs()[0].name
                except Exception as e:
                    print(f"Warning: Failed to load ENet model: {e}")
            else:
                 # Fallback to OpenCV if enabled or warn
                 print(f"Warning: ENet model not found at {model_path}, using OpenCV fallback.")
                 self.enet_session = None
                 
    def detect_opencv(self, image: np.ndarray) -> Dict:
        """
        Fast crack detection using image processing
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cracks_found = []
        total_crack_length = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum area filter
                # Fit ellipse or bounding rect
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Cracks are often long and thin
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    cracks_found.append(cnt)
                    total_crack_length += cv2.arcLength(cnt, True)
                    
        return {
            'has_cracks': len(cracks_found) > 0,
            'crack_count': len(cracks_found),
            'severity': 'high' if total_crack_length > 500 else 'low',
            'method': 'opencv'
        }
        
    def detect_enet(self, image: np.ndarray) -> Dict:
        """
        Accurate crack segmentation using ENet
        """
        if self.enet_session is None:
             return self.detect_opencv(image)
             
        # Preprocess
        input_size = (512, 512) # ENet standard
        img = cv2.resize(image, input_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # Inference
        outputs = self.enet_session.run(None, {self.input_name: img})
        mask = outputs[0][0] # Assuming batch 0, channel 0 (binary mask)
        
        # Calculate crack percentage
        crack_pixels = np.sum(mask > 0.5)
        total_pixels = mask.size
        severity_score = crack_pixels / total_pixels
        
        return {
             'has_cracks': severity_score > 0.01, # Threshold 1%
             'severity_score': float(severity_score),
             'method': 'enet'
        }
        
    def detect(self, image: np.ndarray) -> Dict:
         """Main detection method"""
         if self.enet_session is not None:
              return self.detect_enet(image)
         else:
              return self.detect_opencv(image)
