import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional
from pathlib import Path

class OptimizedDistractionDetector:
    """
    CPU-optimized distraction detection
    
    Components:
    1. Object Detection (phone, food, etc.) - MobileNetV3-SSD INT8
    2. Head Pose Estimation - Facial landmarks-based (Placeholder for now)
    
    Performance: 30-40ms (CPU)
    """
    
    # Distraction object classes (COCO indices mapped to names, simplified)
    # Note: You'll need to match these to your specific model's class map
    # This is a sample mapping for standard COCO
    DISTRACTION_CLASSES = {
        'cell phone': 0.9,  # High distraction
        'bottle': 0.4,
        'cup': 0.4,
        'fork': 0.6,
        'knife': 0.8,
        'spoon': 0.5,
        'bowl': 0.5,
        'book': 0.8
    }
    
    # COCO Class names (partial list for mapping)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]

    def _get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.COCO_CLASSES):
            return self.COCO_CLASSES[class_id]
        return "unknown"
    
    def __init__(self):
        # Load ONNX model (INT8 quantized)
        model_path = Path("models/mobilenetv3_ssd_int8.onnx")
        
        self.session = None
        if model_path.exists():
            try:
                # Create ONNX Runtime session (CPU optimized)
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4  # Use 4 CPU threads
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.session = ort.InferenceSession(
                    str(model_path),
                    sess_options,
                    providers=['CPUExecutionProvider']
                )
                
                # Get input name
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                print(f"Warning: Failed to load Distraction model: {e}")
        else:
            print(f"Warning: Distraction model not found at {model_path}")
        
        # Head pose state
        self.looking_away_frames = 0
        self.looking_down_frames = 0
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SSD"""
        # Resize to 300x300 (SSD input)
        img = cv2.resize(image, (300, 300))
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_objects(self, image: np.ndarray, min_confidence: float = 0.5) -> List[Dict]:
        """Detect distraction objects"""
        if self.session is None:
            return []

        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Parse outputs
        # Format depends on the specific ONNX export. 
        # Standard SSD output: [boxes, classes, scores] or [batch, num_det, 6]
        # Assuming typical output: boxes, classes, scores separately
        
        # Adjust based on your specific model export. 
        # This assumes: outputs[0]=boxes, outputs[1]=classes, outputs[2]=scores
        if len(outputs) >= 3:
            boxes = outputs[0][0]  # Assuming batch size 1
            classes = outputs[1][0]
            scores = outputs[2][0]
        else:
             # Fallback/Debug for different output format
             return []
        
        h, w = image.shape[:2]
        detections = []
        
        for box, cls, score in zip(boxes, classes, scores):
            if score < min_confidence:
                continue
            
            class_name = self._get_class_name(int(cls))
            
            # Only keep distraction objects
            if class_name in self.DISTRACTION_CLASSES:
                # Box format might be [y1, x1, y2, x2] or [x1, y1, x2, y2]. 
                # Checking typically TF/Mobilenet SSD is [y1, x1, y2, x2] normalized
                y1, x1, y2, x2 = box
                
                # Convert to pixel coordinates
                y1 = int(y1 * h)
                x1 = int(x1 * w)
                y2 = int(y2 * h)
                x2 = int(x2 * w)
                
                detections.append({
                    'class': class_name,
                    'confidence': float(score),
                    'bbox': [x1, y1, x2 - x1, y2 - y1], # x, y, w, h
                    'distraction_score': self.DISTRACTION_CLASSES[class_name]
                })
        
        return detections
    
    def detect_head_pose(self, landmarks: np.ndarray) -> Dict:
        """
        Estimate head pose from landmarks (Placeholder)
        In a full implementation, you'd use PnP/SolvePnP with a generic 3D face model
        """
        # Placeholder logic
        # if landmarks is not None:
        #    ...
        return {'pitch': 0, 'yaw': 0, 'roll': 0, 'is_looking_away': False}
