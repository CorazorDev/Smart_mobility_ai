from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import json
from typing import Dict, Any

from ml_backend.orchestrator import ModelOrchestrator

app = FastAPI(title="Smart Mobility AI - Optimized CPU Backend")

# Global Orchestrator Instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    orchestrator = ModelOrchestrator()

@app.get("/")
def read_root():
    return {"status": "online", "system": "Optimized CPU Backend"}

@app.post("/analyze")
async def analyze_frame(
    file: UploadFile = File(...),
    sensor_data: str = Body(default="{}") # JSON string
):
    """
    Analyze a single frame + sensor data
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Model system not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        # Parse sensor data
        try:
            sensor_json = json.loads(sensor_data)
        except json.JSONDecodeError:
            sensor_json = {}
            
        # Validate sensor data keys if necessary
        # ...
        
        # Process
        results = orchestrator.process_frame(image, sensor_json)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("ml_backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
