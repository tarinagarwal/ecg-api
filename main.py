"""
ECG ML API - FastAPI Application
REST API and WebSocket for real-time ECG analysis
"""

import json
import asyncio
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel, Field

from ecg_processor import ECGProcessor
from diagnosis import get_diagnosis


# ============ Models ============

class ECGInput(BaseModel):
    """Input model for ECG data"""
    ecg: List[float] = Field(..., description="Array of ECG waveform values")
    sampling_rate: Optional[int] = Field(250, description="Sampling rate in Hz")


class ECGOutput(BaseModel):
    """Output model for ECG analysis"""
    p_peaks: List[int]
    q_peaks: List[int]
    r_peaks: List[int]
    s_peaks: List[int]
    t_peaks: List[int]
    heart_rate: int
    abnormalities: List[str]
    final_diagnosis: str


# ============ WebSocket Manager ============

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# ============ App Setup ============

manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("ECG ML API starting...")
    yield
    # Shutdown
    print("ECG ML API shutting down...")


app = FastAPI(
    title="ECG ML API",
    description="Real-time ECG analysis with PQRST detection and diagnosis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")


# ============ REST Endpoints ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/analyze", response_model=ECGOutput)
async def analyze_ecg(data: ECGInput):
    """
    Analyze ECG waveform data
    
    - **ecg**: Array of ECG voltage values from your device
    - **sampling_rate**: Sampling frequency in Hz (default 250)
    
    Returns PQRST peaks, heart rate, abnormalities, and diagnosis
    """
    if len(data.ecg) < 10:
        raise HTTPException(
            status_code=400, 
            detail="ECG data too short. Minimum 10 samples required."
        )
    
    # Process ECG
    processor = ECGProcessor(sampling_rate=data.sampling_rate)
    analysis = processor.analyze(data.ecg)
    
    # Get diagnosis
    result = get_diagnosis(analysis)
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "analysis",
        "data": {
            "p_peaks": result["p_peaks"],
            "q_peaks": result["q_peaks"],
            "r_peaks": result["r_peaks"],
            "s_peaks": result["s_peaks"],
            "t_peaks": result["t_peaks"],
            "heart_rate": result["heart_rate"],
            "abnormalities": result["abnormalities"],
            "final_diagnosis": result["final_diagnosis"],
            "ecg_signal": data.ecg,  # Include for visualization
            "num_beats": result["num_beats"],
            "duration_seconds": result["duration_seconds"]
        }
    })
    
    return ECGOutput(
        p_peaks=result["p_peaks"],
        q_peaks=result["q_peaks"],
        r_peaks=result["r_peaks"],
        s_peaks=result["s_peaks"],
        t_peaks=result["t_peaks"],
        heart_rate=result["heart_rate"],
        abnormalities=result["abnormalities"],
        final_diagnosis=result["final_diagnosis"]
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ECG ML API"}


# ============ WebSocket Endpoint ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ECG updates
    
    Clients can connect to receive live analysis results
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
                
                if "ecg" in payload:
                    # Process incoming ECG data
                    ecg_data = payload["ecg"]
                    sampling_rate = payload.get("sampling_rate", 250)
                    
                    processor = ECGProcessor(sampling_rate=sampling_rate)
                    analysis = processor.analyze(ecg_data)
                    result = get_diagnosis(analysis)
                    
                    # Send result back
                    await websocket.send_json({
                        "type": "analysis",
                        "data": {
                            "p_peaks": result["p_peaks"],
                            "q_peaks": result["q_peaks"],
                            "r_peaks": result["r_peaks"],
                            "s_peaks": result["s_peaks"],
                            "t_peaks": result["t_peaks"],
                            "heart_rate": result["heart_rate"],
                            "abnormalities": result["abnormalities"],
                            "final_diagnosis": result["final_diagnosis"],
                            "ecg_signal": ecg_data,
                            "num_beats": result["num_beats"],
                            "duration_seconds": result["duration_seconds"]
                        }
                    })
                    
                    # Also broadcast to other clients
                    await manager.broadcast({
                        "type": "analysis",
                        "data": {
                            "p_peaks": result["p_peaks"],
                            "q_peaks": result["q_peaks"],
                            "r_peaks": result["r_peaks"],
                            "s_peaks": result["s_peaks"],
                            "t_peaks": result["t_peaks"],
                            "heart_rate": result["heart_rate"],
                            "abnormalities": result["abnormalities"],
                            "final_diagnosis": result["final_diagnosis"],
                            "ecg_signal": ecg_data,
                            "num_beats": result["num_beats"],
                            "duration_seconds": result["duration_seconds"]
                        }
                    })
                
                elif payload.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
