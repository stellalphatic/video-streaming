import uvicorn
import logging
import asyncio
import os
import time
import gc
import aiohttp
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from typing import Optional, Dict, Any
import sys
import json
from pathlib import Path
from collections import deque

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default_video_service_key_change_in_production")
DEVICE = "cpu"
TARGET_FPS = 15  # Reduced FPS for Cloud Run
AUDIO_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8080))

# FastAPI App
app = FastAPI(title="Real-time Avatar Video Service", version="10.0.0")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global state
video_generator: Optional[object] = None
pipeline_ready = asyncio.Event()
active_sessions: Dict[str, Any] = {}

class SimpleVideoGenerator:
    """Optimized simple video generator for Cloud Run"""
    
    def __init__(self):
        self.device = "cpu"
        logger.info("Initialized Simple Video Generator")
        
    def prepare_avatar(self, image: np.ndarray):
        """Prepare avatar for generation"""
        try:
            # Resize to optimal size for Cloud Run
            image = cv2.resize(image, (384, 384))  # Smaller size for better performance
            
            # Simple face detection for mouth area
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            return {
                'image': image,
                'gray': gray,
                'height': image.shape[0],
                'width': image.shape[1],
                'mouth_region': (int(image.shape[1] * 0.35), int(image.shape[0] * 0.65), 
                               int(image.shape[1] * 0.65), int(image.shape[0] * 0.85))
            }
        except Exception as e:
            logger.error(f"Error preparing avatar: {e}")
            return None
    
    def generate_frame(self, avatar_data: dict, audio_data: np.ndarray, is_speaking: bool = False):
        """Generate frame with optimized mouth movement"""
        try:
            frame = avatar_data['image'].copy()
            
            if is_speaking and len(audio_data) > 0:
                # Calculate audio energy
                audio_energy = np.mean(np.abs(audio_data))
                
                if audio_energy > 0.005:  # Lower threshold
                    # Get mouth region
                    x1, y1, x2, y2 = avatar_data['mouth_region']
                    
                    # Create mouth movement effect
                    mouth_open = min(int(audio_energy * 3000), 12)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Draw subtle mouth opening
                    cv2.ellipse(frame, (center_x, center_y + mouth_open//2), 
                              (mouth_open + 8, mouth_open), 0, 0, 360, (40, 40, 40), -1)
                    
                    # Add slight head movement for liveliness
                    if audio_energy > 0.02:
                        # Slight random movement
                        offset_x = int((np.random.random() - 0.5) * 2)
                        offset_y = int((np.random.random() - 0.5) * 1)
                        
                        # Apply subtle transform
                        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            return avatar_data['image']

async def initialize_pipeline():
    """Initialize pipeline quickly for Cloud Run"""
    global video_generator
    
    try:
        logger.info("Initializing Simple Video Generator for Cloud Run...")
        video_generator = SimpleVideoGenerator()
        pipeline_ready.set()
        logger.info("Video pipeline ready")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        return False

class VideoSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.avatar_data: Optional[dict] = None
        self.is_prepared = False
        self.last_activity = time.time()
        self.is_speaking = False
        self.audio_buffer = deque(maxlen=20)  # Smaller buffer for Cloud Run

    async def prepare_avatar(self, image_url: str, pipeline):
        """Prepare avatar with timeout for Cloud Run"""
        logger.info(f"[{self.session_id}] Preparing avatar from URL")
        try:
            timeout = aiohttp.ClientTimeout(total=15)  # Shorter timeout
            async with aiohttp.ClientSession(timeout=timeout) as http_session:
                async with http_session.get(image_url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
            
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            # Prepare with pipeline
            self.avatar_data = pipeline.prepare_avatar(img)
            if self.avatar_data is None:
                raise ValueError("Failed to prepare avatar data")
            
            self.is_prepared = True
            logger.info(f"Avatar prepared for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to prepare avatar: {e}")
            return False

    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk with error handling"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            if len(audio_array) > 0:
                self.audio_buffer.append(audio_array)
                self.last_activity = time.time()
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")

    def get_combined_audio(self) -> np.ndarray:
        """Get combined audio optimized for Cloud Run"""
        if len(self.audio_buffer) == 0:
            return np.array([])
        
        # Use fewer chunks for better performance
        recent_chunks = list(self.audio_buffer)[-2:]
        return np.concatenate(recent_chunks) if recent_chunks else np.array([])

def verify_api_key(auth: str = Depends(api_key_header)):
    """Verify API key"""
    if not auth or auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.on_event("startup")
async def startup_event():
    """Fast startup for Cloud Run"""
    logger.info(f"Starting video service on port {PORT}")
    # Initialize pipeline immediately (no background task)
    await initialize_pipeline()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Avatar Video Service", 
        "status": "running", 
        "version": "10.0.0",
        "device": DEVICE,
        "ready": pipeline_ready.is_set()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": pipeline_ready.is_set(),
        "device": DEVICE,
        "port": PORT,
        "active_sessions": len(active_sessions),
        "torch_version": torch.__version__,
        "memory_usage": f"{torch.cuda.memory_allocated() if torch.cuda.is_available() else 0} bytes"
    }

@app.post("/init-stream")
async def init_stream(request: dict, _: None = Depends(verify_api_key)):
    """Initialize streaming session"""
    session_id = request.get("session_id")
    image_url = request.get("image_url")
    avatar_id = request.get("avatar_id")
    
    if not all([session_id, image_url]):
        raise HTTPException(status_code=400, detail="Missing session_id or image_url")
    
    # Wait for pipeline if not ready
    if not pipeline_ready.is_set():
        try:
            await asyncio.wait_for(pipeline_ready.wait(), timeout=10)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Service not ready")
    
    # Create session
    session = VideoSession(session_id)
    if not await session.prepare_avatar(image_url, video_generator):
        raise HTTPException(status_code=500, detail="Failed to prepare avatar")
    
    active_sessions[session_id] = session
    logger.info(f"Session initialized: {session_id}")
    
    return {
        "status": "initialized", 
        "session_id": session_id,
        "avatar_id": avatar_id,
        "timestamp": time.time()
    }

@app.post("/end-stream")
async def end_stream(request: dict, _: None = Depends(verify_api_key)):
    """End streaming session"""
    session_id = request.get("session_id")
    if session_id in active_sessions:
        del active_sessions[session_id]
        gc.collect()
        logger.info(f"Session ended: {session_id}")
    
    return {"status": "ended", "session_id": session_id}

@app.websocket("/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """Optimized WebSocket endpoint for Cloud Run"""
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session or not session.is_prepared:
        await websocket.close(code=1008, reason="Session not found or not prepared")
        return

    logger.info(f"WebSocket connected for session: {session_id}")
    
    frame_count = 0
    last_frame_time = time.time()
    
    # Send initial frame
    try:
        initial_frame = session.avatar_data['image']
        _, buffer = cv2.imencode('.jpg', initial_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        await websocket.send_bytes(buffer.tobytes())
    except Exception as e:
        logger.error(f"Error sending initial frame: {e}")
    
    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
                session.last_activity = time.time()
                
                if message['type'] == 'websocket.receive':
                    if 'bytes' in message:
                        # Handle audio data
                        audio_bytes = message['bytes']
                        if len(audio_bytes) > 0:
                            session.add_audio_chunk(audio_bytes)
                            
                    elif 'text' in message:
                        # Handle control messages
                        try:
                            control_msg = json.loads(message['text'])
                            
                            if control_msg.get('type') == 'speech_start':
                                session.is_speaking = True
                                
                            elif control_msg.get('type') == 'speech_end':
                                session.is_speaking = False
                                session.audio_buffer.clear()
                                
                            elif control_msg.get('type') == 'request_keyframe':
                                # Send current frame immediately
                                frame = session.avatar_data['image']
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                await websocket.send_bytes(buffer.tobytes())
                                
                        except json.JSONDecodeError:
                            pass
                            
            except asyncio.TimeoutError:
                # Generate and send frame on timeout (no audio received)
                pass
            
            # Generate frame regardless of audio input
            current_time = time.time()
            frame_interval = 1.0 / TARGET_FPS
            
            if current_time - last_frame_time >= frame_interval:
                try:
                    # Get audio data
                    audio_data = session.get_combined_audio()
                    
                    # Generate frame
                    frame = video_generator.generate_frame(
                        session.avatar_data, 
                        audio_data, 
                        session.is_speaking
                    )
                    
                    # Encode and send
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    await websocket.send_bytes(buffer.tobytes())
                    
                    frame_count += 1
                    last_frame_time = current_time
                    
                    # Performance logging
                    if frame_count % 50 == 0:  # Log every 50 frames
                        elapsed = current_time - (last_frame_time - (50 * frame_interval))
                        fps = 50 / elapsed if elapsed > 0 else 0
                        logger.info(f"Session {session_id}: {fps:.1f} FPS")
                        
                except Exception as e:
                    logger.error(f"Error generating/sending frame: {e}")
                    
            # Small sleep to prevent overwhelming CPU
            await asyncio.sleep(0.01)
            
            # Check for session timeout
            if time.time() - session.last_activity > 300:  # 5 minutes
                logger.info(f"Session {session_id} timed out")
                break
                    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup
        if session_id in active_sessions:
            del active_sessions[session_id]
        gc.collect()
        logger.info(f"Session {session_id} cleaned up")

if __name__ == "__main__":
    # Optimized for Cloud Run
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=PORT, 
        workers=1,
        loop="asyncio",
        access_log=False  # Disable access logs for performance
    )