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
from typing import Optional
import sys
from pathlib import Path

# Add MuseTalk to Python path
musetalk_path = Path(__file__).parent / "MuseTalk"
if musetalk_path.exists():
    sys.path.insert(0, str(musetalk_path))

try:
    # Import MuseTalk components
    from musetalk.utils.utils import load_all_model
    from musetalk.pipelines.pipeline_musetalk import MuseTalkPipeline
    MUSETALK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MuseTalk not available: {e}")
    MUSETALK_AVAILABLE = False

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default_video_service_key_change_in_production")
DEVICE = "cpu"  # Force CPU for Cloud Run (no GPU support)
TARGET_FPS = 25
AUDIO_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8080))  # Cloud Run uses PORT=8080

# FastAPI App
app = FastAPI(title="Real-time Avatar Video Service", version="8.0.0")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global state
musetalk_pipeline: Optional[object] = None
pipeline_ready = asyncio.Event()
active_sessions = {}
is_loading_models = False

class SimpleVideoGenerator:
    """Fallback video generator when MuseTalk is not available"""
    
    def __init__(self):
        self.device = "cpu"
        
    def prepare_avatar(self, image: np.ndarray):
        """Prepare avatar for generation"""
        return cv2.resize(image, (512, 512))
    
    def generate_frame(self, reference_image: np.ndarray, audio_data: np.ndarray):
        """Generate frame with simple lip movement simulation"""
        frame = reference_image.copy()
        
        # Simple mouth movement based on audio energy
        if len(audio_data) > 0:
            audio_energy = np.mean(np.abs(audio_data))
            if audio_energy > 0.01:  # Threshold for mouth movement
                # Simple mouth opening effect
                h, w = frame.shape[:2]
                mouth_y = int(h * 0.7)  # Approximate mouth position
                mouth_x = int(w * 0.5)
                
                # Draw mouth opening based on audio energy
                mouth_open = min(int(audio_energy * 1000), 10)
                cv2.ellipse(frame, (mouth_x, mouth_y), (15, mouth_open), 0, 0, 360, (0, 0, 0), -1)
        
        return frame

async def load_models_background():
    """Load models in background - with fallback for Cloud Run"""
    global musetalk_pipeline, is_loading_models
    
    if is_loading_models:
        return
        
    is_loading_models = True
    logger.info("Starting model loading...")
    
    try:
        if MUSETALK_AVAILABLE:
            # Try to load MuseTalk models
            logger.info("Loading MuseTalk models...")
            models = load_all_model()
            musetalk_pipeline = MuseTalkPipeline(models=models, device=DEVICE)
            logger.info("MuseTalk models loaded successfully")
        else:
            # Use simple fallback generator
            logger.info("Using simple video generator (MuseTalk not available)")
            musetalk_pipeline = SimpleVideoGenerator()
            
        pipeline_ready.set()
        logger.info("Pipeline ready for requests")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Use fallback generator
        logger.info("Falling back to simple video generator")
        musetalk_pipeline = SimpleVideoGenerator()
        pipeline_ready.set()
    finally:
        is_loading_models = False

async def get_ready_pipeline():
    """Dependency to get ready pipeline"""
    if not pipeline_ready.is_set():
        logger.info("Pipeline not ready, starting background loading...")
        if not is_loading_models:
            asyncio.create_task(load_models_background())
        
        try:
            await asyncio.wait_for(pipeline_ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            # Return simple generator as fallback
            global musetalk_pipeline
            if musetalk_pipeline is None:
                musetalk_pipeline = SimpleVideoGenerator()
                pipeline_ready.set()
    
    return musetalk_pipeline

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.reference_image: Optional[np.ndarray] = None
        self.is_prepared = False
        self.last_activity = time.time()

    async def prepare_avatar(self, image_url: str, pipeline):
        logger.info(f"[{self.session_id}] Preparing avatar from URL: {image_url}")
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.get(image_url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
                
            self.reference_image = cv2.resize(img, (512, 512))
            
            # Prepare with pipeline
            if hasattr(pipeline, 'prepare_avatar'):
                self.reference_image = pipeline.prepare_avatar(self.reference_image)
            
            self.is_prepared = True
            logger.info(f"Avatar prepared successfully for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to prepare avatar: {e}")
            return False

def verify_api_key(auth: str = Depends(api_key_header)):
    if not auth or auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.on_event("startup")
async def startup_event():
    """Fast startup - models load in background"""
    logger.info(f"Starting video service on port {PORT}")
    # Don't block startup on model loading
    asyncio.create_task(load_models_background())

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {"message": "Avatar Video Service", "status": "running", "version": "8.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint - responds immediately"""
    return {
        "status": "healthy",
        "models_loaded": pipeline_ready.is_set(),
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
        "port": PORT,
        "active_sessions": len(active_sessions),
        "musetalk_available": MUSETALK_AVAILABLE
    }

@app.post("/init-stream")
async def init_stream(request: dict, _: None = Depends(verify_api_key)):
    """Initialize streaming session"""
    session_id = request.get("session_id")
    image_url = request.get("image_url")
    avatar_id = request.get("avatar_id")
    
    if not all([session_id, image_url]):
        raise HTTPException(status_code=400, detail="Missing session_id or image_url")
    
    # Get pipeline (will wait if not ready)
    pipeline = await get_ready_pipeline()
    
    session = Session(session_id)
    if not await session.prepare_avatar(image_url, pipeline):
        raise HTTPException(status_code=500, detail="Failed to prepare avatar from image URL.")
        
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
    """WebSocket endpoint for real-time video streaming"""
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session or not session.is_prepared:
        await websocket.close(code=1008, reason="Session not found or not prepared")
        return

    logger.info(f"WebSocket connected for session: {session_id}")
    
    # Get pipeline
    pipeline = await get_ready_pipeline()
    is_speaking = False
    frame_count = 0
    last_frame_time = time.time()
    
    try:
        while True:
            try:
                # Wait for audio data with timeout
                audio_chunk_bytes = await asyncio.wait_for(
                    websocket.receive_bytes(), 
                    timeout=0.04  # 25 FPS = 40ms per frame
                )
                
                session.last_activity = time.time()
                is_speaking = True

                # Convert audio data format
                if len(audio_chunk_bytes) > 0:
                    # Handle different audio formats
                    try:
                        # Try as float32 first
                        audio_float32 = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
                    except:
                        try:
                            # Try as int16 and convert
                            audio_16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
                            audio_float32 = audio_16.astype(np.float32) / 32768.0
                        except:
                            # Skip this chunk if format is unrecognizable
                            continue

                    # Generate video frame
                    if MUSETALK_AVAILABLE and hasattr(pipeline, 'audio_processor'):
                        # Use real MuseTalk pipeline
                        mel_chunks = pipeline.audio_processor.audio_to_mel_chunks(audio_float32, fps=TARGET_FPS)
                        motion_frames = pipeline.motion_generator.generate_motion(mel_chunks, session.face_bbox)
                        video_frame = pipeline.video_generator.generate_video(session.face_crop, motion_frames)
                        frame_to_send = video_frame[0]
                    else:
                        # Use simple fallback
                        frame_to_send = pipeline.generate_frame(session.reference_image, audio_float32)

            except asyncio.TimeoutError:
                # No audio received - send idle frame
                if is_speaking and (time.time() - session.last_activity > 0.2):
                    is_speaking = False
                
                frame_to_send = session.reference_image
                
                # Control frame rate for idle frames
                current_time = time.time()
                if current_time - last_frame_time < 1.0 / 15:  # 15 FPS for idle
                    await asyncio.sleep(0.01)
                    continue

            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                frame_to_send = session.reference_image

            # Encode and send frame
            try:
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_frame_time >= 1.0 / TARGET_FPS:
                    _, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    await websocket.send_bytes(buffer.tobytes())
                    
                    frame_count += 1
                    last_frame_time = current_time
                    
                    # Log performance every 100 frames
                    if frame_count % 100 == 0:
                        elapsed = current_time - (last_frame_time - (100 / TARGET_FPS))
                        fps = 100 / elapsed if elapsed > 0 else 0
                        logger.info(f"Session {session_id}: {fps:.1f} FPS")

            except Exception as e:
                logger.error(f"Error sending frame: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"WebSocket closed for session {session_id}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)