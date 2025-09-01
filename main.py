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
import threading
from collections import deque

# Add MuseTalk to Python path
musetalk_path = Path(__file__).parent / "MuseTalk"
if musetalk_path.exists():
    sys.path.insert(0, str(musetalk_path))

try:
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.utils import audio_to_mel_chunks
    from musetalk.utils.preprocessing import get_landmark_and_bbox, get_bbox_range
    from musetalk.utils.blending import get_image
    from musetalk.model import MuseTalkModel
    MUSETALK_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("MuseTalk modules imported successfully")
except ImportError as e:
    logging.warning(f"MuseTalk not available: {e}")
    MUSETALK_AVAILABLE = False

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default_video_service_key_change_in_production")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

TARGET_FPS = 25
AUDIO_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8080))

# FastAPI App
app = FastAPI(title="Real-time Avatar Video Service", version="9.0.0")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global state
musetalk_model: Optional[object] = None
pipeline_ready = asyncio.Event()
active_sessions: Dict[str, Any] = {}
is_loading_models = False

class MuseTalkVideoGenerator:
    """MuseTalk-based video generator with proper lip sync"""
    
    def __init__(self, model):
        self.model = model
        self.device = DEVICE
        
    def prepare_avatar(self, image: np.ndarray):
        """Prepare avatar image for MuseTalk processing"""
        try:
            # Resize and process image
            image = cv2.resize(image, (512, 512))
            
            # Get face landmarks and bbox
            landmarks, bbox = get_landmark_and_bbox(image)
            if landmarks is None or bbox is None:
                raise ValueError("No face detected in avatar image")
            
            # Get bbox range
            bbox_range = get_bbox_range(bbox, image.shape)
            
            # Crop face region
            face_crop = image[bbox_range[1]:bbox_range[3], bbox_range[0]:bbox_range[2]]
            
            return {
                'image': image,
                'landmarks': landmarks,
                'bbox': bbox,
                'bbox_range': bbox_range,
                'face_crop': face_crop
            }
        except Exception as e:
            logger.error(f"Error preparing avatar: {e}")
            return None
    
    def generate_frame(self, avatar_data: dict, audio_data: np.ndarray, is_speaking: bool = False):
        """Generate video frame with lip sync"""
        try:
            if not is_speaking or len(audio_data) == 0:
                # Return static frame when not speaking
                return avatar_data['image']
            
            # Convert audio to mel spectrogram chunks
            mel_chunks = audio_to_mel_chunks(
                audio_data, 
                fps=TARGET_FPS,
                sample_rate=AUDIO_SAMPLE_RATE
            )
            
            if len(mel_chunks) == 0:
                return avatar_data['image']
            
            # Generate lip-synced frame using MuseTalk
            with torch.no_grad():
                latent_out = self.model.inference(
                    mel_chunks[0],  # Use first mel chunk
                    avatar_data['face_crop'],
                    avatar_data['bbox_range']
                )
                
                # Blend generated face with original image
                result_frame = get_image(
                    latent_out,
                    avatar_data['image'],
                    avatar_data['bbox_range']
                )
                
                return result_frame
                
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            return avatar_data['image']  # Return static frame on error

class SimpleVideoGenerator:
    """Fallback video generator when MuseTalk is not available"""
    
    def __init__(self):
        self.device = "cpu"
        
    def prepare_avatar(self, image: np.ndarray):
        """Prepare avatar for simple generation"""
        image = cv2.resize(image, (512, 512))
        return {'image': image}
    
    def generate_frame(self, avatar_data: dict, audio_data: np.ndarray, is_speaking: bool = False):
        """Generate frame with simple mouth movement simulation"""
        frame = avatar_data['image'].copy()
        
        # Simple mouth movement based on audio energy
        if is_speaking and len(audio_data) > 0:
            audio_energy = np.mean(np.abs(audio_data))
            if audio_energy > 0.01:  # Threshold for mouth movement
                h, w = frame.shape[:2]
                mouth_y = int(h * 0.75)  # Mouth position
                mouth_x = int(w * 0.5)
                
                # Draw mouth opening based on audio energy
                mouth_open = min(int(audio_energy * 2000), 15)
                cv2.ellipse(frame, (mouth_x, mouth_y), (20, mouth_open), 0, 0, 360, (50, 50, 50), -1)
        
        return frame

async def load_models_background():
    """Load models in background with proper error handling"""
    global musetalk_model, is_loading_models
    
    if is_loading_models:
        return
        
    is_loading_models = True
    logger.info("Starting model loading...")
    
    try:
        if MUSETALK_AVAILABLE:
            logger.info("Loading MuseTalk models...")
            
            # Load MuseTalk models
            models = load_all_model(
                model_path="models",
                device=DEVICE
            )
            
            # Create MuseTalk model instance
            musetalk_model = MuseTalkVideoGenerator(models['musetalk'])
            logger.info("MuseTalk models loaded successfully")
        else:
            logger.info("Using simple video generator (MuseTalk not available)")
            musetalk_model = SimpleVideoGenerator()
            
        pipeline_ready.set()
        logger.info("Video pipeline ready for requests")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Falling back to simple video generator")
        musetalk_model = SimpleVideoGenerator()
        pipeline_ready.set()
    finally:
        is_loading_models = False

async def get_ready_pipeline():
    """Get ready pipeline with timeout"""
    if not pipeline_ready.is_set():
        logger.info("Pipeline not ready, starting background loading...")
        if not is_loading_models:
            asyncio.create_task(load_models_background())
        
        try:
            await asyncio.wait_for(pipeline_ready.wait(), timeout=60)
        except asyncio.TimeoutError:
            # Fallback to simple generator
            global musetalk_model
            if musetalk_model is None:
                musetalk_model = SimpleVideoGenerator()
                pipeline_ready.set()
    
    return musetalk_model

class VideoSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.avatar_data: Optional[dict] = None
        self.is_prepared = False
        self.last_activity = time.time()
        self.is_speaking = False
        self.audio_buffer = deque(maxlen=50)  # Buffer for audio chunks
        self.frame_lock = threading.Lock()

    async def prepare_avatar(self, image_url: str, pipeline):
        """Prepare avatar for video generation"""
        logger.info(f"[{self.session_id}] Preparing avatar from URL: {image_url}")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http_session:
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
            logger.info(f"Avatar prepared successfully for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to prepare avatar: {e}")
            return False

    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            if len(audio_array) > 0:
                self.audio_buffer.append(audio_array)
                self.last_activity = time.time()
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")

    def get_combined_audio(self) -> np.ndarray:
        """Get combined audio from buffer"""
        if len(self.audio_buffer) == 0:
            return np.array([])
        
        # Combine recent audio chunks
        recent_chunks = list(self.audio_buffer)[-3:]  # Use last 3 chunks
        return np.concatenate(recent_chunks) if recent_chunks else np.array([])

def verify_api_key(auth: str = Depends(api_key_header)):
    """Verify API key"""
    if not auth or auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.on_event("startup")
async def startup_event():
    """Startup event - begin model loading"""
    logger.info(f"Starting video service on port {PORT}")
    asyncio.create_task(load_models_background())

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Avatar Video Service", "status": "running", "version": "9.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": pipeline_ready.is_set(),
        "device": DEVICE,
        "port": PORT,
        "active_sessions": len(active_sessions),
        "musetalk_available": MUSETALK_AVAILABLE,
        "torch_version": torch.__version__
    }

@app.post("/init-stream")
async def init_stream(request: dict, _: None = Depends(verify_api_key)):
    """Initialize streaming session"""
    session_id = request.get("session_id")
    image_url = request.get("image_url")
    avatar_id = request.get("avatar_id")
    
    if not all([session_id, image_url]):
        raise HTTPException(status_code=400, detail="Missing session_id or image_url")
    
    # Get pipeline
    pipeline = await get_ready_pipeline()
    
    # Create session
    session = VideoSession(session_id)
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
    frame_count = 0
    last_frame_time = time.time()
    
    # Send initial frame
    try:
        initial_frame = session.avatar_data['image']
        _, buffer = cv2.imencode('.jpg', initial_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        await websocket.send_bytes(buffer.tobytes())
        logger.info(f"Sent initial frame for session {session_id}")
    except Exception as e:
        logger.error(f"Error sending initial frame: {e}")
    
    async def frame_generator():
        """Generate and send video frames"""
        nonlocal frame_count, last_frame_time
        
        while True:
            try:
                current_time = time.time()
                
                # Control frame rate
                frame_interval = 1.0 / TARGET_FPS
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(0.01)
                    continue
                
                # Get audio data for lip sync
                audio_data = session.get_combined_audio()
                
                # Generate frame
                with session.frame_lock:
                    frame = pipeline.generate_frame(
                        session.avatar_data, 
                        audio_data, 
                        session.is_speaking
                    )
                
                # Encode and send frame
                try:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    await websocket.send_bytes(buffer.tobytes())
                    
                    frame_count += 1
                    last_frame_time = current_time
                    
                    # Performance logging
                    if frame_count % 100 == 0:
                        elapsed = current_time - (last_frame_time - (100 * frame_interval))
                        fps = 100 / elapsed if elapsed > 0 else 0
                        logger.info(f"Session {session_id}: {fps:.1f} FPS, Speaking: {session.is_speaking}")
                        
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in frame generation: {e}")
                await asyncio.sleep(0.1)  # Small delay before retry
    
    try:
        # Start frame generation task
        frame_task = asyncio.create_task(frame_generator())
        
        # Handle incoming messages
        while True:
            try:
                # Listen for messages (audio data or control messages)
                message = await websocket.receive()
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
                                logger.debug(f"Speech started for session {session_id}")
                                
                            elif control_msg.get('type') == 'speech_end':
                                session.is_speaking = False
                                session.audio_buffer.clear()  # Clear buffer when speech ends
                                logger.debug(f"Speech ended for session {session_id}")
                                
                            elif control_msg.get('type') == 'request_keyframe':
                                # Force send current frame
                                with session.frame_lock:
                                    frame = session.avatar_data['image']
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                await websocket.send_bytes(buffer.tobytes())
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON control message received")
                            
            except asyncio.TimeoutError:
                # Check for session timeout
                if time.time() - session.last_activity > 300:  # 5 minutes timeout
                    logger.info(f"Session {session_id} timed out")
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup
        frame_task.cancel()
        if session_id in active_sessions:
            del active_sessions[session_id]
        gc.collect()
        logger.info(f"WebSocket closed and cleaned up for session {session_id}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)