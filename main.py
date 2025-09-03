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
from omegaconf import OmegaConf

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add MuseTalk to Python path
musetalk_path = Path(__file__).parent / "MuseTalk"
if musetalk_path.exists():
    sys.path.insert(0, str(musetalk_path))
try:
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
    from musetalk.utils.blending import get_image
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.models.talking_head import TalkingHead
    from musetalk.models.pose_guider import PoseGuider
    MUSETALK_AVAILABLE = True
    logger.info("MuseTalk modules imported successfully")
except ImportError as e:
    logger.warning(f"MuseTalk not available: {e}")
    MUSETALK_AVAILABLE = False


API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default_video_service_key_change_in_production")

# Use a deterministic device for consistency
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

TARGET_FPS = 25
AUDIO_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8080))

# FastAPI App
app = FastAPI(title="Real-time Avatar Video Service", version="9.1.0")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global state
musetalk_model: Optional[object] = None
pipeline_ready = asyncio.Event()
active_sessions: Dict[str, Any] = {}
is_loading_models = False

class MuseTalkVideoGenerator:
    """MuseTalk-based video generator for real-time streaming"""
    
    def __init__(self, models):
        self.models = models
        self.device = DEVICE
        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        logger.info("MuseTalk Video Generator initialized with AudioProcessor")
        self.cfg = OmegaConf.load("./MuseTalk/musetalk/config/Musetalk.yaml")
        
    def prepare_avatar(self, image: np.ndarray):
        """Prepare avatar image for MuseTalk processing and get initial pose"""
        try:
            image_resized = cv2.resize(image, (512, 512))
            
            # Use MuseTalk's preprocessing to get landmark and bbox
            coord_list, frame_list = get_landmark_and_bbox([image_resized], bbox_shift=0)
            
            if len(coord_list) > 0 and coord_list[0] != coord_placeholder:
                bbox = coord_list[0]
                processed_frame = frame_list[0]
                
                # Use MuseTalk pose estimator to get initial pose
                pose_keypoints = self.models["pose_estimator"].get_pose(processed_frame)
                
                return {
                    'image': processed_frame,
                    'bbox': bbox,
                    'original_image': image_resized,
                    'last_pose': pose_keypoints
                }
            else:
                logger.warning("No face detected in the avatar image.")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing avatar: {e}")
            return None
    
    @torch.no_grad()
    def generate_frame(self, avatar_data: dict, audio_data: np.ndarray):
        """Generate a single video frame using MuseTalk"""
        if avatar_data is None:
            logger.error("Avatar data is not prepared.")
            return None
        
        try:
            # 1. Process audio chunk to get whisper embeddings
            if audio_data.size == 0:
                logger.debug("No audio data to process. Returning original image.")
                return avatar_data['image']
                
            audio_data_tensor = torch.from_numpy(audio_data).float().to(self.device)
            audio_feat = self.audio_processor.get_whisper_feature(audio_data_tensor)
            
            # Ensure the audio feature is in the correct format (1, seq_len, 1024)
            audio_feat = audio_feat.unsqueeze(0)
            
            # 2. Get initial image and pose
            image = avatar_data['image']
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
            
            # Get last known pose
            pose = avatar_data['last_pose']
            pose_tensor = torch.from_numpy(pose).float().unsqueeze(0).to(self.device)
            
            # 3. Use PoseGuider to predict the next pose
            pose_ref = self.models["pose_guider"](pose_tensor, audio_feat)
            
            # 4. Use TalkingHead model for lip-sync and expression generation
            # This is the core generation step
            gen_frame_latents = self.models["talking_head"](
                image_latents=self.models["image_latents"],
                audio_feature=audio_feat,
                pose_ref=pose_ref,
                ref_eyeblink=None,
                refer_image=image_tensor,
                refer_pose=pose_tensor
            )
            
            # 5. Decode the latents to get the final image
            gen_frame = self.models["vae"].decode(gen_frame_latents.sample).squeeze(0).permute(1, 2, 0)
            gen_frame = gen_frame.clamp(-1, 1)
            gen_frame = (gen_frame + 1.0) / 2.0
            gen_frame = (gen_frame * 255).detach().cpu().numpy().astype(np.uint8)

            # 6. Update the last pose for the next frame
            avatar_data['last_pose'] = pose_ref.squeeze(0).detach().cpu().numpy()

            return gen_frame
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            return None

class SimpleVideoGenerator:
    """Fallback video generator when MuseTalk is not available"""
    
    def __init__(self):
        self.device = "cpu"
        
    def prepare_avatar(self, image: np.ndarray):
        """Prepare avatar for simple generation"""
        image = cv2.resize(image, (512, 512))
        return {'image': image}
    
    def generate_frame(self, avatar_data: dict, audio_data: np.ndarray):
        """Generate frame with simple mouth movement simulation"""
        frame = avatar_data['image'].copy()
        
        # Simple mouth movement based on audio energy
        if len(audio_data) > 0:
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
            logger.info("Loading MuseTalk models from disk...")
            models = load_all_model(
                model_path="./models",
                device=DEVICE
            )
            musetalk_model = MuseTalkVideoGenerator(models)
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
            await asyncio.wait_for(pipeline_ready.wait(), timeout=600)  # Extended timeout for model loading
        except asyncio.TimeoutError:
            global musetalk_model
            if musetalk_model is None:
                musetalk_model = SimpleVideoGenerator()
                pipeline_ready.set()
                logger.error("Model loading timed out. Falling back to simple video generator.")
    
    return musetalk_model

class VideoSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.avatar_data: Optional[dict] = None
        self.is_prepared = False
        self.last_activity = time.time()
        self.audio_buffer = deque(maxlen=TARGET_FPS) # Buffer for 1 second of audio at 25 FPS
        
    async def prepare_avatar(self, image_url: str, pipeline):
        """Prepare avatar for video generation"""
        logger.info(f"[{self.session_id}] Preparing avatar from URL: {image_url}")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http_session:
                async with http_session.get(image_url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
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
            # We assume float32 for simplicity based on typical stream formats
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            if len(audio_array) > 0:
                self.audio_buffer.append(audio_array)
            self.last_activity = time.time()
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")

    def get_combined_audio(self) -> np.ndarray:
        """Get combined audio for one frame"""
        if len(self.audio_buffer) == 0:
            return np.array([])
        
        # We need a small audio chunk for each frame
        # We'll take the first chunk and remove it
        chunk = self.audio_buffer.popleft()
        return chunk

def verify_api_key(auth: str = Depends(api_key_header)):
    """Verify API key"""
    if not auth or auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.on_event("startup")
async def startup_event():
    """Startup event - begin model loading"""
    logger.info(f"Starting video service on port {PORT}")
    pass

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Avatar Video Service", "status": "running", "version": "9.1.0"}

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
    logger.info(f"WebSocket connection request for session: {session_id}")
    
    session = active_sessions.get(session_id)
    if not session or not session.is_prepared:
        await websocket.close(code=1008, reason="Session not found or not prepared")
        return
    
    await websocket.accept()
    logger.info(f"WebSocket accepted for session: {session_id}")
    
    pipeline = await get_ready_pipeline()
    
    # Main streaming loop
    try:
        # Start a separate task to listen for incoming audio data
        async def audio_listener():
            try:
                while True:
                    audio_data = await websocket.receive_bytes()
                    session.add_audio_chunk(audio_data)
            except WebSocketDisconnect:
                logger.info(f"Audio listener for {session_id} disconnected.")
            except Exception as e:
                logger.error(f"Audio listener error for {session_id}: {e}")

        listener_task = asyncio.create_task(audio_listener())

        # Main video generation loop
        while True:
            # Get audio for the next frame from the buffer
            audio_for_frame = session.get_combined_audio()
            
            if audio_for_frame.size > 0:
                frame = pipeline.generate_frame(session.avatar_data, audio_for_frame)
                
                if frame is not None:
                    # Blend with the original image for a smoother effect if needed
                    # Not implemented in this version, but can be added
                    
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send_bytes(buffer.tobytes())
                else:
                    logger.warning(f"Failed to generate frame for session {session_id}")
            else:
                # If no audio is available, send a static frame
                frame = session.avatar_data['image']
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                await websocket.send_bytes(buffer.tobytes())
            
            # Control the FPS
            await asyncio.sleep(1.0 / TARGET_FPS)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        listener_task.cancel()
        await listener_task
        if session_id in active_sessions:
            del active_sessions[session_id]
            gc.collect()
            logger.info(f"Session ended: {session_id}")

        
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)
