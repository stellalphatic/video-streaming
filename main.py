import uvicorn
import logging
import asyncio
import os
import time
import gc
import aiohttp
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Optional, Dict, Any
import sys
import json
from pathlib import Path
from collections import deque
from PIL import Image
import io
import librosa
import base64
from omegaconf import OmegaConf

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add MuseTalk to Python path
musetalk_path = Path(__file__).parent / "MuseTalk"
if str(musetalk_path) not in sys.path:
    sys.path.insert(0, str(musetalk_path))

try:
    from musetalk.utils.utils import load_all_model, datagen, get_audio_feature_from_audio
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder, read_imgs
    from musetalk.utils.blending import get_image
    from diffusers import DDIMScheduler
    # No need to import specific classes, as load_all_model handles them
    MUSETALK_AVAILABLE = True
    logger.info("MuseTalk modules imported successfully")
except ImportError as e:
    logger.warning(f"MuseTalk not available: {e}")
    MUSETALK_AVAILABLE = False

API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default_video_service_key_change_in_production")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

TARGET_FPS = 25
AUDIO_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8080))

# FastAPI App
app = FastAPI(title="Pure MuseTalk Real-time Video Service", version="5.0.0")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global state
# We will load the models once and pass references to new sessions
musetalk_models = None
pipeline_ready = asyncio.Event()
active_sessions: Dict[str, Any] = {}
is_loading_models = False

class AsyncMuseTalkSession:
    """Handles a single real-time video session with MuseTalk"""
    def __init__(self, session_id: str, models: Dict[str, Any]):
        self.session_id = session_id
        self.device = DEVICE
        self.models = models
        self.vae = self.models['vae'].to(self.device).eval()
        self.unet = self.models['unet'].to(self.device).eval()
        self.audio_processor = self.models['audio_processor']
        self.pose_estimator = self.models['dwpose'].to(self.device).eval()
        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.scheduler.set_timesteps(1) # Set to a single step for real-time inference
        
        self.original_image_np = None
        self.processed_frame_np = None
        self.bbox = None
        self.image_latents = None
        self.idle_latents_cache = deque(maxlen=TARGET_FPS * 10) # Cache 10 seconds of idle video
        self.idle_noise_state = None
        self.audio_buffer = deque() # Use a simple deque for audio features
        self.frame_count = 0
        self.is_speaking = False

    async def prepare_avatar(self, image_url: str):
        """Prepare avatar for real-time generation using aiohttp for async downloads"""
        logger.info(f"[{self.session_id}] Preparing avatar from URL: {image_url}")
        
        try:
            # Asynchronous download with aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    resp.raise_for_status()
                    image_bytes = await resp.read()
            
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = image.resize((512, 512))
            image_array = np.array(image)
            
            # Get facial landmarks and pose
            coord_list, frame_list = get_landmark_and_bbox([image_array], bbox_shift=0)
            
            if not coord_list or coord_list[0] == coord_placeholder:
                raise ValueError("No face detected in avatar image")
            
            self.bbox = coord_list[0]
            self.processed_frame_np = frame_list[0]
            self.original_image_np = image_array
            
            # Get pose features
            with torch.no_grad():
                pose_keypoints = self.pose_estimator([self.processed_frame_np])[0]
                self.pose_keypoints = torch.from_numpy(pose_keypoints).float().to(self.device)
            
            # Convert to tensor and encode to latent space
            image_tensor = torch.from_numpy(self.processed_frame_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            image_tensor = image_tensor / 255.0 * 2.0 - 1.0
            
            with torch.no_grad():
                self.image_latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
            
            # Generate and cache idle frames
            self._generate_idle_cache()
            
            logger.info(f"Avatar prepared successfully for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to prepare avatar: {e}")
            self.cleanup()
            return False

    def _generate_idle_cache(self):
        """Generates a loopable sequence of idle frames and caches them as latents."""
        self.idle_latents_cache.clear()
        
        # We only need one UNet pass for idle
        timesteps = torch.tensor([1]).to(self.device)
        noise = torch.randn_like(self.image_latents) * 0.05
        pose_noise = torch.randn_like(self.pose_keypoints) * 0.01

        # Use a silent audio feature for idle
        idle_audio_features = torch.zeros(1, 1, 1024).to(self.device)
        
        for _ in range(self.idle_latents_cache.maxlen):
            # Simulate subtle head movements by adding noise to pose
            current_pose = self.pose_keypoints + pose_noise
            
            # UNet forward pass
            noise_pred = self.unet(
                self.image_latents + noise,
                timesteps,
                encoder_hidden_states=idle_audio_features.unsqueeze(0),
                ref_pose_feature=current_pose
            ).sample
            
            # DDIM denoising step
            denoised_latents = self.scheduler.step(
                noise_pred, timesteps, self.image_latents + noise, return_dict=False
            )[0]
            
            self.idle_latents_cache.append(denoised_latents)
            
            # Slowly change noise for a new subtle movement
            noise = torch.randn_like(self.image_latents) * 0.05
            pose_noise = torch.randn_like(self.pose_keypoints) * 0.01

    def add_audio_chunk(self, audio_data: bytes):
        """Processes and stores audio chunk features for lip sync"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            if len(audio_array) > 0:
                audio_features = get_audio_feature_from_audio(
                    audio_array,
                    self.audio_processor,
                    self.device
                )
                self.audio_buffer.append(audio_features)
                return True
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return False
        return False
        
    def generate_frame(self):
        """Generates a single video frame with audio lip-sync or from cache"""
        if not self.image_latents:
            return None
        
        try:
            with torch.no_grad():
                if self.is_speaking and self.audio_buffer:
                    # Audio-driven generation
                    audio_features = self.audio_buffer.popleft()
                    
                    # Add subtle noise for realism
                    noise = torch.randn_like(self.image_latents) * 0.05
                    
                    timesteps = torch.tensor([1]).to(self.device) # Single denoising step
                    
                    noise_pred = self.unet(
                        self.image_latents + noise,
                        timesteps,
                        encoder_hidden_states=audio_features.unsqueeze(0),
                        ref_pose_feature=self.pose_keypoints
                    ).sample
                    
                    denoised_latents = self.scheduler.step(
                        noise_pred, timesteps, self.image_latents + noise, return_dict=False
                    )[0]
                    
                    # Blend the generated mouth with the original latent
                    # This is a key step for stability and realism
                    final_latents = self._blend_latents(
                        original_latents=self.image_latents,
                        generated_latents=denoised_latents
                    )
                    
                else:
                    # Idle loop
                    if not self.idle_latents_cache:
                        self._generate_idle_cache()
                    final_latents = self.idle_latents_cache.popleft()
                    self.idle_latents_cache.append(final_latents) # Re-add to the end for looping
                
                # Decode latents to image
                decoded_image = self.vae.decode(final_latents / 0.18215).sample
                
                # Post-process and get blended image
                frame_np = get_image(
                    self.processed_frame_np,
                    self.original_image_np,
                    decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    self.bbox
                )
                
                frame_image = Image.fromarray(frame_np)
                img_buffer = io.BytesIO()
                frame_image.save(img_buffer, format='JPEG', quality=85, optimize=True)
                img_bytes = img_buffer.getvalue()
                
                self.frame_count += 1
                return img_bytes
                
        except Exception as e:
            logger.error(f"Frame generation failed: {e}")
            return None
    
    def _blend_latents(self, original_latents, generated_latents, alpha=0.5):
        """Blends the generated mouth region with the original image's latent representation.
           This is a simplified version of a more complex blending approach.
        """
        # Create a mask in latent space (simplified)
        mask = torch.zeros_like(original_latents)
        mask[:, :, 30:50, 25:40] = 1.0 # A rough mask for the mouth region (approx)
        
        # Apply the mask to blend
        blended_latents = mask * generated_latents + (1 - mask) * original_latents
        return blended_latents

    def cleanup(self):
        """Cleans up resources for the session"""
        self.audio_buffer.clear()
        self.idle_latents_cache.clear()
        if self.image_latents:
            del self.image_latents
        if self.pose_keypoints:
            del self.pose_keypoints
        if self.models:
            self.vae.to('cpu')
            self.unet.to('cpu')
            self.pose_estimator.to('cpu')
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Session {self.session_id} cleaned up")

async def load_musetalk_pipeline():
    """Load all MuseTalk models once on startup"""
    global musetalk_models, is_loading_models
    
    if is_loading_models or musetalk_models:
        return
        
    is_loading_models = True
    logger.info("Loading pure MuseTalk pipeline...")
    
    try:
        if MUSETALK_AVAILABLE:
            # Load all models once and store globally
            musetalk_models = load_all_model(
                base_model_path=Path(__file__).parent / "models"
            )
            
            # Warmup models
            await _warmup_models(musetalk_models)
            
            pipeline_ready.set()
            logger.info("Pure MuseTalk pipeline loaded and warmed up successfully")
        else:
            logger.error("MuseTalk not available")
            musetalk_models = None
            pipeline_ready.set()
            
    except Exception as e:
        logger.error(f"Pipeline loading failed: {e}")
        musetalk_models = None
        pipeline_ready.set()
    finally:
        is_loading_models = False

async def _warmup_models(models):
    """Warmup models for faster inference"""
    logger.info("Warming up MuseTalk models...")
    device = DEVICE
    
    try:
        with torch.no_grad():
            # Warmup VAE
            dummy_image = torch.randn(1, 3, 512, 512).to(device)
            latent = models['vae'].encode(dummy_image).latent_dist.sample()
            _ = models['vae'].decode(latent)
            
            # Warmup UNet
            dummy_latent = torch.randn(1, 4, 64, 64).to(device)
            dummy_audio_feat = torch.randn(1, 50, 1024).to(device)
            dummy_pose_feat = torch.randn(1, 25, 4).to(device) # Simplified pose
            timesteps = torch.tensor([1]).to(device)
            _ = models['unet'](
                dummy_latent,
                timesteps,
                encoder_hidden_states=dummy_audio_feat,
                ref_pose_feature=dummy_pose_feat
            ).sample
            
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

async def get_pipeline_models():
    """Get ready pipeline models"""
    if not pipeline_ready.is_set():
        await load_musetalk_pipeline()
    return musetalk_models

def verify_api_key(auth: str = Depends(api_key_header)):
    """Verify API key"""
    if not auth or auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info(f"Starting pure MuseTalk video service on port {PORT}")
    asyncio.create_task(load_musetalk_pipeline())

@app.get("/")
async def root():
    return {"message": "Pure MuseTalk Real-time Video Service", "status": "running", "version": "5.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": pipeline_ready.is_set() and musetalk_models is not None,
        "device": DEVICE,
        "port": PORT,
        "active_sessions": len(active_sessions),
        "musetalk_available": MUSETALK_AVAILABLE,
        "torch_version": torch.__version__ if 'torch' in sys.modules else "not available"
    }

@app.post("/init-stream")
async def init_stream(request: dict, _: None = Depends(verify_api_key)):
    """Initialize streaming session"""
    session_id = request.get("session_id")
    image_url = request.get("image_url")
    
    if not all([session_id, image_url]):
        raise HTTPException(status_code=400, detail="Missing session_id or image_url")
    
    models = await get_pipeline_models()
    if not models:
        raise HTTPException(status_code=503, detail="MuseTalk pipeline not ready")
    
    # Create a new session instance for the user
    session_handler = AsyncMuseTalkSession(session_id, models)
    success = await session_handler.prepare_avatar(image_url)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to prepare avatar")
    
    # Store the session handler
    active_sessions[session_id] = {
        'handler': session_handler,
        'websocket': None,
        'generation_task': None,
        'last_activity': time.time()
    }
    
    logger.info(f"Session initialized: {session_id}")
    
    return {
        "status": "initialized", 
        "session_id": session_id,
        "timestamp": time.time()
    }

@app.post("/end-stream")
async def end_stream(request: dict, _: None = Depends(verify_api_key)):
    """End streaming session"""
    session_id = request.get("session_id")
    
    if session_id in active_sessions:
        session_data = active_sessions[session_id]
        
        if session_data['generation_task']:
            session_data['generation_task'].cancel()
        
        session_data['handler'].cleanup()
        del active_sessions[session_id]
        logger.info(f"Session ended: {session_id}")
    
    return {"status": "ended", "session_id": session_id}

@app.websocket("/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time video streaming"""
    logger.info(f"WebSocket connection request for session: {session_id}")
    
    session_data = active_sessions.get(session_id)
    if not session_data or not session_data['handler'].image_latents:
        logger.error(f"Session {session_id} not found or not prepared")
        await websocket.close(code=1008, reason="Session not found or not prepared")
        return
    
    await websocket.accept()
    session_data['websocket'] = websocket
    session_handler = session_data['handler']
    logger.info(f"WebSocket accepted for session: {session_id}")
    
    # Send initial frame
    try:
        initial_frame = session_handler.generate_frame()
        if initial_frame:
            await websocket.send_bytes(initial_frame)
            logger.info(f"Initial frame sent for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to send initial frame: {e}")
    
    async def frame_generator():
        """Generates and sends frames continuously"""
        last_frame_time = time.time()
        frame_interval = 1.0 / TARGET_FPS
        
        while True:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)
                
                frame_bytes = session_handler.generate_frame()
                
                if frame_bytes and websocket.client_state.CONNECTED:
                    await websocket.send_bytes(frame_bytes)
                else:
                    break
                
                last_frame_time = time.time()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Frame generation error: {e}")
                break
    
    session_data['generation_task'] = asyncio.create_task(frame_generator())
    
    try:
        while True:
            try:
                # Use a small timeout to allow for periodic health checks or other tasks
                message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
                
                if message['type'] == 'websocket.receive':
                    if 'bytes' in message and len(message['bytes']) > 0:
                        session_data['handler'].add_audio_chunk(message['bytes'])
                        session_data['handler'].is_speaking = True
                    
                    elif 'text' in message:
                        try:
                            control_msg = json.loads(message['text'])
                            msg_type = control_msg.get('type')
                            
                            if msg_type == 'speech_end':
                                # When speech ends, give time to process remaining audio buffer
                                # then switch to idle mode
                                await asyncio.sleep(0.5) # Process any final chunks
                                session_data['handler'].is_speaking = False
                                
                        except json.JSONDecodeError:
                            pass
            except asyncio.TimeoutError:
                if not session_data['handler'].audio_buffer and session_data['handler'].is_speaking:
                    session_data['handler'].is_speaking = False
                continue
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
    finally:
        logger.info(f"Cleaning up session {session_id}")
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            if session_data['generation_task']:
                session_data['generation_task'].cancel()
            session_data['handler'].cleanup()
            del active_sessions[session_id]
        logger.info(f"Session {session_id} cleanup completed")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)