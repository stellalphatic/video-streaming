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

# Import the official MuseTalk pipeline and utilities
from musetalk.pipelines.pipeline_musetalk import MuseTalkPipeline
from musetalk.utils.utils import load_all_model

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default_video_service_key_change_in_production")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FPS = 25
AUDIO_SAMPLE_RATE = 16000

# --- FastAPI App and Globals for Background Loading ---
app = FastAPI(title="Real-time Avatar Video Service", version="7.0.0")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Globals to manage model loading state
musetalk_pipeline: Optional[MuseTalkPipeline] = None
pipeline_ready = asyncio.Event() # This event will signal when models are loaded
active_sessions = {}

# --- Background Model Loading Task ---
async def load_models_background():
    """Loads models in the background without blocking the server startup."""
    global musetalk_pipeline
    logger.info("🚀 Starting background model loading...")
    try:
        models = load_all_model()
        musetalk_pipeline = MuseTalkPipeline(models=models, device=DEVICE)
        pipeline_ready.set() # Signal that the pipeline is ready to accept requests
        logger.info("✅ All models loaded and pipeline is ready.")
    except Exception as e:
        logger.error(f"❌ Critical error during background model loading: {e}", exc_info=True)
        # In a real-world scenario, you might want a more robust way to handle this failure.
        # For now, the server will continue running but will be unable to process requests.

# --- Dependency to check if pipeline is ready ---
async def get_ready_pipeline():
    """A dependency that waits until the pipeline is ready."""
    if not pipeline_ready.is_set():
        logger.info("Request received, but pipeline is not ready. Waiting...")
        try:
            # Wait for the pipeline_ready event to be set, with a timeout
            await asyncio.wait_for(pipeline_ready.wait(), timeout=300)
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for models to load.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are still loading, please try again shortly.",
            )
    return musetalk_pipeline

# --- Session Class (No changes needed) ---
class Session:
    # ... (The Session class from the previous version is unchanged) ...
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.reference_image: Optional[np.ndarray] = None
        self.face_crop: Optional[np.ndarray] = None
        self.face_bbox: Optional[np.ndarray] = None
        self.is_prepared = False
        self.last_activity = time.time()

    async def prepare_avatar(self, image_url: str, pipeline: MuseTalkPipeline):
        logger.info(f"[{self.session_id}] Preparing avatar from URL: {image_url}")
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.get(image_url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.reference_image = cv2.resize(img, (512, 512))

            self.face_crop, self.face_bbox = pipeline.face_analyzer.get_face_crop(self.reference_image)
            
            self.is_prepared = True
            logger.info(f"✅ [{self.session_id}] Avatar prepared successfully.")
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to prepare avatar: {e}", exc_info=True)
            return False

# --- FastAPI Events & Endpoints ---

def verify_api_key(auth: str = Depends(api_key_header)):
    if not auth or auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.on_event("startup")
async def startup_event():
    """Starts the server immediately and kicks off model loading in the background."""
    asyncio.create_task(load_models_background())

@app.get("/health")
async def health_check():
    """
    Health check endpoint. Responds immediately to pass Cloud Run's check.
    Includes model readiness status.
    """
    return {
        "status": "healthy",
        "models_loaded": pipeline_ready.is_set(),
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/init-stream")
async def init_stream(
    request: dict,
    _: None = Depends(verify_api_key),
    pipeline: MuseTalkPipeline = Depends(get_ready_pipeline)
):
    session_id = request.get("session_id")
    image_url = request.get("image_url")
    if not all([session_id, image_url]):
        raise HTTPException(status_code=400, detail="Missing session_id or image_url")
    
    session = Session(session_id)
    if not await session.prepare_avatar(image_url, pipeline):
        raise HTTPException(status_code=500, detail="Failed to prepare avatar from image URL.")
        
    active_sessions[session_id] = session
    logger.info(f"🎬 Session initialized and prepared: {session_id}")
    return {"status": "initialized", "session_id": session_id}

@app.post("/end-stream")
async def end_stream(request: dict, _: None = Depends(verify_api_key)):
    session_id = request.get("session_id")
    if session_id in active_sessions:
        del active_sessions[session_id]
        gc.collect()
        logger.info(f"🛑 Session ended and cleaned up by API call: {session_id}")
    return {"status": "ended", "session_id": session_id}

@app.websocket("/stream/{session_id}")
async def websocket_stream(
    websocket: WebSocket,
    session_id: str,
    pipeline: MuseTalkPipeline = Depends(get_ready_pipeline)
):
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session or not session.is_prepared:
        reason = "Session not found or not prepared. Call /init-stream first."
        await websocket.close(code=1008, reason=reason)
        return

    logger.info(f"🔌 WebSocket connected for session: {session_id}")
    is_speaking = False
    
    try:
        while True:
            try:
                audio_chunk_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
                session.last_activity = time.time()
                is_speaking = True

                audio_16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
                audio_float32 = audio_16.astype(np.float32) / 32768.0

                mel_chunks = pipeline.audio_processor.audio_to_mel_chunks(audio_float32, fps=TARGET_FPS)
                motion_frames = pipeline.motion_generator.generate_motion(mel_chunks, session.face_bbox)
                video_frame = pipeline.video_generator.generate_video(session.face_crop, motion_frames)
                
                frame_to_send = video_frame[0]

            except asyncio.TimeoutError:
                if is_speaking and (time.time() - session.last_activity > 0.2):
                    is_speaking = False
                
                frame_to_send = session.reference_image
                await asyncio.sleep(1 / 15)

            _, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 90])
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session: {session_id}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"WebSocket for session {session_id} closed.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), workers=1)