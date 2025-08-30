from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import torch
import cv2
import numpy as np
import json
import io
import base64
from typing import Dict, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import os
from pathlib import Path
import secrets
import hashlib
import aiohttp
import gc
from collections import deque
import websockets

# Import custom modules
from app.core.musetalk_video_processor import RealTimeMuseTalkProcessor
from app.core.audio_processor import AudioProcessor
from app.core.frame_manager import FrameManager
from app.core.session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-time Avatar Video Service", 
    version="2.0.0",
    description="Real-time video streaming service with MuseTalk lip-sync and facial animations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
video_processor = None
session_manager = SessionManager()

VIDEO_SERVICE_API_KEY = os.getenv("VIDEO_SERVICE_API_KEY")
if not VIDEO_SERVICE_API_KEY:
    logger.warning("VIDEO_SERVICE_API_KEY not set, using default key")
    VIDEO_SERVICE_API_KEY = "default_video_service_key_change_in_production"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for authentication"""
    if credentials.credentials != VIDEO_SERVICE_API_KEY:
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.on_event("startup")
async def startup_event():
    """Initialize the video processor on startup"""
    global video_processor
    
    try:
        logger.info("🚀 Starting Real-time Avatar Video Service...")
        logger.info("📦 Initializing video processor...")
        
        video_processor = RealTimeMuseTalkProcessor()
        await video_processor.initialize()
        
        logger.info("✅ Video processor initialized successfully")
        logger.info("🎥 Real-time video service ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize video service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down video service...")
    
    # Cleanup all sessions
    active_sessions = list(session_manager.active_sessions.keys())
    for session_id in active_sessions:
        await session_manager.cleanup_session(session_id)
    
    # Cleanup video processor
    if video_processor:
        video_processor.cleanup()
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Video service shutdown complete")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    cuda_available = torch.cuda.is_available()
    gpu_memory = 0
    gpu_memory_used = 0
    
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_used = torch.cuda.memory_allocated(0)
    
    performance_stats = {}
    if video_processor:
        performance_stats = video_processor.get_performance_stats()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cuda_available": cuda_available,
        "gpu_memory_gb": gpu_memory / (1024**3) if cuda_available else 0,
        "gpu_memory_used_gb": gpu_memory_used / (1024**3) if cuda_available else 0,
        "gpu_utilization": (gpu_memory_used / gpu_memory * 100) if gpu_memory > 0 else 0,
        "active_sessions": session_manager.get_active_count(),
        "model_loaded": video_processor is not None and video_processor.is_initialized,
        "performance_stats": performance_stats
    }

@app.post("/init-stream")
async def init_stream(request: dict, api_key: str = Depends(verify_api_key)):
    """Initialize streaming session for avatar"""
    try:
        session_id = request.get("session_id")
        avatar_id = request.get("avatar_id")
        image_url = request.get("image_url")
        target_fps = request.get("target_fps", 25)
        
        if not all([session_id, avatar_id, image_url]):
            raise HTTPException(
                status_code=400, 
                detail="Missing required parameters: session_id, avatar_id, image_url"
            )
        
        logger.info(f"🎬 Initializing stream session: {session_id} for avatar: {avatar_id}")
        
        # Create session with enhanced error handling
        success = await session_manager.create_session(
            session_id=session_id,
            avatar_id=avatar_id,
            image_url=image_url,
            video_processor=video_processor,
            target_fps=target_fps
        )
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Failed to create session - check logs for details"
            )
        
        logger.info(f"✅ Stream session initialized: {session_id}")
        return {
            "status": "initialized", 
            "session_id": session_id,
            "timestamp": time.time(),
            "target_fps": target_fps
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error initializing stream: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/end-stream")
async def end_stream(request: dict, api_key: str = Depends(verify_api_key)):
    """End streaming session"""
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id")
        
        logger.info(f"🛑 Ending stream session: {session_id}")
        await session_manager.end_session(session_id)
        
        # Force cleanup memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"✅ Stream session ended: {session_id}")
        return {
            "status": "ended", 
            "session_id": session_id,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error ending stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for real-time video streaming"""
    
    await websocket.accept()
    logger.info(f"🔌 WebSocket connection established for session: {session_id}")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_fps_report = start_time
    last_frame_time = 0
    target_fps = 25
    frame_interval = 1.0 / target_fps
    
    # State management
    is_speaking = False
    current_emotion = "neutral"
    audio_buffer = deque(maxlen=50)  # Keep last 50 audio chunks
    
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Session not found. Please initialize first."
            }))
            await websocket.close()
            return
        
        target_fps = session.get('target_fps', 25)
        frame_interval = 1.0 / target_fps
        
        # Initialize frame manager
        frame_manager = FrameManager(session)
        await frame_manager.start_idle_animation()
        
        # Send ready message
        await websocket.send_text(json.dumps({
            "type": "ready",
            "message": "Real-time video stream ready",
            "session_id": session_id,
            "target_fps": target_fps,
            "features": {
                "lip_sync": True,
                "real_time": True,
                "low_latency": True,
                "facial_animation": True
            }
        }))
        
        logger.info(f"🎥 Starting real-time video processing for session: {session_id}")
        
        # Main streaming loop
        while True:
            try:
                # Handle incoming messages with timeout
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.02)
                    
                    if message["type"] == "text":
                        # Handle control messages
                        data = json.loads(message["text"])
                        await handle_control_message(
                            data, frame_manager, session, websocket
                        )
                        
                        # Update speaking state
                        if data.get("type") == "speech_start":
                            is_speaking = True
                        elif data.get("type") == "speech_end":
                            is_speaking = False
                        elif data.get("type") == "change_emotion":
                            current_emotion = data.get("emotion", "neutral")
                        
                    elif message["type"] == "bytes":
                        # Handle audio data for lip sync
                        audio_data = message["bytes"]
                        
                        if len(audio_data) > 0:
                            # Add to audio buffer
                            audio_buffer.append({
                                'data': audio_data,
                                'timestamp': time.time()
                            })
                            
                            # Process audio for video generation
                            success = await video_processor.process_audio_chunk(
                                audio_data,
                                session['reference_image'],
                                current_emotion
                            )
                            
                            if not success:
                                logger.warning(f"Failed to process audio chunk for session {session_id}")
                
                except asyncio.TimeoutError:
                    # No incoming message, continue with frame generation
                    pass
                
                # Generate and send video frame
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_frame_time >= frame_interval:
                    # Get next video frame
                    if is_speaking:
                        # Get lip-synced frame from video processor
                        encoded_frame = await video_processor.get_next_frame()
                        
                        if encoded_frame:
                            # Send video frame
                            await websocket.send_bytes(encoded_frame)
                            frame_count += 1
                            last_frame_time = current_time
                        else:
                            # Fallback to idle frame if no generated frame available
                            idle_frame = await frame_manager.get_next_idle_frame()
                            if idle_frame is not None:
                                # Encode and send idle frame
                                encoded_idle = encode_frame_jpeg(idle_frame)
                                await websocket.send_bytes(encoded_idle)
                                frame_count += 1
                                last_frame_time = current_time
                    else:
                        # Send idle animation frame
                        idle_frame = await frame_manager.get_next_idle_frame()
                        if idle_frame is not None:
                            encoded_idle = encode_frame_jpeg(idle_frame)
                            await websocket.send_bytes(encoded_idle)
                            frame_count += 1
                            last_frame_time = current_time
                
                # Report FPS periodically
                if current_time - last_fps_report >= 5.0:
                    fps = frame_count / (current_time - start_time)
                    logger.info(f"Session {session_id}: {fps:.1f} FPS")
                    
                    # Send performance stats to client
                    stats = video_processor.get_performance_stats()
                    await websocket.send_text(json.dumps({
                        "type": "performance_stats",
                        "fps": fps,
                        "processing_stats": stats,
                        "latency": stats.get('avg_frame_time', 0) * 1000  # Convert to ms
                    }))
                    
                    last_fps_report = current_time
                
                # Cleanup old audio buffer entries (keep last 5 seconds)
                cutoff_time = current_time - 5.0
                while audio_buffer and audio_buffer[0]['timestamp'] < cutoff_time:
                    audio_buffer.popleft()
                        
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from session: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Stream error: {str(e)}"
                    }))
                except:
                    pass
                break
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Stream error: {str(e)}"
            }))
        except:
            pass
    finally:
        logger.info(f"Cleaning up session: {session_id}")
        await session_manager.cleanup_session(session_id)
        
        # Force garbage collection
        gc.collect()

async def handle_control_message(data: dict, frame_manager, session: dict, websocket):
    """Handle control messages from client"""
    
    message_type = data.get("type")
    
    if message_type == "speech_start":
        await frame_manager.set_speaking_mode(True)
        logger.info(f"Speech started for session: {session['session_id']}")
        
    elif message_type == "speech_end":
        await frame_manager.set_speaking_mode(False)
        logger.info(f"Speech ended for session: {session['session_id']}")
        
    elif message_type == "stop_speaking":
        await frame_manager.reset_to_idle()
        logger.info(f"Force stop speaking for session: {session['session_id']}")
        
    elif message_type == "change_emotion":
        emotion = data.get("emotion", "neutral")
        await frame_manager.set_emotion(emotion)
        logger.info(f"Emotion changed to {emotion} for session: {session['session_id']}")
        
    elif message_type == "start_idle_animation":
        await frame_manager.start_idle_animation()
        
    elif message_type == "request_keyframe":
        # Client requesting a keyframe (for recovery)
        logger.info(f"Keyframe requested for session: {session['session_id']}")
        
    elif message_type == "quality_change":
        # Client requesting quality change
        quality = data.get("quality", "high")
        logger.info(f"Quality change requested to {quality} for session: {session['session_id']}")

def encode_frame_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Fast JPEG encoding for video frames"""
    try:
        # Ensure frame is in correct format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        # Encode with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_frame = cv2.imencode('.jpg', frame_bgr, encode_param)
        
        return encoded_frame.tobytes()
        
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return b''

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        reload=False,
        workers=1,  # Single worker for GPU sharing
        access_log=True
    )