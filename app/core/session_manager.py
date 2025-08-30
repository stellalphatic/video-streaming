# app/core/session_manager.py - Fixed Session Management
import asyncio
import time
from typing import Dict, Optional
import logging
import aiohttp
import cv2
import numpy as np
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class SessionManager:  # Changed from EnhancedSessionManager
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.cleanup_task = None
        
        # Start cleanup task
        asyncio.create_task(self.periodic_cleanup())

    async def create_session(
        self, 
        session_id: str, 
        avatar_id: str, 
        image_url: str,
        video_processor,
        target_fps: int = 25
    ) -> bool:
        """Create new streaming session"""
        
        try:
            logger.info(f"Creating session: {session_id}")
            
            # Download and process reference image
            reference_image = await self.download_and_process_image(image_url)
            if reference_image is None:
                logger.error(f"Failed to download/process image for session {session_id}")
                return False
            
            # Verify video processor is available
            if not video_processor or not video_processor.is_initialized:
                logger.error(f"Video processor not available for session {session_id}")
                return False
            
            # Process reference image in video processor
            success = await video_processor.process_reference_image(reference_image)
            if not success:
                logger.error(f"Failed to process reference image for session {session_id}")
                return False
            
            # Create session data
            session_data = {
                'session_id': session_id,
                'avatar_id': avatar_id,
                'image_url': image_url,
                'reference_image': reference_image,
                'video_processor': video_processor,
                'audio_processor': AudioProcessor(),
                'target_fps': target_fps,
                'created_at': time.time(),
                'last_activity': time.time(),
                'is_active': True,
                'performance_stats': {
                    'frames_generated': 0,
                    'audio_chunks_processed': 0,
                    'average_processing_time': 0,
                    'total_session_time': 0
                }
            }
            
            self.active_sessions[session_id] = session_data
            logger.info(f"Session created successfully: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            return False

    async def download_and_process_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download and process reference image"""
        try:
            logger.info(f"Downloading image: {image_url}")
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        # Convert to opencv format
                        nparr = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            # Process for video generation
                            processed_image = await self.process_reference_image(image)
                            logger.info("Image downloaded and processed successfully")
                            return processed_image
                    else:
                        logger.error(f"Failed to download image: HTTP {response.status}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading/processing image: {e}")
            return None

    async def process_reference_image(self, image: np.ndarray) -> np.ndarray:
        """Process reference image for optimal video generation"""
        try:
            # Resize to standard size (512x512 for MuseTalk compatibility)
            target_size = (512, 512)
            
            # Maintain aspect ratio
            h, w = image.shape[:2]
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize with high quality
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create canvas with padding
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Convert BGR to RGB
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            
            return canvas
            
        except Exception as e:
            logger.error(f"Error processing reference image: {e}")
            return image

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        session = self.active_sessions.get(session_id)
        if session:
            # Update last activity
            session['last_activity'] = time.time()
            session['performance_stats']['total_session_time'] = time.time() - session['created_at']
            
        return session

    async def update_session_performance(self, session_id: str, stats: Dict):
        """Update session performance statistics"""
        session = self.active_sessions.get(session_id)
        if session:
            session['performance_stats'].update(stats)
            session['last_activity'] = time.time()

    async def end_session(self, session_id: str):
        """End session and cleanup"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['is_active'] = False
            
            # Log session statistics
            stats = session.get('performance_stats', {})
            logger.info(f"Session {session_id} ended - Stats: {stats}")

    async def cleanup_session(self, session_id: str):
        """Immediate cleanup of session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Clean up audio processor
            audio_processor = session.get('audio_processor')
            if audio_processor:
                audio_processor.audio_buffer = []
            
            # Note: Don't cleanup the global video processor
            # Just remove the session reference
            
            del self.active_sessions[session_id]
            logger.info(f"Session cleaned up: {session_id}")

    def get_active_count(self) -> int:
        """Get number of active sessions"""
        return len([s for s in self.active_sessions.values() if s.get('is_active', False)])

    async def periodic_cleanup(self):
        """Periodic cleanup of inactive sessions"""
        while True:
            try:
                current_time = time.time()
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    # Remove sessions inactive for more than 30 minutes
                    inactive_time = current_time - session.get('last_activity', 0)
                    
                    if inactive_time > 1800:  # 30 minutes
                        sessions_to_remove.append(session_id)
                        logger.info(f"Marking session {session_id} for cleanup - inactive for {inactive_time/60:.1f} minutes")
                
                # Clean up inactive sessions
                for session_id in sessions_to_remove:
                    await self.cleanup_session(session_id)
                
                # Clean up every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)