# app/core/frame_manager.py - Frame Management and Animation
import cv2
import numpy as np
import time
import asyncio
from typing import Optional, List, Dict
import logging
import math

logger = logging.getLogger(__name__)

class FrameManager:
    def __init__(self, session: dict):
        self.session = session
        self.is_speaking = False
        self.current_emotion = "neutral"
        self.idle_frame_index = 0
        self.last_frame_time = time.time()
        self.target_fps = 25  # MuseTalk standard
        self.frame_interval = 1.0 / self.target_fps
        
        # Load base animations
        self.idle_animations = self.load_idle_animations()
        self.current_base_frame = None
        
        # Animation parameters
        self.animation_speed = 1.0
        self.head_movement_amplitude = 2.0  # degrees
        self.blink_frequency = 0.1  # blinks per second

    def load_idle_animations(self) -> Dict[str, List[np.ndarray]]:
        """Load pre-recorded idle animation frames"""
        animations = {
            'neutral': self.generate_neutral_movements(),
            'happy': self.generate_happy_movements(),
            'thoughtful': self.generate_thoughtful_movements(),
            'excited': self.generate_excited_movements()
        }
        return animations

    def generate_neutral_movements(self) -> List[np.ndarray]:
        """Generate subtle neutral head movement frames"""
        base_frame = self.session['reference_image'].copy()
        frames = []
        
        # Generate 2 seconds of animation at 25fps
        for i in range(50):
            frame = base_frame.copy()
            
            # Subtle head movement
            timestamp = i / 25.0  # Convert to seconds
            head_x = math.sin(timestamp * 0.5) * 1.0  # Very subtle horizontal movement
            head_y = math.cos(timestamp * 0.3) * 0.5  # Very subtle vertical movement
            
            # Apply subtle transformation
            frame = self.apply_head_movement(frame, head_x, head_y)
            
            # Random blink
            if np.random.random() < 0.02:  # 2% chance per frame
                frame = self.apply_blink(frame)
            
            frames.append(frame)
            
        return frames

    def generate_happy_movements(self) -> List[np.ndarray]:
        """Generate happy emotion movements"""
        base_frame = self.session['reference_image'].copy()
        frames = []
        
        for i in range(50):
            frame = base_frame.copy()
            
            # More animated movement for happy emotion
            timestamp = i / 25.0
            head_x = math.sin(timestamp * 0.8) * 1.5
            head_y = math.cos(timestamp * 0.6) * 1.0
            
            frame = self.apply_head_movement(frame, head_x, head_y)
            
            # More frequent blinking when happy
            if np.random.random() < 0.03:
                frame = self.apply_blink(frame)
            
            frames.append(frame)
            
        return frames

    def generate_thoughtful_movements(self) -> List[np.ndarray]:
        """Generate thoughtful emotion movements"""
        base_frame = self.session['reference_image'].copy()
        frames = []
        
        for i in range(50):
            frame = base_frame.copy()
            
            # Slower, more deliberate movement
            timestamp = i / 25.0
            head_x = math.sin(timestamp * 0.3) * 0.8
            head_y = math.cos(timestamp * 0.2) * 0.6
            
            frame = self.apply_head_movement(frame, head_x, head_y)
            
            # Less frequent blinking when thoughtful
            if np.random.random() < 0.015:
                frame = self.apply_blink(frame)
            
            frames.append(frame)
            
        return frames

    def generate_excited_movements(self) -> List[np.ndarray]:
        """Generate excited emotion movements"""
        base_frame = self.session['reference_image'].copy()
        frames = []
        
        for i in range(50):
            frame = base_frame.copy()
            
            # More energetic movement
            timestamp = i / 25.0
            head_x = math.sin(timestamp * 1.2) * 2.0
            head_y = math.cos(timestamp * 1.0) * 1.5
            
            frame = self.apply_head_movement(frame, head_x, head_y)
            
            # More frequent blinking when excited
            if np.random.random() < 0.04:
                frame = self.apply_blink(frame)
            
            frames.append(frame)
            
        return frames

    def apply_head_movement(self, frame: np.ndarray, head_x: float, head_y: float) -> np.ndarray:
        """Apply subtle head movement to frame"""
        try:
            h, w = frame.shape[:2]
            
            # Create transformation matrix for subtle movement
            tx = head_x
            ty = head_y
            
            # Translation matrix
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            # Apply transformation
            moved_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            return moved_frame
            
        except Exception as e:
            logger.error(f"Error applying head movement: {e}")
            return frame

    def apply_blink(self, frame: np.ndarray) -> np.ndarray:
        """Apply blink effect to frame"""
        try:
            # This is a simplified blink - in real implementation,
            # you would detect eye regions and modify them
            # For now, just return the frame as-is
            return frame
            
        except Exception as e:
            logger.error(f"Error applying blink: {e}")
            return frame

    async def get_current_base_frame(self) -> np.ndarray:
        """Get current base frame for lip sync"""
        if self.is_speaking:
            # Use reference image when speaking for lip sync
            return self.session['reference_image']
        else:
            # Use idle animation frame
            return await self.get_next_idle_frame()

    async def get_next_idle_frame(self) -> np.ndarray:
        """Get next frame from idle animation"""
        try:
            current_time = time.time()
            
            # Control frame rate
            if current_time - self.last_frame_time < self.frame_interval:
                # Return cached frame if too early
                if self.current_base_frame is not None:
                    return self.current_base_frame
            
            # Get animation frames
            animation_frames = self.idle_animations.get(
                self.current_emotion, 
                self.idle_animations['neutral']
            )
            
            # Cycle through frames
            frame = animation_frames[self.idle_frame_index % len(animation_frames)]
            self.idle_frame_index += 1
            
            self.current_base_frame = frame
            self.last_frame_time = current_time
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting idle frame: {e}")
            return self.session['reference_image']  # Fallback

    async def set_speaking_mode(self, speaking: bool):
        """Set speaking mode"""
        self.is_speaking = speaking
        if not speaking:
            # Reset to idle
            self.idle_frame_index = 0

    async def set_emotion(self, emotion: str):
        """Change avatar emotion"""
        if emotion in self.idle_animations:
            self.current_emotion = emotion
            self.idle_frame_index = 0  # Reset animation
            logger.info(f"Emotion changed to: {emotion}")

    async def start_idle_animation(self):
        """Start idle animation"""
        self.is_speaking = False
        self.idle_frame_index = 0

    async def reset_to_idle(self):
        """Reset to idle state"""
        self.is_speaking = False
        self.idle_frame_index = 0
        self.current_emotion = "neutral"
