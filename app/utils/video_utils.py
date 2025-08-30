# app/utils/video_utils.py - Video Encoding and Utilities
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoEncoder:
    def __init__(self, quality: int = 85, format: str = 'JPEG'):
        self.quality = quality
        self.format = format
        
        # JPEG encoding parameters for streaming
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        # WebP encoding parameters (better compression for modern browsers)
        self.webp_params = [cv2.IMWRITE_WEBP_QUALITY, quality]

    def encode_frame(self, frame: np.ndarray, use_webp: bool = False) -> bytes:
        """Encode frame for streaming with optimal compression"""
        try:
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Ensure correct color format for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Encode based on format preference
            if use_webp:
                success, encoded = cv2.imencode('.webp', frame, self.webp_params)
            else:
                success, encoded = cv2.imencode('.jpg', frame, self.encode_params)
            
            if success:
                return encoded.tobytes()
            else:
                logger.error("Failed to encode frame")
                return b''
                
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return b''

    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame maintaining aspect ratio"""
        try:
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor to maintain aspect ratio
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize with high-quality interpolation
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size if needed
            if new_w < target_w or new_h < target_h:
                # Create black background
                padded = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
                
                # Center the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded
            
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame

    def optimize_for_streaming(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame for real-time streaming"""
        try:
            # Apply slight blur to reduce compression artifacts
            frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
            
            # Enhance contrast slightly for better compression
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error optimizing frame: {e}")
            return frame
