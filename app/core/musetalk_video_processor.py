# app/core/musetalk_video_processor.py - Real MuseTalk Integration
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import asyncio
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import librosa
import base64
from io import BytesIO
from PIL import Image
import yaml
import pickle

# Import MuseTalk components
try:
    from musetalk.models.musetalk import MuseTalk
    from musetalk.models.audio_processor import AudioProcessor as MuseTalkAudioProcessor
    from musetalk.models.face_renderer import FaceRenderer
    from musetalk.utils.preprocessing import preprocess_image
    from musetalk.utils.face_detection import FaceAnalyzer
    MUSETALK_AVAILABLE = True
except ImportError:
    MUSETALK_AVAILABLE = False
    logging.warning("MuseTalk not available - using mock implementation")

logger = logging.getLogger(__name__)

class RealTimeMuseTalkProcessor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.face_analyzer = None
        self.face_renderer = None
        self.audio_processor = None
        self.is_initialized = False
        
        # Real-time performance settings
        self.target_fps = 25
        self.frame_interval = 1.0 / self.target_fps
        self.audio_chunk_duration = 0.04  # 40ms chunks for lower latency
        self.audio_sample_rate = 16000
        
        # Buffers and queues for real-time processing
        self.video_frame_queue = queue.Queue(maxsize=10)
        self.audio_buffer = queue.Queue(maxsize=30)  # Buffer for audio chunks
        self.mel_feature_queue = queue.Queue(maxsize=20)
        self.output_frame_queue = queue.Queue(maxsize=5)
        
        # Threading
        self.audio_processing_thread = None
        self.video_generation_thread = None
        self.frame_encoding_thread = None
        self.is_processing = False
        
        # Cache for performance
        self.reference_features = {}
        self.current_reference_image = None
        self.current_face_data = None
        
        # Synchronization
        self.audio_video_sync = threading.Lock()
        self.current_audio_timestamp = 0
        self.current_video_timestamp = 0
        
        # Performance monitoring
        self.frame_times = []
        self.processing_stats = {
            'frames_generated': 0,
            'avg_frame_time': 0,
            'fps': 0,
            'audio_latency': 0,
            'video_latency': 0
        }
        
        logger.info(f"RealTimeMuseTalkProcessor initialized on {device}")

    async def initialize(self):
        """Initialize the MuseTalk model and components"""
        try:
            logger.info("Initializing MuseTalk model...")
            
            if MUSETALK_AVAILABLE:
                # Load MuseTalk configuration
                config_path = "models/musetalk_1.5/config.yaml"
                if Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        self.config = yaml.safe_load(f)
                else:
                    self.config = self._get_default_config()
                
                # Initialize MuseTalk model
                self.model = MuseTalk(self.config).to(self.device)
                self.model.eval()
                
                # Load checkpoint
                checkpoint_path = "models/musetalk_1.5/pytorch_model.bin"
                if Path(checkpoint_path).exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("MuseTalk model loaded from checkpoint")
                
                # Initialize face analyzer
                self.face_analyzer = FaceAnalyzer(device=self.device)
                
                # Initialize face renderer
                self.face_renderer = FaceRenderer(
                    device=self.device,
                    config=self.config.get('renderer', {})
                )
                
                # Initialize audio processor
                self.audio_processor = MuseTalkAudioProcessor(
                    sample_rate=self.audio_sample_rate,
                    n_mels=80,
                    device=self.device
                )
            else:
                logger.warning("Using mock MuseTalk implementation")
                self._initialize_mock_components()
            
            # Start processing threads
            self._start_processing_threads()
            
            self.is_initialized = True
            logger.info("MuseTalk model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MuseTalk: {e}")
            raise

    def _get_default_config(self):
        """Get default MuseTalk configuration"""
        return {
            'model': {
                'audio_encoder': {
                    'hidden_size': 256,
                    'num_layers': 3
                },
                'motion_generator': {
                    'hidden_size': 512,
                    'num_layers': 6
                },
                'video_decoder': {
                    'hidden_size': 512,
                    'num_layers': 4
                }
            },
            'training': {
                'batch_size': 1,
                'learning_rate': 0.0001
            },
            'inference': {
                'chunk_size': 5,  # T=5 frames per chunk
                'overlap': 2
            }
        }

    def _initialize_mock_components(self):
        """Initialize mock components for testing without MuseTalk"""
        self.model = None
        self.face_analyzer = None
        self.face_renderer = None
        self.audio_processor = None

    def _start_processing_threads(self):
        """Start background processing threads"""
        self.is_processing = True
        
        # Audio processing thread
        self.audio_processing_thread = threading.Thread(
            target=self._audio_processing_worker,
            daemon=True
        )
        self.audio_processing_thread.start()
        
        # Video generation thread
        self.video_generation_thread = threading.Thread(
            target=self._video_generation_worker,
            daemon=True
        )
        self.video_generation_thread.start()
        
        # Frame encoding thread  
        self.frame_encoding_thread = threading.Thread(
            target=self._frame_encoding_worker,
            daemon=True
        )
        self.frame_encoding_thread.start()
        
        logger.info("Processing threads started")

    def _audio_processing_worker(self):
        """Process audio chunks into mel spectrograms"""
        audio_accumulator = []
        chunk_size = int(self.audio_sample_rate * self.audio_chunk_duration)
        
        while self.is_processing:
            try:
                # Get audio chunk from buffer
                audio_chunk = self.audio_buffer.get(timeout=0.1)
                
                # Accumulate audio
                audio_accumulator.extend(audio_chunk)
                
                # Process when we have enough audio
                if len(audio_accumulator) >= chunk_size:
                    # Extract chunk
                    chunk = np.array(audio_accumulator[:chunk_size], dtype=np.float32)
                    audio_accumulator = audio_accumulator[chunk_size:]
                    
                    # Convert to mel spectrogram
                    if MUSETALK_AVAILABLE and self.audio_processor:
                        mel_features = self.audio_processor.audio_to_mel(chunk)
                    else:
                        mel_features = self._mock_audio_to_mel(chunk)
                    
                    # Add timestamp for synchronization
                    feature_data = {
                        'mel': mel_features,
                        'timestamp': time.time(),
                        'audio_chunk': chunk
                    }
                    
                    try:
                        self.mel_feature_queue.put_nowait(feature_data)
                    except queue.Full:
                        # Drop oldest feature if queue is full
                        try:
                            self.mel_feature_queue.get_nowait()
                            self.mel_feature_queue.put_nowait(feature_data)
                        except queue.Empty:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing worker: {e}")

    def _mock_audio_to_mel(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """Mock audio to mel conversion for testing"""
        mel = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.audio_sample_rate,
            n_mels=80,
            hop_length=256,
            win_length=1024
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.from_numpy(mel_db).float()

    def _video_generation_worker(self):
        """Generate video frames from mel features"""
        mel_buffer = []
        frame_count = 0
        
        while self.is_processing:
            try:
                # Collect mel features for chunk processing
                feature_data = self.mel_feature_queue.get(timeout=0.04)
                mel_buffer.append(feature_data)
                
                # Process when we have T=5 frames worth of audio
                if len(mel_buffer) >= 5:
                    start_time = time.time()
                    
                    # Prepare batch
                    mel_batch = torch.stack([f['mel'] for f in mel_buffer[:5]]).unsqueeze(0)
                    
                    if self.current_reference_image is not None:
                        # Generate video frames
                        if MUSETALK_AVAILABLE and self.model:
                            frames = self._generate_frames_musetalk(
                                mel_batch,
                                self.current_reference_image,
                                self.current_face_data
                            )
                        else:
                            frames = self._generate_frames_mock(
                                mel_batch,
                                self.current_reference_image
                            )
                        
                        # Add frames to output queue
                        for i, frame in enumerate(frames):
                            frame_data = {
                                'frame': frame,
                                'timestamp': mel_buffer[i]['timestamp'],
                                'frame_index': frame_count + i
                            }
                            
                            try:
                                self.output_frame_queue.put_nowait(frame_data)
                            except queue.Full:
                                # Drop oldest frame
                                try:
                                    self.output_frame_queue.get_nowait()
                                    self.output_frame_queue.put_nowait(frame_data)
                                except queue.Empty:
                                    pass
                        
                        frame_count += len(frames)
                        
                        # Update stats
                        generation_time = time.time() - start_time
                        self.frame_times.append(generation_time / len(frames))
                        if len(self.frame_times) > 100:
                            self.frame_times.pop(0)
                        
                        self.processing_stats['frames_generated'] += len(frames)
                        self.processing_stats['avg_frame_time'] = np.mean(self.frame_times)
                        self.processing_stats['fps'] = 1.0 / self.processing_stats['avg_frame_time']
                    
                    # Remove processed features
                    mel_buffer = mel_buffer[5:]
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in video generation worker: {e}")

    def _generate_frames_musetalk(
        self, 
        mel_batch: torch.Tensor, 
        reference_image: np.ndarray,
        face_data: Dict
    ) -> List[np.ndarray]:
        """Generate frames using real MuseTalk model"""
        try:
            with torch.no_grad():
                # Move to device
                mel_batch = mel_batch.to(self.device)
                
                # Extract motion from audio
                motion_features = self.model.audio_encoder(mel_batch)
                
                # Generate video frames
                generated_frames = self.model.video_decoder(
                    motion_features,
                    face_data['latent_code'],
                    face_data['face_landmarks']
                )
                
                # Post-process frames
                frames = []
                for frame_tensor in generated_frames[0]:
                    # Convert to numpy
                    frame = frame_tensor.cpu().numpy().transpose(1, 2, 0)
                    frame = (frame * 255).astype(np.uint8)
                    
                    # Apply face rendering
                    rendered_frame = self.face_renderer.render(
                        frame,
                        reference_image,
                        face_data['face_landmarks']
                    )
                    
                    frames.append(rendered_frame)
                
                return frames
                
        except Exception as e:
            logger.error(f"Error generating frames with MuseTalk: {e}")
            return self._generate_frames_mock(mel_batch, reference_image)

    def _generate_frames_mock(
        self, 
        mel_batch: torch.Tensor, 
        reference_image: np.ndarray
    ) -> List[np.ndarray]:
        """Generate mock frames for testing"""
        frames = []
        
        for i in range(5):  # Generate 5 frames
            frame = reference_image.copy()
            
            # Simple animation based on audio energy
            energy = float(mel_batch[0, i].mean())
            
            # Add simple mouth movement visualization
            h, w = frame.shape[:2]
            mouth_y = int(h * 0.7)
            mouth_x = int(w * 0.5)
            mouth_height = int(10 + energy * 20)
            
            # Draw simple mouth opening
            cv2.ellipse(
                frame,
                (mouth_x, mouth_y),
                (30, mouth_height),
                0, 0, 180,
                (0, 0, 0),
                -1
            )
            
            frames.append(frame)
        
        return frames

    def _frame_encoding_worker(self):
        """Encode frames for streaming"""
        last_frame_time = 0
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Get frame from output queue
                frame_data = self.output_frame_queue.get(timeout=0.04)
                frame = frame_data['frame']
                
                # Encode frame
                encoded_frame = self._encode_frame_fast(frame)
                
                # Add to video frame queue
                try:
                    self.video_frame_queue.put_nowait(encoded_frame)
                    last_frame_time = current_time
                    
                    # Calculate latency
                    latency = current_time - frame_data['timestamp']
                    self.processing_stats['video_latency'] = latency
                    
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.video_frame_queue.get_nowait()
                        self.video_frame_queue.put_nowait(encoded_frame)
                    except queue.Empty:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in frame encoding worker: {e}")

    async def process_reference_image(self, image: np.ndarray) -> bool:
        """Process reference image for video generation"""
        try:
            self.current_reference_image = image
            
            if MUSETALK_AVAILABLE and self.face_analyzer:
                # Analyze face in reference image
                face_data = self.face_analyzer.analyze(image)
                
                if face_data is None:
                    logger.error("No face detected in reference image")
                    return False
                
                # Extract face features
                with torch.no_grad():
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    image_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(self.device)
                    
                    # Extract latent code
                    if self.model:
                        latent_code = self.model.image_encoder(image_tensor)
                        face_data['latent_code'] = latent_code
                
                self.current_face_data = face_data
            else:
                # Mock face data
                self.current_face_data = {
                    'face_landmarks': np.random.randn(468, 3),
                    'latent_code': torch.randn(1, 512).to(self.device)
                }
            
            logger.info("Reference image processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing reference image: {e}")
            return False

    async def process_audio_chunk(
        self, 
        audio_chunk: bytes, 
        reference_image: np.ndarray, 
        emotion: str = 'neutral'
    ) -> bool:
        """Process audio chunk for real-time video generation"""
        try:
            if not self.is_initialized:
                return False
            
            # Process reference image if needed
            if self.current_reference_image is None:
                await self.process_reference_image(reference_image)
            
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            if len(audio_data) > 0:
                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Add to audio buffer
                try:
                    self.audio_buffer.put_nowait(audio_data)
                    return True
                except queue.Full:
                    # Drop oldest audio if buffer is full
                    try:
                        self.audio_buffer.get_nowait()
                        self.audio_buffer.put_nowait(audio_data)
                        return True
                    except queue.Empty:
                        pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return False

    async def get_next_frame(self) -> Optional[bytes]:
        """Get next encoded video frame"""
        try:
            return self.video_frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _encode_frame_fast(self, frame: np.ndarray) -> bytes:
        """Fast frame encoding for streaming"""
        try:
            # Ensure correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Resize for performance if needed
            target_height = 512
            if frame.shape[0] != target_height:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                target_width = int(target_height * aspect_ratio)
                frame = cv2.resize(frame, (target_width, target_height))
            
            # Encode as JPEG with optimization
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            
            return encoded_frame.tobytes()
            
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return b''

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.processing_stats.copy()
        
        # Add queue sizes
        stats['audio_buffer_size'] = self.audio_buffer.qsize()
        stats['mel_queue_size'] = self.mel_feature_queue.qsize()
        stats['output_queue_size'] = self.output_frame_queue.qsize()
        stats['video_queue_size'] = self.video_frame_queue.qsize()
        
        return stats

    def stop_processing(self):
        """Stop all processing threads"""
        self.is_processing = False
        
        threads = [
            self.audio_processing_thread,
            self.video_generation_thread,
            self.frame_encoding_thread
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        
        logger.info("Processing threads stopped")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_processing()
        
        # Clear all queues
        for q in [self.audio_buffer, self.mel_feature_queue, 
                  self.output_frame_queue, self.video_frame_queue]:
            self._clear_queue(q)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("MuseTalk processor cleaned up")

    def _clear_queue(self, q: queue.Queue):
        """Clear a queue safely"""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break