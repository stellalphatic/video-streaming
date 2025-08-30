# app/core/audio_processor.py - Audio Processing for MuseTalk
import librosa
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, hop_length: int = 256):
        # MuseTalk specifications
        self.sample_rate = sample_rate  # 16kHz as required by MuseTalk
        self.hop_length = hop_length
        self.n_mels = 80  # 80-channel mel spectrogram for MuseTalk
        self.audio_buffer = []
        
        # For real-time processing - 200ms chunks (T=5 at 25fps)
        self.chunk_duration = 0.2  # 200ms chunks for optimal lip-sync
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        logger.info(f"AudioProcessor initialized: {sample_rate}Hz, {self.n_mels} mel channels")

    def process_audio_chunk(self, audio_chunk: bytes) -> Optional[np.ndarray]:
        """Process audio chunk for MuseTalk following best practices"""
        try:
            # Convert bytes to numpy array
            if len(audio_chunk) == 0:
                return None
                
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Resample to 16kHz if needed (MuseTalk requirement)
            if len(audio_data) > 0:
                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Add to buffer for context
                self.audio_buffer.extend(audio_data)
                
                # Keep only recent audio (1 second for context)
                max_buffer_size = self.sample_rate
                if len(self.audio_buffer) > max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                
                # Use buffered audio for better context
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                
                # Resample to 16kHz if not already
                if len(audio_array) > 0:
                    # Extract mel spectrogram following MuseTalk specs
                    mel_features = librosa.feature.melspectrogram(
                        y=audio_array,
                        sr=self.sample_rate,
                        n_mels=self.n_mels,
                        hop_length=self.hop_length,
                        win_length=1024,
                        window='hann',
                        center=True,
                        pad_mode='reflect',
                        power=2.0
                    )
                    
                    # Convert to log magnitude (MuseTalk requirement)
                    mel_features = librosa.power_to_db(mel_features, ref=np.max)
                    
                    # Normalize to [-1, 1] range
                    if np.max(mel_features) > np.min(mel_features):
                        mel_features = 2 * (mel_features - np.min(mel_features)) / (np.max(mel_features) - np.min(mel_features)) - 1
                    
                    # Take last T=5 frames (200ms at 25fps)
                    target_frames = 5
                    if mel_features.shape[1] >= target_frames:
                        return mel_features[:, -target_frames:]
                    else:
                        # Pad if not enough frames
                        padded = np.zeros((self.n_mels, target_frames))
                        padded[:, :mel_features.shape[1]] = mel_features
                        return padded
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

    def detect_silence(self, audio_chunk: bytes, threshold: float = 0.01) -> bool:
        """Detect if audio chunk is silence"""
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            if len(audio_data) == 0:
                return True
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            return rms_energy < threshold
        except:
            return True

    def get_audio_energy(self, audio_chunk: bytes) -> float:
        """Get audio energy level for animation intensity"""
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            if len(audio_data) == 0:
                return 0.0
            return float(np.sqrt(np.mean(audio_data ** 2)))
        except:
            return 0.0
