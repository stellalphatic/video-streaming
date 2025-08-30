
# scripts/download_models.py - Model Download Script
#!/usr/bin/env python3
import os
import sys
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, filepath: str, chunk_size: int = 8192):
    """Download file with progress"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading {os.path.basename(filepath)}: {progress:.1f}%", end='')
        
        print(f"\n✅ Downloaded: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def main():
    """Download all required models"""
    
    models = {
        # MuseTalk 1.5 models
        "models/musetalk_1.5/pytorch_model.bin": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin",
        "models/musetalk_1.5/config.json": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/config.json",
        "models/musetalk_1.5/face_alignment.pth": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/face_alignment.pth",
        
        # VAE models
        "models/musetalk_1.5/vae/config.json": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json",
        "models/musetalk_1.5/vae/diffusion_pytorch_model.bin": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
        
        # Face detection models
        "models/face_detection/face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    }
    
    logger.info("Starting model download...")
    
    for filepath, url in models.items():
        if not os.path.exists(filepath):
            logger.info(f"Downloading {filepath}...")
            if not download_file(url, filepath):
                logger.error(f"Failed to download {filepath}")
                sys.exit(1)
        else:
            logger.info(f"✓ Model already exists: {filepath}")
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main()
