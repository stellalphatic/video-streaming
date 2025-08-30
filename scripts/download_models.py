
# scripts/download_models.py - Model Download Script
#!/usr/bin/env python3
import os
import sys
import requests
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hugging Face Models ---
# This structure defines the models to be downloaded from Hugging Face Hub.
HF_MODELS = [
    # MuseTalk UNet (the main model)
    {"repo_id": "TMElyralab/MuseTalk", "filename": "unet/config.json", "local_dir": "models/musetalk_1.5"},
    {"repo_id": "TMElyralab/MuseTalk", "filename": "unet/diffusion_pytorch_model.bin", "local_dir": "models/musetalk_1.5"},
    
    # MuseTalk Face Alignment
    {"repo_id": "TMElyralab/MuseTalk", "filename": "face_alignment.pth", "local_dir": "models/musetalk_1.5"},
    
    # StabilityAI VAE (used by MuseTalk)
    {"repo_id": "stabilityai/sd-vae-ft-mse", "filename": "config.json", "local_dir": "models/musetalk_1.5/vae"},
    {"repo_id": "stabilityai/sd-vae-ft-mse", "filename": "diffusion_pytorch_model.bin", "local_dir": "models/musetalk_1.5/vae"},
]

# --- Other Models (e.g., from Google Storage) ---
OTHER_MODELS = {
    "models/face_detection/face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
}

def download_from_hf(repo_id: str, filename: str, local_dir: str):
    """Downloads a file from Hugging Face Hub."""
    full_path = Path(local_dir) / Path(filename).name
    if full_path.exists():
        logger.info(f"✓ Model already exists: {full_path}")
        return True
    
    logger.info(f"Downloading {filename} from {repo_id}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {filename} from {repo_id}. Error: {e}")
        return False
    


def download_from_url(url: str, filepath: str):
    """Downloads a file from a direct URL."""
    full_path = Path(filepath)
    if full_path.exists():
        logger.info(f"✓ Model already exists: {full_path}")
        return True
        
    logger.info(f"Downloading {filepath} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Downloaded: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
    

def main():
    """Download all required models."""
    logger.info("--- Starting Model Download Process ---")
    
    all_downloads_succeeded = True

    # Download from Hugging Face
    for model in HF_MODELS:
        if not download_from_hf(model["repo_id"], model["filename"], model["local_dir"]):
            all_downloads_succeeded = False

    # Download from other URLs
    for filepath, url in OTHER_MODELS.items():
        if not download_from_url(url, filepath):
            all_downloads_succeeded = False

    if not all_downloads_succeeded:
        logger.error("One or more model downloads failed. Please check the logs.")
        sys.exit(1)
    
    logger.info("--- All models downloaded successfully! ---")

if __name__ == "__main__":
    main()