#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import gdown
import requests

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_MODELS_DIR = Path("models")

# This structure is based on the official download_weights.sh script
MODELS_TO_DOWNLOAD = {
    "musetalk_v1": {
        "type": "hf",
        "repo_id": "TMElyralab/MuseTalk",
        "files": ["musetalk/musetalk.json", "musetalk/pytorch_model.bin"],
        "local_dir": BASE_MODELS_DIR
    },
    "musetalk_v1.5": {
        "type": "hf",
        "repo_id": "TMElyralab/MuseTalk",
        "files": ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
        "local_dir": BASE_MODELS_DIR
    },
    "sd_vae": {
        "type": "hf",
        "repo_id": "stabilityai/sd-vae-ft-mse",
        "files": ["config.json", "diffusion_pytorch_model.bin"],
        "local_dir": BASE_MODELS_DIR / "sd-vae"
    },
    "whisper": {
        "type": "hf",
        "repo_id": "openai/whisper-tiny",
        "files": ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
        "local_dir": BASE_MODELS_DIR / "whisper"
    },
    "dwpose": {
        "type": "hf",
        "repo_id": "yzd-v/DWPose",
        "files": ["dw-ll_ucoco_384.pth"],
        "local_dir": BASE_MODELS_DIR / "dwpose"
    },
    "syncnet": {
        "type": "hf",
        "repo_id": "ByteDance/LatentSync",
        "files": ["latentsync_syncnet.pt"],
        "local_dir": BASE_MODELS_DIR / "syncnet"
    },
    "face_parse_bisent": {
        "type": "gdown",
        "id": "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
        "output": BASE_MODELS_DIR / "face-parse-bisent" / "79999_iter.pth"
    },
    "resnet18": {
        "type": "url",
        "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "output": BASE_MODELS_DIR / "face-parse-bisent" / "resnet18-5c106cde.pth"
    }
}

def download_from_hf(repo_id: str, filename: str, local_dir: Path):
    """Downloads a file from Hugging Face Hub."""
    try:
        logger.info(f"Downloading {filename} from {repo_id}...")
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

def download_from_gdown(file_id: str, output_path: Path):
    """Downloads a file from Google Drive using gdown."""
    try:
        logger.info(f"Downloading {output_path.name} from Google Drive...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=file_id, output=str(output_path), quiet=False)
        logger.info(f"✅ Successfully downloaded {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download from gdown (ID: {file_id}). Error: {e}")
        return False

def download_from_url(url: str, output_path: Path):
    """Downloads a file from a direct URL."""
    try:
        logger.info(f"Downloading {output_path.name} from {url}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"✅ Successfully downloaded {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download from URL {url}. Error: {e}")
        return False

def main():
    """Download all required models based on the official scripts."""
    logger.info("--- Starting Model Download Process ---")
    all_downloads_succeeded = True

    for model_name, details in MODELS_TO_DOWNLOAD.items():
        if details["type"] == "hf":
            for f in details["files"]:
                if not download_from_hf(details["repo_id"], f, details["local_dir"]):
                    all_downloads_succeeded = False
        elif details["type"] == "gdown":
            if not download_from_gdown(details["id"], details["output"]):
                all_downloads_succeeded = False
        elif details["type"] == "url":
            if not download_from_url(details["url"], details["output"]):
                all_downloads_succeeded = False

    if not all_downloads_succeeded:
        logger.error("One or more model downloads failed. Please check the logs.")
        sys.exit(1)
    
    logger.info("--- All models downloaded successfully! ---")

if __name__ == "__main__":
    main()
