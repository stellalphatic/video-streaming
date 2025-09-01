# Fixed Dockerfile for Cloud Run - Downloads models during build, not runtime
FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

# --- THIS IS THE CRUCIAL NEW LINE ---
# Install MuseTalk directly from the Git repository
RUN pip install git+https://github.com/TMElyralab/MuseTalk.git

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install MMLab packages
RUN pip install openmim
RUN mim install mmengine==0.10.3
RUN mim install "mmcv>=2.0.0"
RUN mim install mmdet==3.1.0
RUN mim install mmpose==1.1.0

# Copy application code
COPY . .

# Download models DURING BUILD (not runtime) - this fixes the timeout
COPY scripts/download_models.py .
RUN python download_models.py

# Change ownership to non-root user
RUN chown -R appuser:appuser /home/app/

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]