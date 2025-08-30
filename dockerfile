
# Dockerfile - Optimized for GPU and Real-time Performance
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install MuseTalk from source
RUN git clone https://github.com/TMElyralab/MuseTalk.git /tmp/musetalk && \
    cd /tmp/musetalk && \
    pip install -e . && \
    cd / && rm -rf /tmp/musetalk

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/musetalk_1.5 logs assets/base_videos

# Download MuseTalk models
RUN python3 scripts/download_models.py

# Set permissions
RUN chmod +x scripts/*.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
