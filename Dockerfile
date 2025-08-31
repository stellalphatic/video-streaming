# Fixed Dockerfile for Cloud Run - Downloads models during build, not runtime
FROM python:3.10-slim

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
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

# Clone MuseTalk repository DURING BUILD
RUN git clone https://github.com/TMElyralab/MuseTalk.git /home/appuser/app/MuseTalk

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch for Cloud Run (no GPU)
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install MMLab packages
RUN pip install openmim
RUN mim install mmengine==0.10.3
RUN mim install "mmcv>=2.0.0"
RUN mim install mmdet==3.1.0
RUN mim install mmpose==1.1.0

# Copy application code
COPY . .

# Download models DURING BUILD (not runtime) - this fixes the timeout
COPY download_models.py .
RUN python download_models.py

# Change ownership to non-root user
RUN chown -R appuser:appuser /home/appuser/app

# Switch to non-root user
USER appuser

# Set Python path to include MuseTalk
ENV PYTHONPATH="/home/appuser/app/MuseTalk:${PYTHONPATH}"

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Start command - simplified for faster startup
CMD ["python", "main.py"]