# Corrected Dockerfile for Cloud Run with GPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for Python, Git, and OpenCV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for better security
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
USER appuser

# Copy requirements file and install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# --- START: MuseTalk Integration ---
# Clone the MuseTalk repository because it's not an installable package
RUN git clone https://github.com/TMElyralab/MuseTalk.git
# Add the cloned repository to Python's path so we can import from it
ENV PYTHONPATH "${PYTHONPATH}:/home/appuser/app/MuseTalk"
# --- END: MuseTalk Integration ---

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Make the model download script executable and run it
# This script will download models into the directories your app expects
RUN chmod +x scripts/download_models.py
RUN python3 scripts/download_models.py

# Expose the port the application will run on
EXPOSE 8000

# Health check to ensure the service is responsive before serving traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the command to run the application
CMD ["/home/appuser/.local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "60"]