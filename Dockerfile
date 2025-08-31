
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies as ROOT user
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user and create the app directory
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# --- START: MuseTalk Integration ---
# Clone the MuseTalk repository as ROOT first
RUN git clone https://github.com/TMElyralab/MuseTalk.git /home/appuser/app/MuseTalk
# --- END: MuseTalk Integration ---

# Copy requirements and install them as ROOT
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Install MMLab packages using mim as ROOT
RUN mim install mmengine==0.10.3
RUN mim install mmcv==2.0.1
RUN mim install mmdet==3.1.0
RUN mim install mmpose==1.1.0

# Copy the rest of the application code
COPY . .

# --- FIX: Change ownership of the entire app directory to the non-root user ---
RUN chown -R appuser:appuser /home/appuser/app

# Now, switch to the non-root user for all subsequent commands
USER appuser

# Add the cloned repository to Python's path so we can import from it
ENV PYTHONPATH "${PYTHONPATH}:/home/appuser/app/MuseTalk"

# Make the model download script executable and run it as the appuser
RUN chmod +x scripts/download_models.py
RUN python3 scripts/download_models.py

# Expose the port the application will run on
EXPOSE 8000

# Health check to ensure the service is responsive before serving traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the command to run the application
CMD ["/home/appuser/.local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "60"]