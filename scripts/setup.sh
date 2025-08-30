# scripts/setup.sh - Environment Setup Script
#!/bin/bash

echo "🚀 Setting up Avatar Video Service..."

# Create directories
mkdir -p models/musetalk_1.5/vae
mkdir -p models/face_detection
mkdir -p assets/base_videos
mkdir -p logs
mkdir -p cache

# Download models
echo "📦 Downloading AI models..."
python3 scripts/download_models.py

# Set permissions
chmod 755 scripts/*.py
chmod 755 scripts/*.sh

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment configuration..."
    cat > .env << EOL
# Video Service Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0

# Performance Settings
MAX_CONCURRENT_SESSIONS=10
TARGET_FPS=25
FRAME_QUALITY=85
USE_HALF_PRECISION=true
ENABLE_CACHING=true

# Model Settings
MODEL_PATH=models/musetalk_1.5
FACE_DETECTION_MODEL=models/face_detection/face_landmarker.task

# AWS Settings (if using)
AWS_REGION=us-east-1
S3_BUCKET=your-avatar-assets
EOL
fi

echo "✅ Setup complete!"
echo "📋 Next steps:"
echo "   1. Build Docker image: docker build -t avatar-video-service ."
echo "   2. Run locally: docker-compose up"
echo "   3. Test health: curl http://localhost:8000/health"