#!/bin/bash
# startup.sh - Fast startup script for Cloud Run

set -e

echo "Starting Avatar Video Service..."

# Get the port from environment (Cloud Run sets PORT=8080)
PORT=${PORT:-8000}

echo "Binding to port $PORT immediately..."

# Start the server in background to bind port quickly
python3.9 -c "
import socket
import threading
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Create a simple FastAPI app that binds immediately
quick_app = FastAPI()

@quick_app.get('/health')
async def quick_health():
    return {'status': 'starting', 'models_loaded': False}

@quick_app.get('/')
async def root():
    return {'message': 'Video service starting...'}

# Start server in a way that binds to port immediately
config = uvicorn.Config(quick_app, host='0.0.0.0', port=$PORT, log_level='info')
server = uvicorn.Server(config)

# This will bind to the port immediately
def start_server():
    import asyncio
    asyncio.run(server.serve())

# Start in background
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

print(f'Quick server started on port $PORT')

# Wait a moment to ensure port is bound
time.sleep(2)

# Now start the real application
print('Starting main application with model loading...')
" &

# Wait for port to be bound
sleep 3

# Kill the quick server and start the real one
pkill -f "python3.9 -c"

echo "Starting main application with model loading in background..."

# Start the real application
exec python3.9 -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1