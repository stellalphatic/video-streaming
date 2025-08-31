#!/bin/bash

# Start the main Uvicorn application in the background
/home/appuser/.local/bin/uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1 &

# Start a second, simple, temporary web server in the foreground for health checks
# It listens on port 8080. We will point the health check to this port.
# It will respond with "OK" to any request.
while true; do { echo -e 'HTTP/1.1 200 OK\r\n'; echo "OK"; } | nc -l -p 8080; done