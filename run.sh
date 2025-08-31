#!/bin/bash

# Cloud Run defaults to PORT=8080. Uvicorn will listen on this port.
# The `run.sh` script should run the main application in the foreground.
/home/appuser/.local/bin/uvicorn main:app --host 0.0.0.0 --port "$PORT"