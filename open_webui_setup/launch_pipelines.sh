#!/bin/bash

# Configuration
PORT=9099
WHEELER_PATH="/home/tristan/wheeler memory"
DATA_PATH="/home/tristan/.wheeler_memory"

echo "Starting Open WebUI Pipelines on port $PORT..."

# Create pipelines dir if not exists
mkdir -p "$(dirname "$0")/pipelines"

# We mount:
# 1. The pipelines directory to /app/pipelines
# 2. The local wheeler_memory source to /app/wheeler_memory (so we can import it)
# 3. The actual memory data to /app/data/.wheeler_memory (where memories live)

sudo docker run -d -p $PORT:9099 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(dirname "$0")/pipelines":/app/pipelines \
  -v "$WHEELER_PATH":/app/wheeler_memory \
  -v "$DATA_PATH":/app/data/.wheeler_memory \
  --name open-webui-pipelines \
  --restart always \
  ghcr.io/open-webui/pipelines:main

echo "Pipelines running at http://localhost:$PORT"
echo "Don't forget to add this URL to Open WebUI settings -> Admin Settings -> Connections"
