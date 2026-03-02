#!/bin/bash
echo "Stopping and removing existing Open WebUI container..."
sudo docker rm -f open-webui 2>/dev/null

echo "Starting Open WebUI connected to host Ollama (GPU accelerated)..."
# Using the Docker bridge IP (172.17.0.1) for reliable communication
# AIOHTTP_CLIENT_TIMEOUT='' disables the timeout, allowing slow models like Coder Next 80B to finish.
sudo docker run -d -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://172.17.0.1:11434 \
  -e AIOHTTP_CLIENT_TIMEOUT='' \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

echo "Setup complete! Open http://localhost:3000 in your browser."
