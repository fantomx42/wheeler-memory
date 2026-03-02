#!/bin/bash

# 1. Ensure Host Ollama is running (for GPU support)
if ! systemctl is-active --quiet ollama; then
    echo "Starting Host Ollama..."
    sudo systemctl start ollama
fi

# 2. Run your optimized setup (GPU, No-Timeout, Bridge IP)
echo "Configuring Open WebUI container..."
bash "$(dirname "$0")/setup_open_webui.sh"

# 3. Wait for the service to wake up (max 30 seconds)
echo "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null; then
        echo "Service is ready!"
        break
    fi
    sleep 1
done

# 4. Open the browser
xdg-open http://localhost:3000
