import asyncio
import json
import logging
import sys
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append("/app/wheeler_memory")
try:
    from wheeler_memory.hashing import hash_to_frame
    from wheeler_memory.dynamics import apply_ca_dynamics
    logger.info("Successfully imported wheeler_memory components.")
except ImportError as e:
    logger.error(f"Failed to import wheeler_memory: {e}")
    # Fallback dummies for stand-alone testing if wheeler_memory not mounted properly
    def hash_to_frame(*args, **kwargs): return np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
    def apply_ca_dynamics(f): return np.clip(f + np.random.uniform(-0.1, 0.1, f.shape), -1, 1)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="viewer/static"), name="static")

@app.get("/")
async def get():
    with open("viewer/static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/evolve")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /ws/evolve")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received request to evolve matching: {data[:50]}...")
            
            # Setup
            frame = hash_to_frame(data)
            max_iters = 1000
            stability_threshold = 1e-4
            
            # Send initial frame
            await websocket.send_text(json.dumps({
                "type": "frame",
                "tick": 0,
                "data": frame.flatten().tolist()
            }))
            await asyncio.sleep(0.05) # Add small delay so ui can catch up

            # Evolve loop
            for i in range(1, max_iters + 1):
                frame_old = frame.copy()
                frame = apply_ca_dynamics(frame)
                delta = np.abs(frame - frame_old).mean()
                
                # We can skip sending every single frame if bandwidth is an issue, 
                # but let's try every frame for smoothest 60 FPS look, 
                # or every 2nd frame depending on performance.
                await websocket.send_text(json.dumps({
                    "type": "frame",
                    "tick": i,
                    "data": frame.flatten().tolist(),
                    "delta": float(delta)
                }))
                
                # Yield control to event loop so websocket can flush
                await asyncio.sleep(0.016) # ~60 fps target max rate
                
                if delta < stability_threshold:
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "state": "CONVERGED",
                        "tick": i
                    }))
                    break
            else:
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "state": "CHAOTIC",
                    "tick": max_iters
                }))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
