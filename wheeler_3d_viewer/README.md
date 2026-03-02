# Wheeler Memory - 3D Local CA Viewer

This is a standalone, containerized FastAPI & Three.js application designed to view the cellular automata evolution of the Wheeler Memory core system in realtime 3D.

## Architecture
- **Backend**: FastAPI + WebSockets to iteratively compute the CA grid asynchronously and stream it to the browser.
- **Frontend**: Three.js `InstancedMesh` handles drawing 4,096 realtime animated cubes at 60 FPS without stressing the CPU.
- **Integration**: The viewer is containerized. It mounts the `wheeler_memory` original source code structure strictly as a Read-Only volume at runtime, completely preventing any unintended changes to the core memory project.

## How to Run

1. **Build the Docker Image**:
```bash
cd ~/wheeler_3d_viewer
docker build -t wheeler-3d-viewer .
```

2. **Run the Container**:
Be sure to mount your `wheeler memory` directory.
```bash
docker run -p 8000:8000 -v "/home/tristan/wheeler memory:/app/wheeler_memory:ro" wheeler-3d-viewer
```

3. **Open Browser**:
Navigate to `http://localhost:8000`

## Usage
- Enter any text seed into the input box on the top left.
- Click **Evolve Memory**.
- The 64x64 grid will dynamically deform. **Local maximums** push upward (+1) glowing blue, **local minimums** push downward (-1) glowing red, and slopes flow across in dark teal.
- Use your **mouse** (Left click to orbit, Right click to pan, Scroll to zoom) to view the windtunnel space in complete 3D 360 degrees.
