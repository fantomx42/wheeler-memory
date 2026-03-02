import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const uiStatus = document.getElementById('status-pill');
const uiTick = document.getElementById('val-tick');
const uiDelta = document.getElementById('val-delta');
const uiFps = document.getElementById('val-fps');
const btnEvolve = document.getElementById('btn-evolve');
const inputSeed = document.getElementById('seed-input');

let socket = null;
let lastTime = performance.now();
let framesRendered = 0;

// Three.js setup
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0d1117, 0.015);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 40, 60);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
// Use modern colorspace mapping
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.maxPolarAngle = Math.PI / 2 + 0.1; // allow slight under-view

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(50, 100, 50);
scene.add(dirLight);

const pointLight = new THREE.PointLight(0x4ade80, 2, 100);
pointLight.position.set(0, 20, 0);
scene.add(pointLight);

// Create grid using InstancedMesh for high performance
// 64x64 = 4096 cubes
const size = 64;
const totalCount = size * size;
const cubeGeometry = new THREE.BoxGeometry(0.8, 1, 0.8);

// Give the cubes a slight glossy material
const cubeMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0.1,
    roughness: 0.2,
    clearcoat: 0.8,
    clearcoatRoughness: 0.2
});

const instancedMesh = new THREE.InstancedMesh(cubeGeometry, cubeMaterial, totalCount);
instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(instancedMesh);

// Base transform offsets
const half = size / 2;
const spacing = 1.0;
const dummy = new THREE.Object3D();
const colorObj = new THREE.Color();

// Initialize grid layout flat
for (let idx = 0; idx < totalCount; idx++) {
    const i = Math.floor(idx / size);
    const j = idx % size;

    dummy.position.set(
        (i - half) * spacing,
        0,
        (j - half) * spacing
    );
    dummy.updateMatrix();
    instancedMesh.setMatrixAt(idx, dummy.matrix);

    // Default color dark grey
    colorObj.setHSL(0, 0, 0.2);
    instancedMesh.setColorAt(idx, colorObj);
}
instancedMesh.instanceMatrix.needsUpdate = true;
if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;

// Grid Helper for base reference
const gridHelper = new THREE.GridHelper(size * spacing, size, 0x3b82f6, 0xffffff);
gridHelper.material.opacity = 0.1;
gridHelper.material.transparent = true;
gridHelper.position.y = -2;
scene.add(gridHelper);


// Map [-1, 1] to height and color
// values push toward -1 (valleys) and +1 (peaks)
function updateGrid(dataArray) {
    if (!dataArray || dataArray.length !== totalCount) return;

    for (let idx = 0; idx < totalCount; idx++) {
        const val = dataArray[idx]; // [-1.0, 1.0]

        const i = Math.floor(idx / size);
        const j = idx % size;

        // Height: map [-1, 1] -> [-5, 5] space
        const height = val * 5.0;

        dummy.position.set(
            (i - half) * spacing,
            height,
            (j - half) * spacing
        );
        // Scale Y slightly to exaggerate peaks
        dummy.scale.set(1, Math.max(0.1, Math.abs(height) * 1.5 + 0.5), 1);
        dummy.updateMatrix();
        instancedMesh.setMatrixAt(idx, dummy.matrix);

        // Color mapping
        // Local max (+1) -> blue #3b82f6
        // Local min (-1) -> red/orange #ef4444
        // Slopes (~0) -> dark teal #0f766e

        // Normalize val to [0, 1]
        const norm = (val + 1.0) / 2.0;

        if (norm > 0.6) {
            colorObj.setHSL(0.6, 0.8, 0.3 + val * 0.4); // Blues
        } else if (norm < 0.4) {
            // -val because val is negative
            colorObj.setHSL(0.05, 0.8, 0.3 + (-val) * 0.4); // Oranges/Reds
        } else {
            // Flow state
            colorObj.setHSL(0.45, 0.6, 0.2); // Dark teal
        }

        instancedMesh.setColorAt(idx, colorObj);
    }

    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.instanceColor.needsUpdate = true;
}

// Animation Loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);

    // FPS counter
    framesRendered++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        uiFps.innerText = framesRendered;
        framesRendered = 0;
        lastTime = now;
    }
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});


function setStatus(className, text) {
    uiStatus.className = className;
    uiStatus.innerText = text;
}

function stopWebSocket() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }
}

function startEvolution() {
    const text = inputSeed.value.trim();
    if (!text) return;

    stopWebSocket();
    btnEvolve.disabled = true;
    btnEvolve.innerText = "Connecting...";

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/evolve`;

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        btnEvolve.innerText = "Evolving...";
        setStatus('status-running', 'EVOLVING');
        socket.send(text);
    };

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'frame') {
            uiTick.innerText = msg.tick;
            if (msg.delta !== undefined) {
                uiDelta.innerText = msg.delta.toFixed(5);
            }
            updateGrid(msg.data);
        } else if (msg.type === 'status') {
            stopWebSocket();
            btnEvolve.disabled = false;
            btnEvolve.innerText = "Evolve Memory";

            if (msg.state === 'CONVERGED') {
                setStatus('status-converged', `CONVERGED AT TICK ${msg.tick}`);
                // Stop auto rotate to inspect
                controls.autoRotate = false;
            } else {
                setStatus('status-chaotic', `CHAOTIC (${msg.tick})`);
            }
        }
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        setStatus('status-idle', 'CONNECTION ERROR');
        btnEvolve.disabled = false;
        btnEvolve.innerText = "Evolve Memory";
    };

    socket.onclose = () => {
        btnEvolve.disabled = false;
        btnEvolve.innerText = "Evolve Memory";
        if (uiStatus.innerText === 'EVOLVING') {
            setStatus('status-idle', 'DISCONNECTED');
        }
    };
}

// Controls
btnEvolve.addEventListener('click', () => {
    controls.autoRotate = true; // restart spin
    startEvolution();
});
inputSeed.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        controls.autoRotate = true;
        startEvolution();
    }
});
