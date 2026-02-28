# GPU Acceleration

Wheeler Memory supports GPU-accelerated CA evolution via **HIP kernels** on AMD GPUs
(ROCm) and via **PyTorch CUDA** on NVIDIA GPUs.

---

## AMD (ROCm / HIP) — primary target

### Requirements

- AMD GPU with ROCm support
- `hipcc` compiler (ships with `rocm-hip-sdk`)
- Tested on: **AMD Radeon RX 9070 XT** (gfx1201, RDNA 4) under CachyOS + ROCm

### Install ROCm

```bash
# Arch / CachyOS
sudo pacman -S rocm-hip-sdk

# Ubuntu / Debian — see https://rocm.docs.amd.com/en/latest/deploy/linux/
```

### Build the HIP kernel

```bash
cd wheeler_memory/gpu
make v2                    # recommended: v2 (variable grid, global memory)
make                       # v1 only (64×64 fixed grid, legacy)
make all                   # build both v1 and v2

GPU_ARCH=gfx1100 make v2  # RDNA 3 (RX 7000 series)
GPU_ARCH=gfx906  make v2  # Vega / RX 5000 series
```

This produces `wheeler_memory/gpu/libwheeler_ca_v2.so` (or `libwheeler_ca.so`
for v1). The Python bindings in `wheeler_memory/gpu_dynamics.py` try v2 first,
falling back to v1 if only v1 is built.

After a successful build, `gpu_available()` returns `True` and batch evolution
automatically uses the GPU. Call `gpu_version()` to confirm which kernel loaded.

---

## NVIDIA (CUDA)

Wheeler Memory uses PyTorch's CUDA support for NVIDIA GPUs. Install a
CUDA-enabled PyTorch build matching your driver version:

```bash
# CUDA 12.4 example
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check https://pytorch.org/get-started/locally/ for the correct URL for your
driver. Once installed, `get_optimal_device()` returns `"cuda"` and
`sentence-transformers` will use the GPU for embedding inference automatically.

The HIP kernel (`libwheeler_ca.so`) is AMD-specific. NVIDIA users get GPU
acceleration for embedding but CA evolution runs on CPU unless you adapt the
kernel to CUDA (not currently supported).

---

## Auto-detection

`get_optimal_device()` in `wheeler_memory/hardware.py` selects the best
available accelerator at runtime:

```python
from wheeler_memory.hardware import get_optimal_device

device = get_optimal_device()
# "cuda"  — NVIDIA GPU with CUDA
# "mps"   — Apple Silicon GPU
# "cpu"   — no usable accelerator found
```

The embedding model (`embed_to_frame`) uses this device automatically when
`sentence-transformers` is installed.

Run `wheeler-info` to see what the system detects:

```bash
wheeler-info
```

```
OS:      Linux 6.19.0-rc6-1-cachyos-rc-lto
CPU:     x86_64 — 16 physical / 32 logical cores @ 5200 MHz
RAM:     64.00 GB total, 48.23 GB available
Storage: / — 931.51 GB total, 212.44 GB used (22.8%)
GPU/NPU: [00:03.0 VGA compatible controller: AMD/ATI Radeon RX 9070 XT]
Device:  cpu
Warnings:
  Discrete GPU detected but PyTorch is using CPU.
  Verify CUDA/ROCm installation matches your hardware.
```

---

## Benchmark

Batch CA evolution on an RX 9070 XT vs. CPU (single-threaded):

| Batch size | CPU (s) | GPU (s) | Speedup | GPU samples/s |
|---|---|---|---|---|
| 1 | 0.0024 | 0.0021 | 1.1× | 472 |
| 10 | 0.0221 | 0.0085 | 2.6× | 1,172 |
| 100 | 0.2388 | 0.0055 | **43.8×** | 18,321 |
| 500 | 1.1904 | 0.0205 | **58.1×** | 24,423 |
| 1,000 | 2.3838 | 0.0337 | **70.7×** | 29,648 |

> The first GPU call has a ~250 ms cold-start (HIP runtime initialisation).
> Subsequent calls run in single-digit milliseconds.

Single inputs are roughly equivalent to CPU. GPU shines for batch processing —
diversity tests, bulk imports, and benchmark suites.

---

## Mismatch warnings

`check_software_hardware_mismatch()` returns a list of warning strings when it
detects a gap between the hardware and what PyTorch is using:

| Situation | Warning |
|---|---|
| Discrete AMD/NVIDIA GPU in `lspci` but PyTorch uses CPU | `"Discrete GPU detected but PyTorch is using CPU. Verify CUDA/ROCm installation matches your hardware."` |
| PyTorch not installed at all | `"PyTorch not installed. Hardware acceleration unavailable."` |

### How to fix

**AMD GPU showing as CPU:**
1. Confirm ROCm is installed: `rocminfo` should list your GPU.
2. Confirm `hipcc` is on `$PATH`: `hipcc --version`.
3. Build the kernel: `cd wheeler_memory/gpu && make`.
4. Verify: `wheeler-info` — `Device` should still show `cpu` (the HIP kernel
   is separate from PyTorch's device selection) but `gpu_available()` in Python
   will return `True` after the build.

**NVIDIA GPU showing as CPU:**
1. Confirm the driver: `nvidia-smi`.
2. Reinstall PyTorch with the CUDA wheel that matches your driver version.
3. Rerun `wheeler-info`.

---

## `wheeler-bench-gpu` CLI

```bash
# Default benchmark (batch sizes 1, 10, 100, 500, 1000)
wheeler-bench-gpu

# Correctness check only — no timing
wheeler-bench-gpu --verify-only

# Custom batch sizes
wheeler-bench-gpu --batch-sizes 100,500,2000,5000
```

The benchmark prints a table comparing CPU and GPU throughput and verifies
numerical correctness with `np.allclose(atol=1e-4)`.

---

## Python API

```python
from wheeler_memory import gpu_available, gpu_evolve_batch, gpu_evolve_single
from wheeler_memory.gpu_dynamics import gpu_version, gpu_query_vram
from wheeler_memory.hashing import hash_to_frame

if gpu_available():
    print(f"GPU kernel v{gpu_version()} loaded")  # v2 or v1

    # Single input (64×64 default)
    frame = hash_to_frame("some text")
    result = gpu_evolve_single(frame)
    print(result["state"], result["convergence_ticks"])
    print(result["metadata"])  # {"backend": "gpu_v2", "grid_w": 64}

    # Batch (where the speedup is)
    texts = ["memory one", "memory two", "memory three"]
    frames = [hash_to_frame(t) for t in texts]
    results = gpu_evolve_batch(frames)
    for r in results:
        print(r["state"])

    # v2: variable grid size
    import numpy as np
    big_frame = np.random.rand(256, 256).astype(np.float32)
    result = gpu_evolve_single(big_frame, grid_w=256)

    # v2: VRAM estimation before allocating a large batch
    vram = gpu_query_vram(batch_size=1000, grid_w=256)
    if vram:
        print(f"~{vram / 1e6:.1f} MB needed")
else:
    print("GPU kernel not built — run: cd wheeler_memory/gpu && make v2")
```

---

## Architecture of the HIP kernel (v2 — current)

```
Input frames  (B × W × W)          W = any grid width (64 … 1000+)
    ↓
hipMemcpy → GPU device memory
    ↓  (iteration loop)
Kernel 1 — ca_step_kernel: B×W² threads (16×16 blocks, 2D spatial tiling)
  each thread: read 4 neighbours → apply 3-state rule → atomic delta accumulation
    ↓
Kernel 2 — convergence_check_kernel: mean |Δ| < stability_threshold → CONVERGED
    ↓
Kernel 3 — role_compute_kernel: classify cells → write to 20-slot ring buffer
    ↓
Kernels 4+5 — oscillation_detect / finalize: period-2…10 detection (every 10 iters)
    ↓
(poll for all-done every 50 iterations)
    ↓
hipMemcpy → CPU result arrays
```

V2 uses global memory and a ping-pong grid layout, removing the 64×64 shared-memory
limit of v1. The ring buffer enables oscillation detection across large grids.

### v1 architecture (legacy fallback)

```
Input frames  (B × 64 × 64)        fixed 64×64 grid only
    ↓
hipMemcpy → GPU device memory
    ↓
ca_step_kernel: B × 4096 threads (shared-memory reduction)
    ↓
reduce_delta_kernel: mean |Δ| per frame
    ↓
check compiled-in threshold → repeat or stop
    ↓
hipMemcpy → CPU result arrays
```

Both GPU paths produce numerically identical results to the CPU path
(verified via `np.allclose` with `atol=1e-4`).

### Key files

| File | Purpose |
|---|---|
| `wheeler_memory/gpu/ca_kernel_v2.hip` | HIP C++ kernel v2 (variable grid, global memory) |
| `wheeler_memory/gpu/ca_kernel.hip` | HIP C++ kernel v1 (64×64, shared memory — legacy) |
| `wheeler_memory/gpu/Makefile` | Build script |
| `wheeler_memory/gpu/libwheeler_ca_v2.so` | Compiled v2 library (not in git — build with `make v2`) |
| `wheeler_memory/gpu/libwheeler_ca.so` | Compiled v1 library (not in git — build with `make`; verified working on RX 9070 XT) |
| `wheeler_memory/gpu_dynamics.py` | Python `ctypes` bindings (loads v2, falls back to v1) |
| `scripts/bench_gpu.py` | `wheeler-bench-gpu` entry point |

---

## Building v2

```bash
cd wheeler_memory/gpu
make v2                    # gfx1201 (RX 9070 XT)
GPU_ARCH=gfx1100 make v2  # RDNA 3 (RX 7000 series)
```

The Python bindings load `libwheeler_ca_v2.so` first. If found, v2 is used
automatically — no code changes required. `gpu_version()` returns `2`.

### Variable grid size (v2 only)

```python
from wheeler_memory import gpu_evolve_batch
from wheeler_memory.gpu_dynamics import gpu_version, gpu_query_vram
import numpy as np

print(gpu_version())   # 2

# 256×256 grid — not possible with v1
big_frame = np.random.rand(256, 256).astype(np.float32)
result = gpu_evolve_batch([big_frame], grid_w=256)
print(result[0]["state"])

# Estimate VRAM before allocating
vram = gpu_query_vram(batch_size=500, grid_w=256)
print(f"~{vram / 1e6:.1f} MB needed")
```

---

## Fallback

If neither `libwheeler_ca_v2.so` nor `libwheeler_ca.so` is built, `gpu_available()`
returns `False` and all CPU functionality works unchanged. The GPU backend is
strictly opt-in — nothing breaks if you skip the build step.
