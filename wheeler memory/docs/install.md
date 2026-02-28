# Installation

## Prerequisites

- Python 3.11 or later
- `pip` (comes with Python)

## Install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Core dependencies (`numpy`, `scipy`, `matplotlib`, `psutil`) are installed automatically.

## Optional: Semantic Embedding

Semantic recall requires `sentence-transformers`. Install the `embed` extra:

```bash
pip install -e ".[embed]"
```

This pulls in `sentence-transformers>=3.0` and its transitive dependencies
(`torch`, `transformers`, etc.). The model (`all-MiniLM-L6-v2`) is downloaded
from HuggingFace on first use and cached locally.

Without this extra, `--embed` flags raise an `ImportError` at runtime.

## Optional: GPU Acceleration

GPU acceleration speeds up the cellular automata evolution step significantly.
See [GPU Acceleration](gpu.md) for benchmark numbers.

### AMD (ROCm / HIP) — primary target

```bash
# 1. Install ROCm for your distro:
#    Arch / CachyOS: sudo pacman -S rocm-hip-sdk
#    See https://rocm.docs.amd.com for other distros

# 2. Build the HIP kernel
cd wheeler_memory/gpu
make                    # default: gfx1201 (RX 9070 XT / RDNA 4)
GPU_ARCH=gfx1100 make  # RDNA 3 (RX 7000 series)
```

The Makefile calls `hipcc` to compile `libwheeler_ca.so`. After a successful
build, `gpu_available()` returns `True` and the GPU path is used automatically
for batch operations.

Tested on an **RX 9070 XT** (gfx1201, RDNA 4) under CachyOS with ROCm.

### NVIDIA (CUDA)

Install a CUDA-enabled PyTorch build matching your driver:

```bash
# CUDA 12.4 example — check https://pytorch.org for the right URL
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`get_optimal_device()` then returns `"cuda"` and `sentence-transformers` will
use the GPU automatically.

## Data Directory

Wheeler Memory creates `~/.wheeler_memory/` on first use. No manual setup is
required. The directory layout after storing some memories:

```
~/.wheeler_memory/
├── chunks/
│   ├── code/
│   │   ├── attractors/    # one .npy per memory (64×64 float32 attractor)
│   │   ├── bricks/        # one .npz per memory (full evolution timeline)
│   │   ├── index.json     # text → hex_key metadata
│   │   └── metadata.json  # per-chunk last_accessed, store_count
│   ├── hardware/
│   ├── daily_tasks/
│   ├── science/
│   ├── meta/
│   └── general/           # default catch-all chunk
└── rotation_stats.json    # per-angle convergence counts
```

See [Architecture](architecture.md) for details on chunks and bricks.

## Verify the Installation

```bash
wheeler-info
```

Sample output:

```
OS:      Linux 6.19.0-rc6-1-cachyos-rc-lto
CPU:     x86_64 — 16 physical / 32 logical cores @ 5200 MHz
RAM:     64.00 GB total, 48.23 GB available
Storage: / — 931.51 GB total, 212.44 GB used (22.8%)
GPU/NPU: [00:03.0 VGA compatible controller: Advanced Micro Devices, Inc.
          [AMD/ATI] Radeon RX 9070 XT]
Device:  cpu   ← becomes "cuda" or "mps" when GPU is fully configured
Warnings:
  Discrete GPU detected but PyTorch is using CPU.
  Verify CUDA/ROCm installation matches your hardware.
```

If `Device` shows `cpu` but you have a GPU, follow the steps in
[GPU Acceleration](gpu.md) to resolve the mismatch.

## Available CLI Commands

| Command | Description |
|---|---|
| `wheeler-store` | Encode and store a memory |
| `wheeler-recall` | Search for stored memories |
| `wheeler-info` | Print hardware and device summary |
| `wheeler-bench-gpu` | Run GPU vs CPU benchmark |
| `wheeler-temps` | List memories with temperature tiers |
| `wheeler-scrub` | Inspect a brick evolution timeline |
| `wheeler-diversity` | Test attractor diversity across inputs |

See [CLI Reference](cli.md) for full flag documentation.
