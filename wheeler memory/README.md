# Project Darman: Wheeler Memory

**A memory system that remembers like you do — imperfectly, associatively, and influenced by context.**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## What Is This?

Wheeler Memory stores text by running it through a cellular automaton — a self-organising grid that evolves until it settles into a stable pattern (an *attractor*). Each input produces a unique attractor that serves as its fingerprint. When you search, your query is evolved the same way, and the closest fingerprints are returned.

Memories fade over time if you don't use them. Frequently recalled memories stay warm; stale ones cool down and can be archived. When you recall a memory, it comes back slightly shaped by your current context — the same stored thought reconstructs differently depending on what you're thinking about today.

**Darman doesn't retrieve. Darman reconstructs.**

---

## Quick Start

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
pip install -e .
wheeler-ui
```

Open **http://localhost:7437** in your browser. That's it.

> **You do NOT need a GPU.** CPU works fine. Python 3.11+ required.
> Want to understand how it works? Open `ui/demo.html` in your browser for an interactive walkthrough.

---

## What Does It Look Like?

<img src="docs/assets/diagrams/evolution.gif" alt="CA evolution — a random grid converging to a stable attractor" width="320">

---

## Want Fuzzy Search?

```bash
pip install -e ".[embed]"
```

This enables `--embed` / semantic search — finds memories by meaning, not just exact wording. (Pulls in `sentence-transformers` + PyTorch, ~1–2 GB.)

---

## CLI Cheat Sheet

| Command | What it does |
|---|---|
| `wheeler-store "text"` | Save a memory |
| `wheeler-recall "text"` | Find similar memories |
| `wheeler-ui` | Open the web dashboard |
| `wheeler-temps` | See all memories and their freshness |
| `wheeler-forget --text "text"` | Delete a specific memory |
| `wheeler-sleep` | Compress old memories to save space |
| `wheeler-agent` | Start the AI chat agent (needs Ollama) |
| `wheeler-info` | Show your system info |
| `wheeler-scrub --text "text"` | Visualise how a memory formed |
| `wheeler-bench-gpu` | Benchmark GPU vs CPU speed |

---

## How It Works

```
Input → SHA-256 → CA_seed (64×64 float) → CA_evolution → Attractor
                                                  ↓
                                     Temperature tracking (wall-clock)
                                                  ↓
                         Recall(query) → Pearson correlation search
                                                  ↓
                              [optional] Blend + re-evolve → Reconstruction
```

The cellular automaton uses a 3-state rule: local peaks push toward +1, valleys toward -1, and slopes flow uphill. Convergence typically takes 40–100 ticks (~3 ms on CPU). The result is a unique binary-like pattern per input — similar inputs produce similar patterns when using semantic embeddings.

---

## Learn More

- [Installation Guide](docs/install.md) — venv setup, Windows/macOS, GPU acceleration, Ollama
- [Interactive Demo](ui/demo.html) — see the CA engine in action in your browser (open as a file, no server needed)
- [CLI Reference](docs/cli.md) — every flag documented
- [Architecture](docs/architecture.md) — CA dynamics, temperature system, chunked storage, the math
- [Concepts](docs/concepts.md) — theoretical foundation, reconstructive recall, semantic vs exact search
- [API Reference](docs/api.md) — Python library usage
- [GPU Acceleration](docs/gpu.md) — HIP/ROCm and CUDA setup
- [Web UI](ui/README.md) — dashboard details

---

## Related Tools

- **OpenWebUI Integration** (`open_webui_setup/`) — inject Wheeler memories into any LLM conversation
- **3D Viewer** (`wheeler_3d_viewer/`) — explore attractor landscapes in 3D

---

**The formula is the foundation. Everything else is commentary.**

**Darman doesn't retrieve. Darman reconstructs.**
