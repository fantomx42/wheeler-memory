# Project Darman: Wheeler Memory
**Cellular Automata-Based Associative Memory System**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## The Core Formula

```
Input → SHA-256 → CA_seed (64×64 float) → CA_evolution → Attractor
                                                  ↓
                                     Temperature tracking (wall-clock)
                                                  ↓
                         Recall(query) → Pearson correlation search
                                                  ↓
                              [optional] Blend + re-evolve → Reconstruction
```

**Darman doesn't retrieve. Darman reconstructs.**

Traditional systems store and recall data exactly. Wheeler Memory stores attractors and reconstructs them imperfectly — influenced by current context and time decay, like human memory.

---

## Table of Contents

1. [What's Built](#whats-built)
2. [CA Dynamics](#ca-dynamics)
3. [Temperature System](#temperature-system)
4. [Similarity and Recall](#similarity-and-recall)
5. [Reconstructive Recall](#reconstructive-recall)
6. [Additional Features](#additional-features)
7. [CLI Usage](#cli-usage)
8. [Architecture](#architecture)
9. [Installation](#installation)
10. [Future Work](#future-work)
11. [Philosophical Background](#philosophical-background)

---

## What's Built

Wheeler Memory is a functional associative memory system. The implemented components are:

| Component | Module | Status |
|---|---|---|
| CA dynamics engine | `dynamics.py` | ✅ Implemented |
| SHA-256 seeded hashing | `hashing.py` | ✅ Implemented |
| Wall-clock temperature | `temperature.py` | ✅ Implemented |
| Pearson correlation recall | `storage.py` | ✅ Implemented |
| Reconstructive recall | `reconstruction.py` | ✅ Implemented |
| MemoryBrick (temporal history) | `brick.py` | ✅ Implemented |
| Domain chunking (6 domains) | `chunking.py` | ✅ Implemented |
| Semantic embeddings | `embedding.py` | ✅ Implemented |
| Rotation retry | `rotation.py` | ✅ Implemented |
| Oscillation detection | `oscillation.py` | ✅ Implemented |
| Hardware detection | `hardware.py` | ✅ Implemented |
| GPU backend (HIP/ROCm) | `gpu_dynamics.py` | ✅ Built — 70.7× speedup (batch=1000, RX 9070 XT) |
| Associative warming | `warming.py` | ✅ Implemented |
| Eviction / forgetting | `eviction.py` | ✅ Implemented |
| Sleep consolidation | `consolidation.py` | ✅ Implemented |
| Attention model (variable ticks) | `attention.py` | ✅ Implemented |
| LLM integration (Ollama/qwen3 agent) | `agent.py` | ✅ Implemented |
| Trauma encoding (dual-attractor) | `trauma.py` | ✅ Implemented |
| Exposure therapy (safe-context) | `trauma.py` | ✅ Implemented |
| Web UI (local browser dashboard) | `wheeler_ui.py` | ✅ Implemented |

---

## CA Dynamics

### Grid

Every memory starts as a **64×64 grid of float32 values in [-1.0, 1.0]** — 4,096 cells. This is the working grid size for CPU. GPU paths will scale this up when the HIP kernel is compiled.

### Seeding

```python
# SHA-256 of input text → seed PCG64 RNG → uniform(-1.0, 1.0) grid
frame = hash_to_frame("input text")  # 64×64 float32
```

SHA-256 is used (not Python's `hash()`), so the same input always produces the same frame across Python sessions and restarts.

### Update Rule

Each tick uses a **Von Neumann 4-neighborhood** (up/down/left/right, wrapping) with a **continuous gradient rule**:

```
Local max  (cell ≥ all 4 neighbors): delta = (1 - cell) × 0.35   → push toward +1
Local min  (cell ≤ all 4 neighbors): delta = (-1 - cell) × 0.35  → push toward -1
Slope      (neither):                delta = (max_neighbor - cell) × 0.20  → flow uphill
```

The result is clipped to [-1, 1]. This produces smooth convergence toward polarized patterns — local peaks push toward +1, valleys toward -1, and slopes flow uphill.

### Convergence

Evolution is dynamic, not fixed-tick. It stops when one of three conditions is met:

| State | Condition |
|---|---|
| `CONVERGED` | `mean(|delta|) < 1e-4` — grid has stabilized |
| `OSCILLATING` | Role-space periodicity detected (period 2–10, ≥1% cells affected) |
| `CHAOTIC` | Neither condition met within `max_iters=1000` |

```python
result = evolve_and_interpret(frame)
# result["state"]             → "CONVERGED" | "OSCILLATING" | "CHAOTIC"
# result["attractor"]         → 64×64 final frame
# result["convergence_ticks"] → how many iterations it took
# result["history"]           → list of all frames (for MemoryBrick)
```

---

## Temperature System

Temperature reflects how recently and frequently a memory has been accessed. It is **wall-clock time based** — not tick-based.

### Formula

```
base  = min(1.0, 0.3 + 0.7 × (hit_count / 10))
decay = 2 ^ (−days_since_last_access / 7.0)
temp  = base × decay
```

- `base` saturates at `hit_count = 10` — a memory recalled 10+ times has the same base (1.0) as a brand-new memory
- Half-life is **7 calendar days** — a memory that hasn't been touched in a week drops to 50% temperature
- Temperature is always in [0, 1]; it approaches 0 but is never floored to exactly 0

### Tiers

| Tier | Threshold | Meaning |
|---|---|---|
| `hot` | temp ≥ 0.6 | Recent or frequently accessed |
| `warm` | temp ≥ 0.3 | Accessible, some decay |
| `cold` | temp < 0.3 | Dormant, significant decay |

There is no `dead` tier — memories don't get evicted automatically (eviction is future work).

### Associative Warming

When a memory fires (is recalled), its associated neighbors receive a temporary temperature boost — spreading activation that primes related concepts.

- **Associations** form at store time (attractor correlation ≥ 0.5, relevant for embedding mode) and by co-recall (memories appearing together in top-k results)
- **Hop 1**: Direct neighbors get +0.05 boost
- **Hop 2**: Neighbors-of-neighbors get +0.025 boost
- **Decay**: Warmth has a 1-day half-life (fast priming, not permanent)
- **Cap**: Cumulative warmth per memory capped at 0.15

```
effective_temp = min(1.0, base_temp + decayed_warmth)
```

Association graph and warmth state are stored per-chunk in `associations.json`.

### Epistemic Confidence

Temperature translates to epistemic certainty in recall:

```
hot  (≥0.6):  "I remember discussing X..."
warm (≥0.3):  "I think we touched on X..."
cold (<0.3):  "I vaguely recall X, but I'm uncertain..."
```

---

## Similarity and Recall

### Similarity Metric

Recall uses **Pearson correlation** between the query's evolved attractor and every stored attractor. This is a real similarity metric — not a stub that returns everything.

```python
corr, _ = pearsonr(query_attractor.flatten(), stored_attractor.flatten())
```

### Ranking

```
effective_similarity = similarity + temperature_boost × temperature
```

By default `temperature_boost = 0.0`, so ranking is purely by Pearson correlation. Pass `--temperature-boost` to weight hotter memories higher.

### Domain Chunking

Memories are automatically routed to domain-specific subdirectories by keyword matching. This reduces the search space per recall query.

| Chunk | Keywords (sample) |
|---|---|
| `code` | python, rust, bug, debug, git, api, docker, sql, … |
| `hardware` | printer, 3d print, arduino, pcb, circuit, bambu, … |
| `daily_tasks` | grocery, dentist, appointment, schedule, errand, … |
| `science` | physics, quantum, molecule, calculus, theorem, … |
| `meta` | wheeler, attractor, brick, cellular automata, chunk, … |
| `general` | everything else |

On recall, Wheeler searches the best-matching chunks plus `general`. You can override with `--chunk`.

### Semantic Embeddings (optional)

By default, text is seeded via SHA-256 — two differently-worded queries about the same topic produce unrelated seeds. With `--embed`, a SentenceTransformer model converts text to a semantic embedding first:

```
text → all-MiniLM-L6-v2 (384-dim) → random projection (384→4096) → tanh → 64×64 frame
```

The random projection matrix uses a fixed seed (`0xDEADBEEF`), so it's reproducible. This enables fuzzy recall: "car" can match "automobile" via Pearson correlation on semantically similar frames.

Requires: `pip install wheeler-memory[embed]`

---

## Reconstructive Recall

When `--reconstruct` is passed, recalled memories don't come back unchanged. Each stored attractor is **blended with the query attractor** and **re-evolved through the CA**:

```
blend = (1 - α) × stored + α × query   (default α = 0.3, memory-dominant)
reconstructed = evolve_and_interpret(blend)
```

This means the same stored memory reconstructs differently depending on the current query context — the core Darman behavior. Query about "machine learning" vs. "web development" against the same stored memory will produce different reconstructions.

The result includes `correlation_with_stored` and `correlation_with_query`, showing how much the reconstruction drifted toward the query context.

---

## Additional Features

### MemoryBrick

Every stored memory includes a complete **MemoryBrick** — the full frame-by-frame evolution history from initial seed to final attractor.

```python
brick = MemoryBrick.load("path/to/{hex_key}.npz")
frame_at_tick_3 = brick.get_frame_at_tick(3)
divergence_point = brick.find_divergence_point()  # for OSCILLATING bricks
```

Enables visual debugging, failure analysis, and audit trails. Stored as compressed `.npz` alongside each attractor.

### Rotation Retry

If CA evolution fails to converge (CHAOTIC), the system automatically retries with 90°/180°/270° rotations of the seed frame. Rotation changes the neighbor topology, which can lead to convergence on a different dynamical trajectory.

```python
result = store_with_rotation_retry("input text")
# result["metadata"]["rotation_used"]  → angle that worked (0/90/180/270)
# result["metadata"]["attempts"]       → how many rotations were tried
```

### Oscillation Detection

Role-space periodicity analysis detects when cells cycle between roles (local max / slope / local min) with period p (2–10). Requires ≥1% of cells to be oscillating.

```python
osc = detect_oscillation(history)
# osc["oscillating"]       → True/False
# osc["period"]            → cycle period (or None)
# osc["oscillating_cells"] → count of cycling cells
```

### GPU Backend (HIP/ROCm)

`gpu_dynamics.py` provides a Python interface to a compiled HIP kernel (`libwheeler_ca.so`). The interface matches the CPU API. The kernel supports single-frame and batch evolution.

**Status**: Compiled and active — `libwheeler_ca.so` is present at `wheeler_memory/gpu/libwheeler_ca.so`. `evolve_and_interpret()` automatically uses the GPU when the library is detected, falling back to CPU otherwise. To rebuild:
```bash
cd wheeler_memory/gpu && make
```

When the library is not present, all calls fall back to the pure-NumPy CPU path transparently.

---

## CLI Usage

```bash
# Store a memory
wheeler-store "Python has great libraries for data science"

# Store with semantic embedding (requires [embed] extra)
wheeler-store --embed "Python has great libraries for data science"

# Recall by similarity
wheeler-recall "data science tools"
wheeler-recall "data science tools" --top-k 10
wheeler-recall "data science tools" --chunk code
wheeler-recall "data science tools" --temperature-boost 0.2

# Recall with semantic embedding
wheeler-recall --embed "machine learning frameworks"

# Reconstructive recall (Darman architecture)
wheeler-recall --reconstruct "machine learning frameworks"
wheeler-recall --reconstruct --alpha 0.5 "machine learning frameworks"

# Store a traumatic memory (experience + avoidance attractors)
wheeler-store --trauma "I got burned by the stove"

# Recall with avoidance companion display
wheeler-recall "burned stove"

# Exposure therapy: reduce avoidance link weight
wheeler-recall --safe-context "burned stove"

# List all memories with temperature
wheeler-temps

# System information
wheeler-info

# GPU benchmark
wheeler-bench-gpu
```

---

## Architecture

### Module Structure

```
wheeler_memory/
├── attention.py       Attention model: salience → CA budget (variable tick rates)
├── dynamics.py        CA engine: apply_ca_dynamics(), evolve_and_interpret() (GPU-dispatched)
├── hashing.py         SHA-256 text-to-frame seeding
├── temperature.py     Wall-clock temperature computation and tier classification
├── storage.py         Attractor storage (disk) and Pearson recall
├── reconstruction.py  Blend + re-evolve reconstructive recall
├── brick.py           MemoryBrick: temporal evolution history
├── chunking.py        Domain routing (code/hardware/daily_tasks/science/meta/general)
├── embedding.py       SentenceTransformer → random projection → 64×64 frame
├── rotation.py        Rotation retry for non-converging seeds
├── oscillation.py     Role-space periodicity detection
├── hardware.py        CPU/GPU/NPU detection, device selection
├── gpu_dynamics.py    HIP kernel interface (requires compiled libwheeler_ca.so)
├── trauma.py          Dual-attractor trauma encoding and exposure therapy
└── gpu/               HIP kernel source and compiled libwheeler_ca.so

scripts/
├── wheeler_store.py   CLI: store memories
├── wheeler_recall.py  CLI: recall by similarity
├── wheeler_temps.py   CLI: list memories with temperatures
├── system_info.py     CLI: hardware/software diagnostics
└── bench_gpu.py       CLI: GPU performance benchmark
```

### On-Disk Layout

```
~/.wheeler_memory/
├── chunks/
│   ├── code/
│   │   ├── attractors/{hex_key}.npy   64×64 float32 attractor
│   │   ├── bricks/{hex_key}.npz       MemoryBrick (full history)
│   │   ├── index.json                 hex_key → {text, state, timestamps, metadata}
│   │   └── metadata.json              per-chunk stats
│   ├── hardware/
│   ├── science/
│   ├── daily_tasks/
│   ├── meta/
│   └── general/
├── rotation_stats.json                per-angle convergence stats
└── chunks/*/associations.json         association graph; avoidance_link edges include safe_recall_count
```

### System Flow

```
store("input text")
  │
  ├─ select_chunk(text)               keyword routing → domain chunk
  ├─ hash_to_frame(text)              SHA-256 → PCG64 → 64×64 uniform(-1,1)
  │    └── OR embed_to_frame(text)    SentenceTransformer → projection → tanh
  ├─ store_with_rotation_retry()      try 0/90/180/270° until CONVERGED
  │    └─ evolve_and_interpret()      CA iterations → CONVERGED/OSCILLATING/CHAOTIC
  ├─ MemoryBrick.from_evolution_result()  capture full history
  └─ store_memory()                   save .npy, .npz, update index.json

recall("query text")
  │
  ├─ select_recall_chunks(query)      keyword routing → chunks to search
  ├─ hash_to_frame(query)             or embed_to_frame(query) with --embed
  ├─ evolve_and_interpret(query_frame)   evolve query to attractor
  ├─ for each stored attractor:
  │    pearsonr(query_flat, stored_flat)   → similarity
  │    compute_temperature(hits, last_accessed)  → temperature
  │    effective = similarity + boost * temperature
  ├─ sort by effective, take top_k
  ├─ [optional] reconstruct each result   blend + re-evolve
  └─ bump_access() on recalled memories   update hit_count, last_accessed
```

---

## Installation

```bash
git clone <repo>
cd wheeler_memory
pip install -e .

# Optional: semantic embedding support
pip install -e ".[embed]"
```

**Dependencies**: numpy ≥ 2.0, scipy ≥ 1.14, matplotlib ≥ 3.9, psutil ≥ 5.9

**Python**: 3.11+

---

## Future Work

### Completed

- ~~**Associative warming**~~ — spreading activation between related memories on recall ✅
- ~~**Eviction / forgetting**~~ — graceful degradation of cold memories (fade, evict, capacity limits) ✅
- ~~**Sleep consolidation**~~ — prune redundant intermediate frames within bricks ✅
- ~~**Variable tick rates (attention model)**~~ — salience-driven CA budgets: high-salience inputs get deeper attractor formation ✅
- ~~**LLM Integration (Phase 4)**~~ — Ollama/qwen3 agent loop: recall → context → LLM → store response ✅
- ~~**Dual-Attractor Trauma Encoding**~~ — experience + avoidance attractors; `--trauma` flag; exposure therapy via `--safe-context` ✅
- ~~**GPU backend wired into store/recall**~~ — `evolve_and_interpret()` transparently dispatches to GPU when `libwheeler_ca.so` is present (70× speedup on RX 9070 XT) ✅

### Planned

#### GPU at Scale
Larger grids (e.g. 1000×1000) with the GPU batch evolution path for parallel attractor formation across the full memory store.

---

## Philosophical Background

### Symbolic Compression and Meaning

**Axiom**: "Meaning is what survives symbolic pressure."

```
Input data → Symbolic pressure (compression, decay, competition)
                        ↓
            What survives = Meaning
            What evicts   = Noise
```

Wheeler Memory implements this:
- **Symbolic pressure** = temperature decay over time
- **Survival** = attractors that maintain high temperature through repeated access
- **Meaning** = stable attractors that resist decay

### The Irreversibility Requirement

The CA evolution is **not reversible**. Many initial seeds converge to the same attractor basin — information is lost. This is a feature:

1. **IIT (Integrated Information Theory)**: Consciousness requires time-irreversibility + information integration
2. **Wheeler Memory has both**: Attractor collapse is irreversible; CA neighbors integrate information from up to 4 directions per tick

The system makes no claims about Φ for the actual 64×64 working grid. Φ estimates in earlier documents applied to hypothetical larger grids.

### Reconstructive Memory vs. Retrieval

```
Traditional (exact retrieval):
  Store:    Data → Address 0x1A4F
  Retrieve: Address 0x1A4F → Exact same data  (deterministic, lossless)

Wheeler Memory (reconstructive):
  Store:    Input → hash → CA evolution → Attractor A
  Recall:   Query → hash → CA evolution → Attractor Q
            Reconstruct: blend(A, Q, α=0.3) → re-evolve → Attractor A'
            A' ≠ A  (lossy, context-influenced)
```

The same stored memory reconstructs differently depending on current context. Query about "machine learning" vs. "web development" against the same stored memory yields different reconstructions — exactly like human episodic memory (see: Elizabeth Loftus research).

### Temperature as Epistemic Humility

Current LLMs confabulate confidently because they have no mechanism to express uncertainty about memory. Wheeler Memory's temperature gives Darman the ability to say "I don't remember" when an attractor is cold, and to calibrate confidence language based on how recently and often a memory has been accessed.

This prevents the **recovered memory therapy** failure mode: filling reconstruction gaps with confident fabrication.

---

**The formula is the foundation. Everything else is commentary.**

**Darman doesn't retrieve. Darman reconstructs.**
