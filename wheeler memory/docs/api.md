# Python API Reference

## Quick imports

```python
from wheeler_memory import (
    store_with_rotation_retry,
    recall_memory,
    list_memories,
)
from wheeler_memory.reconstruction import reconstruct
from wheeler_memory.embedding import embed_to_frame
from wheeler_memory.temperature import compute_temperature
from wheeler_memory.hardware import get_system_summary
```

---

## `store_with_rotation_retry`

```python
def store_with_rotation_retry(
    text: str,
    max_rotations: int = 4,
    save: bool = True,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    use_embedding: bool = False,
) -> dict:
```

Encode `text` as a 64×64 CA seed frame, evolve it to an attractor, and store
the result. If the initial evolution does not converge, the seed frame is
rotated by 90°, 180°, and 270° successively until convergence or all rotations
are exhausted.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `text` | — | Text to store |
| `max_rotations` | `4` | How many rotation angles to try (1–4) |
| `save` | `True` | Write attractor + brick to disk |
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `chunk` | `None` | Force a specific chunk; auto-routes if `None` |
| `use_embedding` | `False` | Use sentence embedding instead of SHA-256 |

**Returns** a `dict` with:

| Key | Type | Description |
|---|---|---|
| `state` | `str` | `CONVERGED`, `OSCILLATING`, `CHAOTIC`, or `FAILED_ALL_ROTATIONS` |
| `attractor` | `np.ndarray` | Final 64×64 attractor frame |
| `convergence_ticks` | `int` | Ticks until convergence |
| `history` | `list[np.ndarray]` | All frames from seed to attractor |
| `metadata` | `dict` | Includes `rotation_used`, `attempts`, `wall_time_seconds` |

**Example**

```python
result = store_with_rotation_retry("fix the auth bug in login.py")
print(result["state"])          # CONVERGED
print(result["metadata"])       # {'rotation_used': 0, 'attempts': 1, ...}

# Semantic store
result = store_with_rotation_retry(
    "the sky is blue",
    use_embedding=True,
    chunk="science",
)
```

---

## `recall_memory`

```python
def recall_memory(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    temperature_boost: float = 0.0,
    use_embedding: bool = False,
    reconstruct: bool = False,
    reconstruct_alpha: float = 0.3,
) -> list[dict]:
```

Evolve the query `text` to an attractor, then search stored attractors by
Pearson correlation. Returns the `top_k` best matches, sorted by
`effective_similarity`.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `text` | — | Query text |
| `top_k` | `5` | Maximum results to return |
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `chunk` | `None` | Search only this chunk; searches all matching chunks if `None` |
| `temperature_boost` | `0.0` | Adds `boost × temperature` to ranking score |
| `use_embedding` | `False` | Use sentence embedding for the query frame |
| `reconstruct` | `False` | Apply Darman reconstruction to each result |
| `reconstruct_alpha` | `0.3` | Blend factor for reconstruction (0 = pure stored, 1 = pure query) |

**Returns** a `list[dict]`, each entry containing:

| Key | Type | Description |
|---|---|---|
| `text` | `str` | Original stored text |
| `similarity` | `float` | Pearson correlation with query attractor |
| `temperature` | `float` | Current temperature `[0, 1]` |
| `temperature_tier` | `str` | `hot`, `warm`, or `cold` |
| `effective_similarity` | `float` | `similarity + temperature_boost × temperature` |
| `state` | `str` | Convergence state when the memory was stored |
| `chunk` | `str` | Which chunk this memory lives in |
| `timestamp` | `str` | ISO-8601 timestamp of when it was stored |
| `hex_key` | `str` | SHA-256 hex key (file identifier) |

When `reconstruct=True`, each result also includes:

| Key | Type | Description |
|---|---|---|
| `reconstructed_attractor` | `np.ndarray` | The context-blended attractor |
| `reconstruction_state` | `str` | Convergence state of the reconstruction |
| `reconstruction_ticks` | `int` | Ticks for reconstruction to converge |
| `reconstruction_alpha` | `float` | The alpha used |
| `correlation_with_stored` | `float` | Pearson between reconstruction and stored attractor |
| `correlation_with_query` | `float` | Pearson between reconstruction and query attractor |

Every recalled memory has its `hit_count` incremented and `last_accessed`
updated automatically.

**Example**

```python
# Basic recall
matches = recall_memory("authentication error", top_k=3)
for m in matches:
    print(f"[{m['temperature_tier']}] {m['text']}  sim={m['similarity']:.3f}")

# Fuzzy semantic recall with temperature ranking bonus
matches = recall_memory(
    "what was I debugging yesterday",
    use_embedding=True,
    temperature_boost=0.2,
)

# Reconstructive recall (Darman architecture)
matches = recall_memory(
    "machine learning tools",
    reconstruct=True,
    reconstruct_alpha=0.3,
)
for m in matches:
    print(m["text"])
    print(f"  correlation with stored: {m['correlation_with_stored']:.3f}")
    print(f"  correlation with query:  {m['correlation_with_query']:.3f}")
```

---

## `list_memories`

```python
def list_memories(
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> list[dict]:
```

Return all stored memories from the index without running any CA evolution.
Temperatures are computed lazily at list time.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `chunk` | `None` | List only this chunk; lists all chunks if `None` |

**Returns** a `list[dict]` with the same fields as `recall_memory` results
(minus similarity / reconstruction fields).

**Example**

```python
memories = list_memories()
for m in sorted(memories, key=lambda x: x["temperature"], reverse=True):
    print(f"[{m['temperature_tier']:4}] {m['temperature']:.2f}  {m['text']}")

# List just the code chunk
code_mems = list_memories(chunk="code")
```

---

## `reconstruct`

```python
# wheeler_memory.reconstruction
def reconstruct(
    stored_attractor: np.ndarray,
    query_attractor: np.ndarray,
    alpha: float = 0.3,
) -> dict:
```

The **Darman architecture**: blend a stored memory with the current query
context and re-evolve through the CA. The same stored memory reconstructs
differently depending on what you're thinking about.

```
blended = (1 − α) × stored + α × query
result  = CA_evolve(blended)
```

`alpha=0` returns the stored memory unchanged (after re-evolution).
`alpha=1` ignores the stored memory entirely.
`alpha=0.3` (default) is memory-dominant but context-aware.

**Returns** a `dict`:

| Key | Type | Description |
|---|---|---|
| `attractor` | `np.ndarray` | Reconstructed 64×64 attractor |
| `state` | `str` | Convergence state of reconstruction |
| `convergence_ticks` | `int` | Ticks to re-converge |
| `alpha` | `float` | The alpha used |
| `correlation_with_stored` | `float` | Pearson(reconstructed, stored) |
| `correlation_with_query` | `float` | Pearson(reconstructed, query) |

**Example**

```python
import numpy as np
from wheeler_memory import recall_memory
from wheeler_memory.reconstruction import reconstruct
from wheeler_memory.storage import DEFAULT_DATA_DIR

# Retrieve stored and query attractors manually
results = recall_memory("Python libraries", top_k=1)
hex_key = results[0]["hex_key"]
chunk   = results[0]["chunk"]

stored_att = np.load(DEFAULT_DATA_DIR / "chunks" / chunk / "attractors" / f"{hex_key}.npy")

from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.dynamics import evolve_and_interpret

query_att = evolve_and_interpret(hash_to_frame("machine learning"))["attractor"]

recon = reconstruct(stored_att, query_att, alpha=0.3)
print(f"state: {recon['state']}")
print(f"correlation with stored: {recon['correlation_with_stored']:.3f}")
print(f"correlation with query:  {recon['correlation_with_query']:.3f}")
```

---

## `embed_to_frame`

```python
# wheeler_memory.embedding  (requires pip install -e ".[embed]")
def embed_to_frame(text: str, size: int = 64) -> np.ndarray:
```

Convert `text` to a 64×64 CA frame via sentence embedding and random
projection.

1. Encode text → 384-dim vector (`all-MiniLM-L6-v2`)
2. Project 384 → 4096 via a fixed Gaussian random matrix (seed `0xDEADBEEF`)
3. Apply `tanh(x × 3)` to map to `(−1, 1)`
4. Reshape to `(64, 64)`

Similar text produces similar frames, enabling fuzzy recall through Pearson
correlation search. The model is lazy-loaded and cached after the first call.

**Example**

```python
from wheeler_memory.embedding import embed_to_frame, embed_available

if embed_available():
    frame = embed_to_frame("The Eiffel Tower is in Paris")
    print(frame.shape)   # (64, 64)
    print(frame.dtype)   # float32
```

---

## `compute_temperature`

```python
# wheeler_memory.temperature
def compute_temperature(
    hit_count: int,
    last_accessed: str | datetime,
    now: datetime | None = None,
) -> float:
```

Compute the temperature scalar `[0, 1]` for a memory from its access history.

```
base_from_hits  = min(1.0,  0.3 + 0.7 × (hit_count / 10))
decay_from_time = 2 ^ (−days_since_last_access / 7)
temp            = base_from_hits × decay_from_time
```

`last_accessed` accepts an ISO-8601 string or a `datetime` object (timezone-aware).
`now` defaults to `datetime.now(timezone.utc)`.

**Example**

```python
from wheeler_memory.temperature import compute_temperature, temperature_tier
from datetime import datetime, timezone, timedelta

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

temp = compute_temperature(hit_count=5, last_accessed=yesterday)
print(f"temp={temp:.4f}  tier={temperature_tier(temp)}")
# temp=0.5743  tier=warm
```

---

## `get_system_summary`

```python
# wheeler_memory.hardware
def get_system_summary() -> dict:
```

Aggregate hardware information useful for debugging environment issues and
understanding which accelerator will be used.

**Returns** a `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `os` | `str` | OS name (`Linux`, `Windows`, …) |
| `release` | `str` | Kernel/OS release string |
| `cpu` | `dict` | `architecture`, `processor`, `physical_cores`, `total_cores`, `frequency_mhz` |
| `memory` | `dict` | `total_gb`, `available_gb`, `used_gb`, `percent_used` |
| `storage` | `dict` | `total_gb`, `used_gb`, `free_gb`, `percent_used` for `/` |
| `gpu_npu` | `dict` | `nvidia_gpu` list or string, `pci_devices` list from `lspci` |
| `optimal_device` | `str` | `"cuda"`, `"mps"`, or `"cpu"` |
| `warnings` | `list[str]` | Mismatch warnings (e.g. GPU found but PyTorch using CPU) |

**Example**

```python
from wheeler_memory.hardware import get_system_summary

info = get_system_summary()
print(info["optimal_device"])       # cpu / cuda / mps
for w in info["warnings"]:
    print("WARNING:", w)
```

---

## Store → Recall → Reconstruct workflow

```python
from wheeler_memory import store_with_rotation_retry, recall_memory

# 1. Store memories
store_with_rotation_retry("Python has great libraries for data science")
store_with_rotation_retry("scikit-learn is perfect for classical ML")
store_with_rotation_retry("PyTorch is used for deep learning")

# 2. Recall with semantic embedding
results = recall_memory(
    "machine learning tools",
    top_k=3,
    use_embedding=True,        # fuzzy semantic match
    reconstruct=True,          # Darman reconstruction
    reconstruct_alpha=0.3,     # 70% stored, 30% query context
    temperature_boost=0.1,     # favour recently accessed memories
)

# 3. Inspect results
for r in results:
    print(f"[{r['temperature_tier']:4}] sim={r['similarity']:.3f}  {r['text']}")
    if "correlation_with_stored" in r:
        print(f"         stored_corr={r['correlation_with_stored']:.3f}"
              f"  query_corr={r['correlation_with_query']:.3f}")
```

---

## Eviction / Forgetting

```python
from wheeler_memory import (
    sweep_and_evict,
    forget_memory,
    forget_by_text,
    score_memories,
    EvictionResult,
    TIER_FADING,
    TIER_DEAD,
    MAX_ATTRACTORS,
)
```

### `sweep_and_evict`

```python
def sweep_and_evict(
    data_dir: str | Path,
    dry_run: bool = False,
) -> EvictionResult:
```

Run all three eviction phases and return a report.

1. **Fade** — delete `.npz` bricks for memories below `TIER_FADING` (0.05)
2. **Evict** — fully remove memories below `TIER_DEAD` (0.01)
3. **Capacity** — if over `MAX_ATTRACTORS` (10,000), evict bottom 10% cold memories

Memories younger than 1 day are never affected.

**Returns** an `EvictionResult`:

| Field | Type | Description |
|---|---|---|
| `bricks_deleted` | `list[dict]` | Memories whose bricks were faded |
| `memories_evicted` | `list[dict]` | Memories fully removed |
| `total_before` | `int` | Memory count before sweep |
| `total_after` | `int` | Memory count after sweep |

**Example**

```python
result = sweep_and_evict("~/.wheeler_memory")
print(f"Faded {len(result.bricks_deleted)} bricks")
print(f"Evicted {len(result.memories_evicted)} memories")
print(f"Total: {result.total_before} → {result.total_after}")

# Dry run — inspect without deleting
result = sweep_and_evict("~/.wheeler_memory", dry_run=True)
```

---

### `forget_memory` / `forget_by_text`

```python
def forget_memory(hex_key: str, data_dir: str | Path) -> bool:
def forget_by_text(text: str, data_dir: str | Path) -> bool:
```

Immediately delete a specific memory. Returns `True` if found.

```python
forget_by_text("fix the python debug error", "~/.wheeler_memory")
forget_memory("a1b2c3d4...", "~/.wheeler_memory")
```

---

### `score_memories`

```python
def score_memories(data_dir: str | Path) -> list[dict]:
```

Score all memories by effective temperature, sorted coldest-first.
Each entry contains `hex_key`, `chunk`, `text`, `temperature`, `age_days`, `hit_count`.
