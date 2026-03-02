"""Shared pytest fixtures for the Wheeler Memory test suite."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.storage import _load_index, store_memory


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for memory storage, auto-cleaned by pytest."""
    return tmp_path


def store_test_memory(text: str, data_dir: Path, chunk: str = "general") -> str:
    """Store a memory and return its hex key. Shared helper for integration tests."""
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False)


def store_test_memory_embed(text: str, data_dir: Path, chunk: str = "general") -> str:
    """Store a memory using semantic embedding (embed_to_frame) instead of SHA-256.

    Both store and recall must use embedding mode for semantic cross-text recall
    to work — this helper ensures the store side uses the embedding path.
    Requires sentence-transformers to be installed.
    """
    from wheeler_memory.embedding import embed_to_frame
    frame = embed_to_frame(text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False)


def age_memory(data_dir: Path, chunk: str, hex_key: str, days: float) -> None:
    """Backdate a memory's timestamps by N days (simulates aging)."""
    chunk_dir = data_dir / "chunks" / chunk
    index = _load_index(chunk_dir)
    if hex_key not in index:
        return
    past = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    index[hex_key]["timestamp"] = past
    index[hex_key]["metadata"]["last_accessed"] = past
    (chunk_dir / "index.json").write_text(json.dumps(index, indent=2))


def make_synthetic_history(
    n_frames: int, transitions: list[int] | None = None
) -> list[np.ndarray]:
    """Create a synthetic evolution history with known transition points.

    Frames are mostly constant with tiny noise. At transition indices,
    there is a large change to create clear keyframe candidates.
    """
    rng = np.random.RandomState(42)
    base = rng.randn(64, 64).astype(np.float32) * 0.5
    history = []
    for i in range(n_frames):
        if transitions and i in transitions:
            base = rng.randn(64, 64).astype(np.float32) * 0.5
            history.append(base.copy())
        else:
            noise = rng.randn(64, 64).astype(np.float32) * 0.001
            history.append(base + noise)
    return history


def make_uniform_history(n_frames: int) -> list[np.ndarray]:
    """Create a history where all frames are nearly identical."""
    rng = np.random.RandomState(99)
    base = rng.randn(64, 64).astype(np.float32) * 0.5
    return [base + rng.randn(64, 64).astype(np.float32) * 0.0001 for _ in range(n_frames)]
