"""Eviction / forgetting — graceful degradation of cold memories.

Memories follow a lifecycle through temperature tiers:

    hot (≥0.6) → warm (≥0.3) → cold (≥0.05) → fading (≥0.01) → dead (<0.01)

Lifecycle order:
  1. consolidation — prune redundant frames within the brick (see consolidation.py)
  2. fading — brick (.npz) deleted, attractor + index remain
  3. eviction — all artifacts removed

- Fading: Brick (.npz) is deleted. Attractor and index entry remain —
  the memory can still be recalled but its formation history is lost.
- Dead: Attractor, index entry, association edges, and warmth are removed.

sweep_and_evict() runs all three phases:
  1. fade  — delete bricks below TIER_FADING
  2. evict — fully remove memories below TIER_DEAD
  3. capacity — if over MAX_ATTRACTORS, remove bottom 10% cold memories
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunking import list_existing_chunks
from .hashing import text_to_hex
from .temperature import (
    EVICTION_RATIO,
    MAX_ATTRACTORS,
    MIN_AGE_DAYS,
    TIER_DEAD,
    TIER_FADING,
    TIER_WARM,
    effective_temperature,
    ensure_access_fields,
)
from .warming import load_warmth, remove_memory_from_associations


@dataclass
class EvictionResult:
    bricks_deleted: list[dict] = field(default_factory=list)
    memories_evicted: list[dict] = field(default_factory=list)
    total_before: int = 0
    total_after: int = 0


def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_index(chunk_dir: Path, index: dict) -> None:
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def score_memories(data_dir: str | Path) -> list[dict]:
    """Score all memories by effective temperature, return sorted coldest-first."""
    data_dir = Path(data_dir)
    now = datetime.now(timezone.utc)
    scored = []

    for chunk_name in list_existing_chunks(data_dir):
        chunk_dir = data_dir / "chunks" / chunk_name
        index = _load_index(chunk_dir)
        warmth_data = load_warmth(chunk_dir)

        for hex_key, entry in index.items():
            ensure_access_fields(entry, entry["timestamp"])
            w = warmth_data.get(hex_key, {})
            temp = effective_temperature(
                entry["metadata"]["hit_count"],
                entry["metadata"]["last_accessed"],
                warmth_boost=w.get("boost", 0.0),
                warmth_applied_at=w.get("applied_at"),
                now=now,
            )
            created = datetime.fromisoformat(entry["timestamp"])
            age_days = (now - created).total_seconds() / 86400.0
            scored.append({
                "hex_key": hex_key,
                "chunk": chunk_name,
                "text": entry["text"],
                "temperature": temp,
                "age_days": age_days,
                "hit_count": entry["metadata"]["hit_count"],
            })

    scored.sort(key=lambda m: m["temperature"])
    return scored


def fade_cold_memories(data_dir: str | Path, dry_run: bool = False) -> list[dict]:
    """Phase 1: delete bricks below TIER_FADING (older than MIN_AGE_DAYS)."""
    data_dir = Path(data_dir)
    faded = []

    for m in score_memories(data_dir):
        if m["temperature"] >= TIER_FADING:
            continue
        if m["age_days"] < MIN_AGE_DAYS:
            continue
        brick_path = data_dir / "chunks" / m["chunk"] / "bricks" / f"{m['hex_key']}.npz"
        if brick_path.exists():
            if not dry_run:
                brick_path.unlink()
            faded.append(m)

    return faded


def _delete_memory_files(data_dir: Path, chunk: str, hex_key: str) -> None:
    """Delete all artifacts for a single memory.

    Deletion order: index entry first (prevents half-deleted recall),
    then .npy, then .npz, then association cleanup.
    """
    chunk_dir = data_dir / "chunks" / chunk

    # 1. Remove from index
    index = _load_index(chunk_dir)
    index.pop(hex_key, None)
    _save_index(chunk_dir, index)

    # 2. Remove attractor
    att_path = chunk_dir / "attractors" / f"{hex_key}.npy"
    if att_path.exists():
        att_path.unlink()

    # 3. Remove brick
    brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"
    if brick_path.exists():
        brick_path.unlink()

    # 4. Remove associations and warmth
    remove_memory_from_associations(chunk_dir, hex_key)


def evict_dead_memories(data_dir: str | Path, dry_run: bool = False) -> list[dict]:
    """Phase 2: fully remove memories below TIER_DEAD (older than MIN_AGE_DAYS)."""
    data_dir = Path(data_dir)
    evicted = []

    for m in score_memories(data_dir):
        if m["temperature"] >= TIER_DEAD:
            continue
        if m["age_days"] < MIN_AGE_DAYS:
            continue
        if not dry_run:
            _delete_memory_files(data_dir, m["chunk"], m["hex_key"])
        evicted.append(m)

    return evicted


def evict_for_capacity(data_dir: str | Path, dry_run: bool = False) -> list[dict]:
    """Phase 3: if over MAX_ATTRACTORS, remove bottom EVICTION_RATIO cold memories."""
    data_dir = Path(data_dir)
    scored = score_memories(data_dir)
    total = len(scored)

    if total <= MAX_ATTRACTORS:
        return []

    n_to_evict = max(1, int(total * EVICTION_RATIO))
    evicted = []

    for m in scored:
        if len(evicted) >= n_to_evict:
            break
        # Never evict warm or hot memories
        if m["temperature"] >= TIER_WARM:
            break
        if m["age_days"] < MIN_AGE_DAYS:
            continue
        if not dry_run:
            _delete_memory_files(data_dir, m["chunk"], m["hex_key"])
        evicted.append(m)

    return evicted


def sweep_and_evict(
    data_dir: str | Path,
    dry_run: bool = False,
) -> EvictionResult:
    """Run all 3 eviction phases, return an EvictionResult report."""
    data_dir = Path(data_dir)
    total_before = len(score_memories(data_dir))

    # Phase 1: fade — delete bricks for fading memories
    bricks_deleted = fade_cold_memories(data_dir, dry_run=dry_run)

    # Phase 2: evict — fully remove dead memories
    memories_evicted = evict_dead_memories(data_dir, dry_run=dry_run)

    # Phase 3: capacity — evict bottom 10% if over limit
    capacity_evicted = evict_for_capacity(data_dir, dry_run=dry_run)
    memories_evicted.extend(capacity_evicted)

    total_after = len(score_memories(data_dir)) if not dry_run else total_before

    return EvictionResult(
        bricks_deleted=bricks_deleted,
        memories_evicted=memories_evicted,
        total_before=total_before,
        total_after=total_after,
    )


def forget_memory(hex_key: str, data_dir: str | Path) -> bool:
    """Targeted: delete a specific memory by hex key. Returns True if found."""
    data_dir = Path(data_dir)

    for chunk_name in list_existing_chunks(data_dir):
        chunk_dir = data_dir / "chunks" / chunk_name
        index = _load_index(chunk_dir)
        if hex_key in index:
            _delete_memory_files(data_dir, chunk_name, hex_key)
            return True

    return False


def forget_by_text(text: str, data_dir: str | Path) -> bool:
    """Targeted: delete by original text (computes hex key). Returns True if found."""
    hex_key = text_to_hex(text)
    return forget_memory(hex_key, data_dir)
