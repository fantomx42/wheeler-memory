"""Sleep consolidation — prune redundant frames within MemoryBricks.

Memories follow a lifecycle through temperature tiers:

    hot (>=0.6) -> warm (>=0.3) -> cold (<0.3)

    consolidation -> fading -> eviction

Consolidation compresses the evolution history stored in each brick by
keeping only salient keyframes — frames where either the mean absolute
delta or the fraction of cells changing roles exceeds a threshold.
Frame 0 (seed) and the final frame (attractor) are always kept.

Hot bricks are skipped (actively used). Warm bricks get light pruning.
Cold bricks get aggressive pruning. Already-consolidated bricks are
skipped (idempotent).
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .brick import MemoryBrick
from .chunking import list_existing_chunks
from .oscillation import get_cell_roles
from .temperature import (
    CONSOLIDATION_DELTA_COLD,
    CONSOLIDATION_DELTA_WARM,
    CONSOLIDATION_MIN_FRAMES,
    CONSOLIDATION_MIN_HISTORY,
    CONSOLIDATION_ROLE_COLD,
    CONSOLIDATION_ROLE_WARM,
    TIER_HOT,
    TIER_WARM,
    effective_temperature,
    ensure_access_fields,
    temperature_tier,
)
from .warming import load_warmth


@dataclass
class ConsolidationResult:
    memories_consolidated: list[dict] = field(default_factory=list)
    memories_skipped: list[dict] = field(default_factory=list)
    total_frames_before: int = 0
    total_frames_after: int = 0


def select_keyframes(
    history: list[np.ndarray],
    delta_threshold: float,
    role_threshold: float,
) -> list[int]:
    """Return sorted list of frame indices to keep.

    A frame is kept if (compared against the last kept frame):
    1. Mean absolute delta >= delta_threshold, OR
    2. Fraction of cells that changed roles >= role_threshold

    Frame 0 (seed) and the final frame (attractor) are always kept.
    At least CONSOLIDATION_MIN_FRAMES are always returned.
    """
    n = len(history)
    if n <= CONSOLIDATION_MIN_FRAMES:
        return list(range(n))

    kept = [0]  # Always keep seed
    last_kept_idx = 0
    total_cells = history[0].size

    last_kept_roles = get_cell_roles(history[0])

    for i in range(1, n - 1):
        # Delta check against last kept frame
        delta = float(np.mean(np.abs(history[i] - history[last_kept_idx])))
        if delta >= delta_threshold:
            kept.append(i)
            last_kept_idx = i
            last_kept_roles = get_cell_roles(history[i])
            continue

        # Role-change check against last kept frame
        current_roles = get_cell_roles(history[i])
        changed = int(np.sum(current_roles != last_kept_roles))
        frac = changed / total_cells
        if frac >= role_threshold:
            kept.append(i)
            last_kept_idx = i
            last_kept_roles = current_roles

    # Always keep final frame (attractor)
    final = n - 1
    if final not in kept:
        kept.append(final)

    # Enforce minimum
    if len(kept) < CONSOLIDATION_MIN_FRAMES and n >= CONSOLIDATION_MIN_FRAMES:
        # Add evenly spaced frames to reach minimum
        all_indices = set(kept)
        candidates = [i for i in range(n) if i not in all_indices]
        step = max(1, len(candidates) // (CONSOLIDATION_MIN_FRAMES - len(kept)))
        for c in candidates[::step]:
            all_indices.add(c)
            if len(all_indices) >= CONSOLIDATION_MIN_FRAMES:
                break
        kept = sorted(all_indices)

    return sorted(kept)


def consolidate_brick(
    brick: MemoryBrick,
    delta_threshold: float,
    role_threshold: float,
) -> MemoryBrick:
    """Return a new MemoryBrick with pruned history + consolidation metadata.

    Already-consolidated bricks are returned unchanged (idempotent).
    Bricks with fewer than CONSOLIDATION_MIN_HISTORY frames are returned unchanged.
    """
    # Skip if already consolidated
    if brick.metadata.get("consolidated"):
        return brick

    n = len(brick.evolution_history)

    # Skip small histories
    if n < CONSOLIDATION_MIN_HISTORY:
        return brick

    kept_indices = select_keyframes(
        brick.evolution_history, delta_threshold, role_threshold,
    )

    # No reduction possible
    if len(kept_indices) >= n:
        return brick

    pruned_history = [brick.evolution_history[i] for i in kept_indices]

    new_metadata = dict(brick.metadata)
    new_metadata["consolidated"] = True
    new_metadata["consolidated_at"] = datetime.now(timezone.utc).isoformat()
    new_metadata["original_frame_count"] = n
    new_metadata["retained_frame_count"] = len(kept_indices)
    new_metadata["frames_pruned"] = n - len(kept_indices)
    new_metadata["consolidation_delta_threshold"] = delta_threshold
    new_metadata["consolidation_role_threshold"] = role_threshold

    return MemoryBrick(
        evolution_history=pruned_history,
        final_attractor=brick.final_attractor,
        convergence_ticks=brick.convergence_ticks,
        state=brick.state,
        metadata=new_metadata,
    )


def thresholds_for_temperature(temperature: float) -> tuple[float, float] | None:
    """Return (delta, role) thresholds for a temperature, or None for hot.

    Hot (>=0.6): skip consolidation entirely.
    Warm (>=0.3): light pruning.
    Cold (<0.3): aggressive pruning.
    """
    if temperature >= TIER_HOT:
        return None
    if temperature >= TIER_WARM:
        return (CONSOLIDATION_DELTA_WARM, CONSOLIDATION_ROLE_WARM)
    return (CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)


def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def consolidate_memory(
    hex_key: str,
    chunk_dir: Path,
    temperature: float,
) -> dict | None:
    """Load brick, consolidate, re-save. Returns info dict or None if skipped."""
    thresholds = thresholds_for_temperature(temperature)
    if thresholds is None:
        return None

    delta_thresh, role_thresh = thresholds

    brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"
    if not brick_path.exists():
        return None

    brick = MemoryBrick.load(brick_path)

    # Skip already consolidated
    if brick.metadata.get("consolidated"):
        return None

    frames_before = len(brick.evolution_history)

    # Skip small histories
    if frames_before < CONSOLIDATION_MIN_HISTORY:
        return None

    consolidated = consolidate_brick(brick, delta_thresh, role_thresh)

    # No reduction
    if len(consolidated.evolution_history) >= frames_before:
        return None

    consolidated.save(brick_path)

    return {
        "hex_key": hex_key,
        "frames_before": frames_before,
        "frames_after": len(consolidated.evolution_history),
        "tier": temperature_tier(temperature),
    }


def sleep_consolidate(
    data_dir: str | Path,
    dry_run: bool = False,
    chunk: str | None = None,
) -> ConsolidationResult:
    """Sweep all chunks (or a specific one), consolidate eligible bricks.

    Returns a ConsolidationResult with details of what was consolidated.
    """
    data_dir = Path(data_dir)
    now = datetime.now(timezone.utc)
    result = ConsolidationResult()

    if chunk is not None:
        chunks = [chunk]
    else:
        chunks = list(list_existing_chunks(data_dir))

    for chunk_name in chunks:
        chunk_dir = data_dir / "chunks" / chunk_name
        if not chunk_dir.exists():
            continue

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

            text = entry.get("text", hex_key[:16])
            tier = temperature_tier(temp)

            # Hot: skip
            thresholds = thresholds_for_temperature(temp)
            if thresholds is None:
                result.memories_skipped.append({
                    "hex_key": hex_key,
                    "chunk": chunk_name,
                    "text": text,
                    "reason": "hot",
                })
                continue

            delta_thresh, role_thresh = thresholds
            brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"

            if not brick_path.exists():
                result.memories_skipped.append({
                    "hex_key": hex_key,
                    "chunk": chunk_name,
                    "text": text,
                    "reason": "no_brick",
                })
                continue

            brick = MemoryBrick.load(brick_path)

            if brick.metadata.get("consolidated"):
                result.memories_skipped.append({
                    "hex_key": hex_key,
                    "chunk": chunk_name,
                    "text": text,
                    "reason": "already_consolidated",
                })
                continue

            frames_before = len(brick.evolution_history)

            if frames_before < CONSOLIDATION_MIN_HISTORY:
                result.memories_skipped.append({
                    "hex_key": hex_key,
                    "chunk": chunk_name,
                    "text": text,
                    "reason": "too_few_frames",
                })
                continue

            consolidated = consolidate_brick(brick, delta_thresh, role_thresh)
            frames_after = len(consolidated.evolution_history)

            if frames_after >= frames_before:
                result.memories_skipped.append({
                    "hex_key": hex_key,
                    "chunk": chunk_name,
                    "text": text,
                    "reason": "no_reduction",
                })
                continue

            result.total_frames_before += frames_before
            result.total_frames_after += frames_after

            info = {
                "hex_key": hex_key,
                "chunk": chunk_name,
                "text": text,
                "frames_before": frames_before,
                "frames_after": frames_after,
                "tier": tier,
            }
            result.memories_consolidated.append(info)

            if not dry_run:
                consolidated.save(brick_path)

    return result


def consolidation_stats(
    data_dir: str | Path,
    chunk: str | None = None,
) -> list[dict]:
    """Read-only: per-memory frame counts + potential savings.

    Returns a list of dicts with hex_key, chunk, text, frame_count,
    temperature, tier, consolidated, potential_frames (after consolidation).
    """
    data_dir = Path(data_dir)
    now = datetime.now(timezone.utc)
    stats = []

    if chunk is not None:
        chunks = [chunk]
    else:
        chunks = list(list_existing_chunks(data_dir))

    for chunk_name in chunks:
        chunk_dir = data_dir / "chunks" / chunk_name
        if not chunk_dir.exists():
            continue

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

            tier = temperature_tier(temp)
            text = entry.get("text", hex_key[:16])

            brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"
            if not brick_path.exists():
                stats.append({
                    "hex_key": hex_key,
                    "chunk": chunk_name,
                    "text": text,
                    "frame_count": 0,
                    "temperature": temp,
                    "tier": tier,
                    "consolidated": False,
                    "potential_frames": 0,
                })
                continue

            brick = MemoryBrick.load(brick_path)
            frame_count = len(brick.evolution_history)
            already_consolidated = brick.metadata.get("consolidated", False)

            # Estimate potential frames after consolidation
            thresholds = thresholds_for_temperature(temp)
            if thresholds is None or already_consolidated or frame_count < CONSOLIDATION_MIN_HISTORY:
                potential = frame_count
            else:
                delta_thresh, role_thresh = thresholds
                kept = select_keyframes(brick.evolution_history, delta_thresh, role_thresh)
                potential = len(kept)

            stats.append({
                "hex_key": hex_key,
                "chunk": chunk_name,
                "text": text,
                "frame_count": frame_count,
                "temperature": temp,
                "tier": tier,
                "consolidated": already_consolidated,
                "potential_frames": potential,
            })

    return stats
