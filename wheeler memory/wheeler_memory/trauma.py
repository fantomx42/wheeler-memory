"""Dual-attractor trauma encoding — experience + avoidance pairs.

Traumatic events create two linked attractors: the experience itself and an
avoidance response.  When the experience attractor fires during recall, the
avoidance attractor automatically co-fires (injected into results).

Exposure therapy: repeatedly activating the experience in a safe context
(trauma_mode="safe") gradually decays the suppression strength, weakening
the automatic avoidance link.  Suppression never reaches zero — a trace
always remains (TRAUMA_SUPPRESSION_FLOOR = 0.05).

The trauma registry lives at the data_dir level (not per-chunk) because
pairs can span chunks.  File: ``trauma.json``.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .temperature import (
    TRAUMA_AVOIDANCE_TEMP_BOOST,
    TRAUMA_DECAY_PER_SAFE_EXPOSURE,
    TRAUMA_INITIAL_SUPPRESSION,
    TRAUMA_SAFE_EXPOSURE_THRESHOLD,
    TRAUMA_SUPPRESSION_FLOOR,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _trauma_path(data_dir: Path) -> Path:
    return data_dir / "trauma.json"


def _load_trauma(data_dir: Path) -> dict:
    path = _trauma_path(data_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"pairs": {}, "index_by_hex": {}}


def _save_trauma(data_dir: Path, trauma: dict) -> None:
    path = _trauma_path(data_dir)
    path.write_text(json.dumps(trauma, indent=2))


def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_index(chunk_dir: Path, index: dict) -> None:
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_trauma_pair(
    experience_text: str,
    avoidance_text: str,
    data_dir: str | Path,
    *,
    chunk: str | None = None,
    use_embedding: bool = False,
    salience: float | None = None,
) -> dict:
    """Store an experience/avoidance trauma pair.

    Both texts are stored as normal memories, then linked in trauma.json
    with an association edge of source ``trauma_pair``.

    Returns a dict with ``pair_id``, ``experience_hex``, ``avoidance_hex``.
    """
    from .chunking import select_chunk
    from .hashing import text_to_hex
    from .rotation import store_with_rotation_retry
    from .warming import _load_associations, _save_associations, _add_edge

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Store both memories
    exp_chunk = chunk if chunk is not None else select_chunk(experience_text)
    store_with_rotation_retry(
        experience_text, save=True, data_dir=data_dir,
        chunk=exp_chunk, use_embedding=use_embedding, salience=salience,
    )
    exp_hex = text_to_hex(experience_text)

    avoid_chunk = chunk if chunk is not None else select_chunk(avoidance_text)
    store_with_rotation_retry(
        avoidance_text, save=True, data_dir=data_dir,
        chunk=avoid_chunk, use_embedding=use_embedding, salience=salience,
    )
    avoid_hex = text_to_hex(avoidance_text)

    # Generate pair ID
    pair_id = uuid.uuid4().hex[:12]
    now_iso = datetime.now(timezone.utc).isoformat()

    # Update trauma registry
    trauma = _load_trauma(data_dir)
    trauma["pairs"][pair_id] = {
        "experience": {
            "hex_key": exp_hex,
            "chunk": exp_chunk,
            "text": experience_text,
        },
        "avoidance": {
            "hex_key": avoid_hex,
            "chunk": avoid_chunk,
            "text": avoidance_text,
        },
        "created": now_iso,
        "suppression_strength": TRAUMA_INITIAL_SUPPRESSION,
        "activation_count": 0,
        "avoidance_co_fire_count": 0,
        "safe_exposure_count": 0,
        "therapy_history": [],
    }
    trauma["index_by_hex"][exp_hex] = pair_id
    trauma["index_by_hex"][avoid_hex] = pair_id
    _save_trauma(data_dir, trauma)

    # Tag index metadata for both memories
    for hex_key, role, chunk_name in [
        (exp_hex, "experience", exp_chunk),
        (avoid_hex, "avoidance", avoid_chunk),
    ]:
        chunk_dir = data_dir / "chunks" / chunk_name
        index = _load_index(chunk_dir)
        if hex_key in index:
            index[hex_key]["metadata"]["trauma_pair_id"] = pair_id
            index[hex_key]["metadata"]["trauma_role"] = role
            _save_index(chunk_dir, index)

    # Create trauma_pair association edge
    # Edges live per-chunk; if cross-chunk, add to experience's chunk
    exp_chunk_dir = data_dir / "chunks" / exp_chunk
    assoc = _load_associations(exp_chunk_dir)
    _add_edge(
        assoc, exp_hex, avoid_hex, weight=1.0,
        source="trauma_pair",
        trauma_pair_id=pair_id,
        role="experience_to_avoidance",
    )
    _save_associations(exp_chunk_dir, assoc)

    # If cross-chunk, also add edge in avoidance's chunk
    if avoid_chunk != exp_chunk:
        avoid_chunk_dir = data_dir / "chunks" / avoid_chunk
        assoc2 = _load_associations(avoid_chunk_dir)
        _add_edge(
            assoc2, avoid_hex, exp_hex, weight=1.0,
            source="trauma_pair",
            trauma_pair_id=pair_id,
            role="avoidance_to_experience",
        )
        _save_associations(avoid_chunk_dir, assoc2)

    return {
        "pair_id": pair_id,
        "experience_hex": exp_hex,
        "avoidance_hex": avoid_hex,
        "experience_chunk": exp_chunk,
        "avoidance_chunk": avoid_chunk,
    }


# ---------------------------------------------------------------------------
# Activation check + injection
# ---------------------------------------------------------------------------

def check_trauma_activation(
    recalled_hex_keys: list[str],
    data_dir: Path,
) -> list[dict]:
    """Check if any recalled memories are experience attractors.

    Returns list of activation dicts: {pair_id, pair_data, role}.
    """
    trauma = _load_trauma(data_dir)
    index_by_hex = trauma.get("index_by_hex", {})
    activations = []

    for hk in recalled_hex_keys:
        if hk in index_by_hex:
            pair_id = index_by_hex[hk]
            pair = trauma["pairs"].get(pair_id)
            if pair and pair["experience"]["hex_key"] == hk:
                activations.append({
                    "pair_id": pair_id,
                    "pair_data": pair,
                    "role": "experience",
                })

    return activations


def record_trauma_activation(
    pair_id: str,
    data_dir: Path,
    safe_context: bool = False,
) -> dict:
    """Record an activation event for a trauma pair.

    If safe_context is True, records a safe exposure and decays suppression
    after the threshold is met.

    Returns updated pair data.
    """
    trauma = _load_trauma(data_dir)
    pair = trauma["pairs"].get(pair_id)
    if pair is None:
        raise KeyError(f"Trauma pair {pair_id!r} not found")

    now_iso = datetime.now(timezone.utc).isoformat()
    pair["activation_count"] += 1

    if safe_context:
        pair["safe_exposure_count"] += 1

        # Decay suppression after threshold
        if pair["safe_exposure_count"] > TRAUMA_SAFE_EXPOSURE_THRESHOLD:
            old = pair["suppression_strength"]
            new = old * (1.0 - TRAUMA_DECAY_PER_SAFE_EXPOSURE)
            pair["suppression_strength"] = max(new, TRAUMA_SUPPRESSION_FLOOR)

        pair["therapy_history"].append({
            "timestamp": now_iso,
            "type": "safe_exposure",
            "suppression_before": pair["suppression_strength"],
            "safe_exposure_count": pair["safe_exposure_count"],
        })
    else:
        if pair["suppression_strength"] > TRAUMA_SUPPRESSION_FLOOR:
            pair["avoidance_co_fire_count"] += 1

    _save_trauma(data_dir, trauma)
    return pair


def inject_avoidance_results(
    results: list[dict],
    activations: list[dict],
    data_dir: Path,
) -> list[dict]:
    """Insert avoidance entries into recall results for active trauma pairs.

    Avoidance entries are injected with effective_similarity equal to
    suppression_strength (so they rank high when strong, fade with therapy).

    Returns augmented results list.
    """
    import numpy as np

    for act in activations:
        pair = act["pair_data"]
        suppression = pair["suppression_strength"]

        if suppression <= TRAUMA_SUPPRESSION_FLOOR:
            continue  # resolved — no injection

        avoid = pair["avoidance"]
        avoid_hex = avoid["hex_key"]

        # If already in results, tag it rather than duplicating
        existing = [r for r in results if r["hex_key"] == avoid_hex]
        if existing:
            for r in existing:
                r["trauma_avoidance"] = True
                r["trauma_pair_id"] = act["pair_id"]
                r["suppression_strength"] = suppression
            continue

        avoid_chunk = avoid["chunk"]
        chunk_dir = data_dir / "chunks" / avoid_chunk
        att_path = chunk_dir / "attractors" / f"{avoid_hex}.npy"
        if not att_path.exists():
            continue

        index = _load_index(chunk_dir)
        meta = index.get(avoid_hex, {})

        results.append({
            "hex_key": avoid_hex,
            "text": avoid.get("text", meta.get("text", "")),
            "similarity": 0.0,
            "temperature": TRAUMA_AVOIDANCE_TEMP_BOOST,
            "temperature_tier": "trauma",
            "effective_similarity": suppression,
            "state": meta.get("state", "UNKNOWN"),
            "convergence_ticks": meta.get("convergence_ticks", 0),
            "timestamp": meta.get("timestamp", ""),
            "chunk": avoid_chunk,
            "trauma_avoidance": True,
            "trauma_pair_id": act["pair_id"],
            "suppression_strength": suppression,
        })

    # Re-sort by effective_similarity
    results.sort(key=lambda r: r["effective_similarity"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Therapy status
# ---------------------------------------------------------------------------

def therapy_status(pair_id: str, data_dir: str | Path) -> dict:
    """Return therapy status for a trauma pair.

    Status labels:
        "active"     — suppression > 0.5
        "in_therapy" — suppression > floor but <= 0.5
        "resolved"   — suppression <= floor
    """
    data_dir = Path(data_dir)
    trauma = _load_trauma(data_dir)
    pair = trauma["pairs"].get(pair_id)
    if pair is None:
        raise KeyError(f"Trauma pair {pair_id!r} not found")

    suppression = pair["suppression_strength"]
    if suppression <= TRAUMA_SUPPRESSION_FLOOR:
        status = "resolved"
    elif suppression > 0.5:
        status = "active"
    else:
        status = "in_therapy"

    return {
        "pair_id": pair_id,
        "status": status,
        "suppression_strength": suppression,
        "activation_count": pair["activation_count"],
        "avoidance_co_fire_count": pair["avoidance_co_fire_count"],
        "safe_exposure_count": pair["safe_exposure_count"],
        "experience": pair["experience"],
        "avoidance": pair["avoidance"],
        "created": pair["created"],
        "therapy_history": pair["therapy_history"],
    }


# ---------------------------------------------------------------------------
# List / remove
# ---------------------------------------------------------------------------

def list_trauma_pairs(data_dir: str | Path) -> list[dict]:
    """List all trauma pairs with status."""
    data_dir = Path(data_dir)
    trauma = _load_trauma(data_dir)
    result = []

    for pair_id, pair in trauma.get("pairs", {}).items():
        suppression = pair["suppression_strength"]
        if suppression <= TRAUMA_SUPPRESSION_FLOOR:
            status = "resolved"
        elif suppression > 0.5:
            status = "active"
        else:
            status = "in_therapy"

        result.append({
            "pair_id": pair_id,
            "status": status,
            "suppression_strength": suppression,
            "experience_text": pair["experience"]["text"],
            "avoidance_text": pair["avoidance"]["text"],
            "activation_count": pair["activation_count"],
            "safe_exposure_count": pair["safe_exposure_count"],
            "created": pair["created"],
        })

    return result


def remove_trauma_pair(pair_id: str, data_dir: str | Path) -> bool:
    """Unlink a trauma pair.  Keeps the memories, removes the trauma link.

    Clears index metadata (trauma_pair_id, trauma_role) from both memories.
    Returns True if the pair was found and removed.
    """
    data_dir = Path(data_dir)
    trauma = _load_trauma(data_dir)
    pair = trauma["pairs"].pop(pair_id, None)
    if pair is None:
        return False

    # Clean up index_by_hex
    for role in ("experience", "avoidance"):
        hk = pair[role]["hex_key"]
        if trauma["index_by_hex"].get(hk) == pair_id:
            del trauma["index_by_hex"][hk]

        # Clear index metadata
        chunk_name = pair[role]["chunk"]
        chunk_dir = data_dir / "chunks" / chunk_name
        index = _load_index(chunk_dir)
        if hk in index:
            index[hk]["metadata"].pop("trauma_pair_id", None)
            index[hk]["metadata"].pop("trauma_role", None)
            _save_index(chunk_dir, index)

    _save_trauma(data_dir, trauma)
    return True


# ---------------------------------------------------------------------------
# Eviction cleanup
# ---------------------------------------------------------------------------

def _cleanup_evicted_memory(data_dir: Path, hex_key: str) -> None:
    """Auto-remove trauma pair when a member is evicted.

    Called from eviction._delete_memory_files after association cleanup.
    """
    trauma = _load_trauma(data_dir)
    pair_id = trauma.get("index_by_hex", {}).get(hex_key)
    if pair_id is None:
        return

    pair = trauma["pairs"].pop(pair_id, None)
    if pair is None:
        return

    # Clean up index_by_hex for both members
    for role in ("experience", "avoidance"):
        hk = pair[role]["hex_key"]
        if trauma["index_by_hex"].get(hk) == pair_id:
            del trauma["index_by_hex"][hk]

        # Clear surviving member's index metadata (if it still exists)
        if hk != hex_key:
            chunk_name = pair[role]["chunk"]
            chunk_dir = data_dir / "chunks" / chunk_name
            index = _load_index(chunk_dir)
            if hk in index:
                index[hk]["metadata"].pop("trauma_pair_id", None)
                index[hk]["metadata"].pop("trauma_role", None)
                _save_index(chunk_dir, index)

    _save_trauma(data_dir, trauma)
