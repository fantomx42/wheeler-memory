"""Associative warming -- spreading activation between related memories.

When a memory fires (is recalled), associated memories receive a temporary
temperature boost that decays with a half-life of 1 day.  Associations
are formed at store time (attractor correlation) and by co-recall patterns.

Association graph and warmth state are persisted per-chunk in
associations.json alongside the existing index.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .temperature import (
    MAX_WARMTH,
    WARMTH_FLOOR,
    WARMTH_HOP1,
    WARMTH_HOP2,
    compute_warmth,
)

ASSOCIATION_THRESHOLD = 0.5  # Minimum Pearson correlation for store-time edge


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_associations(chunk_dir: Path) -> dict:
    """Load associations.json for a chunk, returning default if absent."""
    path = chunk_dir / "associations.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"edges": {}, "warmth": {}}


def _save_associations(chunk_dir: Path, assoc: dict) -> None:
    """Write associations.json for a chunk."""
    path = chunk_dir / "associations.json"
    path.write_text(json.dumps(assoc, indent=2))


def load_associations(chunk_dir: Path) -> dict:
    """Public accessor for the full associations dict."""
    return _load_associations(chunk_dir)


def load_warmth(chunk_dir: Path) -> dict:
    """Load the warmth map from associations.json.

    Returns dict of {hex_key: {"boost": float, "applied_at": str}}.
    Garbage-collects expired entries (below WARMTH_FLOOR).
    """
    assoc = _load_associations(chunk_dir)
    warmth = assoc.get("warmth", {})
    now = datetime.now(timezone.utc)
    cleaned = {}
    for hk, entry in warmth.items():
        decayed = compute_warmth(entry["boost"], entry["applied_at"], now=now)
        if decayed >= WARMTH_FLOOR:
            cleaned[hk] = entry
    if len(cleaned) != len(warmth):
        assoc["warmth"] = cleaned
        _save_associations(chunk_dir, assoc)
    return cleaned


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def _add_edge(assoc: dict, key_a: str, key_b: str, weight: float, source: str) -> None:
    """Add a bidirectional edge to the association graph."""
    now_iso = datetime.now(timezone.utc).isoformat()
    edges = assoc["edges"]
    edge_data = {
        "weight": round(weight, 4),
        "created": now_iso,
        "source": source,
    }
    edges.setdefault(key_a, {})[key_b] = edge_data
    edges.setdefault(key_b, {})[key_a] = {**edge_data}


def _has_edge(assoc: dict, key_a: str, key_b: str) -> bool:
    """Check if an edge exists between two keys."""
    return key_b in assoc.get("edges", {}).get(key_a, {})


def get_neighbors(chunk_dir: Path, hex_key: str) -> dict:
    """Return neighbors of a memory in the association graph.

    Returns dict of {neighbor_hex_key: {"weight", "source", "created"}}.
    """
    assoc = _load_associations(chunk_dir)
    return assoc.get("edges", {}).get(hex_key, {})


# ---------------------------------------------------------------------------
# Association building
# ---------------------------------------------------------------------------

def build_store_associations(
    chunk_dir: Path,
    new_hex_key: str,
    threshold: float = ASSOCIATION_THRESHOLD,
) -> int:
    """Compare new attractor against all in chunk, create edges above threshold.

    Returns the number of new edges created.
    """
    new_att_path = chunk_dir / "attractors" / f"{new_hex_key}.npy"
    if not new_att_path.exists():
        return 0

    new_att = np.load(new_att_path).flatten()
    assoc = _load_associations(chunk_dir)
    count = 0

    for att_path in (chunk_dir / "attractors").glob("*.npy"):
        other_key = att_path.stem
        if other_key == new_hex_key:
            continue
        if _has_edge(assoc, new_hex_key, other_key):
            continue
        other_att = np.load(att_path).flatten()
        corr, _ = pearsonr(new_att, other_att)
        if corr >= threshold:
            _add_edge(assoc, new_hex_key, other_key, weight=float(corr), source="store_correlation")
            count += 1

    if count > 0:
        _save_associations(chunk_dir, assoc)
    return count


def build_co_recall_associations(chunk_dir: Path, hex_keys: list[str]) -> int:
    """Create edges between co-recalled memories within the same chunk.

    Returns the number of new edges created.
    """
    if len(hex_keys) < 2:
        return 0

    assoc = _load_associations(chunk_dir)
    count = 0

    for i, a in enumerate(hex_keys):
        for b in hex_keys[i + 1:]:
            if _has_edge(assoc, a, b):
                continue
            att_a_path = chunk_dir / "attractors" / f"{a}.npy"
            att_b_path = chunk_dir / "attractors" / f"{b}.npy"
            if not att_a_path.exists() or not att_b_path.exists():
                continue
            att_a = np.load(att_a_path).flatten()
            att_b = np.load(att_b_path).flatten()
            corr, _ = pearsonr(att_a, att_b)
            _add_edge(assoc, a, b, weight=float(corr), source="co_recall")
            count += 1

    if count > 0:
        _save_associations(chunk_dir, assoc)
    return count


# ---------------------------------------------------------------------------
# Warmth propagation
# ---------------------------------------------------------------------------

def remove_memory_from_associations(chunk_dir: Path, hex_key: str) -> int:
    """Remove all edges and warmth for a memory. Returns edges removed."""
    assoc = _load_associations(chunk_dir)
    edges = assoc.get("edges", {})
    count = 0

    # Remove edges pointing FROM this key
    if hex_key in edges:
        count += len(edges[hex_key])
        del edges[hex_key]

    # Remove edges pointing TO this key from other nodes
    for other_key in list(edges.keys()):
        if hex_key in edges[other_key]:
            del edges[other_key][hex_key]
            count += 1
            # Clean up empty neighbor dicts
            if not edges[other_key]:
                del edges[other_key]

    # Remove warmth entry
    warmth = assoc.get("warmth", {})
    warmth.pop(hex_key, None)

    _save_associations(chunk_dir, assoc)
    return count


def propagate_warmth(chunk_dir: Path, fired_keys: list[str]) -> dict[str, float]:
    """Spread warmth from fired memories to their neighbors (max 2 hops).

    Returns dict of {hex_key: total_new_boost} for all warmed memories.
    """
    assoc = _load_associations(chunk_dir)
    edges = assoc.get("edges", {})
    warmth = assoc.setdefault("warmth", {})
    now_iso = datetime.now(timezone.utc).isoformat()

    fired_set = set(fired_keys)
    warmed: dict[str, float] = {}

    for fired in fired_keys:
        visited = {fired}

        # Hop 1 — avoidance_link edges do not spread warmth
        for n1, edge1 in edges.get(fired, {}).items():
            if edge1.get("source") == "avoidance_link":
                continue
            if n1 in fired_set:
                continue
            if n1 not in visited:
                visited.add(n1)
                warmed[n1] = warmed.get(n1, 0.0) + WARMTH_HOP1

            # Hop 2 — avoidance_link edges do not spread warmth
            for n2, edge2 in edges.get(n1, {}).items():
                if edge2.get("source") == "avoidance_link":
                    continue
                if n2 in visited or n2 in fired_set:
                    continue
                visited.add(n2)
                warmed[n2] = warmed.get(n2, 0.0) + WARMTH_HOP2

    # Apply warmth, respecting MAX_WARMTH cap
    for hk, boost in warmed.items():
        existing = warmth.get(hk, {})
        existing_boost = existing.get("boost", 0.0)
        if existing_boost > 0 and "applied_at" in existing:
            existing_boost = compute_warmth(existing_boost, existing["applied_at"])
        new_boost = min(existing_boost + boost, MAX_WARMTH)
        warmth[hk] = {"boost": round(new_boost, 4), "applied_at": now_iso}

    if warmed:
        _save_associations(chunk_dir, assoc)

    return warmed
