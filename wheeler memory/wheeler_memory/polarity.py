"""Dual-polarity encoding for Wheeler Memory.

Dual-polarity events create two linked attractors:
  - Experience attractor: stored normally via store_with_rotation_retry
  - Polar attractor: a geometrically antipodal CA state (bitwise negation of
    the experience attractor), permanently linked via a polarity_link edge

Polar decay is modeled by repeated recalls that decay the polarity link weight
until it falls below the neutralization threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .hashing import text_to_hex
from .storage import _get_data_dir, _load_index, store_memory
from .rotation import store_with_rotation_retry
from .chunking import get_chunk_dir, select_chunk
from .warming import _load_associations, _save_associations

POLAR_PREFIX = "[POLAR] "
POLAR_WEIGHT_DECAY = 0.7
POLAR_DECAY_THRESHOLD = 0.1
EDGE_SOURCE_POLARITY = "polarity_link"


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def polar_weight(link_entry: dict) -> float:
    """Compute current link weight from decay count."""
    count = link_entry.get("decay_count", link_entry.get("safe_recall_count", 0))
    return POLAR_WEIGHT_DECAY ** count


def is_neutralized(link_entry: dict) -> bool:
    """True when polar link weight has decayed below the neutralization threshold."""
    return polar_weight(link_entry) < POLAR_DECAY_THRESHOLD


# ---------------------------------------------------------------------------
# Internal: write polarity_link edge
# ---------------------------------------------------------------------------

def _link_polarity(
    chunk_dir: Path,
    experience_hex: str,
    polar_hex: str,
) -> None:
    """Write a directional polarity_link edge in associations.json.

    Only the experience → polar direction is written; warmth never
    traverses this edge (propagate_warmth skips polarity_link sources).
    """
    assoc = _load_associations(chunk_dir)
    now_iso = datetime.now(timezone.utc).isoformat()
    assoc.setdefault("edges", {}).setdefault(experience_hex, {})[polar_hex] = {
        "weight": 1.0,
        "source": EDGE_SOURCE_POLARITY,
        "decay_count": 0,
        "created": now_iso,
    }
    _save_associations(chunk_dir, assoc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_dual(
    text: str,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    use_embedding: bool = False,
    salience: float | None = None,
) -> dict:
    """Store a dual-polarity memory as two linked attractors.

    1. Experience attractor  — stored via store_with_rotation_retry (normal flow)
    2. Polar attractor       — the bitwise negation of the experience attractor,
                               placing it antipodal in CA state space (r ≈ −1.0),
                               stored with memory_type="polar"
    3. polarity_link edge    — written in associations.json (experience → polar)

    Both attractors land in the same chunk so the link is resolved locally.

    Returns dict with keys:
        experience      – result dict from store_with_rotation_retry
        experience_hex  – hex key of the experience attractor
        polar_hex       – hex key of the polar attractor
        polar_state     – CA convergence state of the polar attractor
        link_weight     – initial link weight (always 1.0)
    """
    from .attention import compute_attention_budget
    from .brick import MemoryBrick
    from .temperature import SALIENCE_DEFAULT

    d = _get_data_dir(data_dir)
    if chunk is None:
        chunk = select_chunk(text)
    chunk_dir = get_chunk_dir(d, chunk)

    # ── 1. Experience attractor ──────────────────────────────────────────────
    exp_result = store_with_rotation_retry(
        text, data_dir=d, chunk=chunk, use_embedding=use_embedding, salience=salience,
    )
    experience_hex = text_to_hex(text)

    # ── 2. Polar attractor — opposite polarity of the experience ─────────────
    # The polar attractor is the bitwise negation of the experience attractor
    # (negated frame), placing it antipodal in CA state space.  No separate CA
    # evolution is needed; the two attractors are geometrically opposite by
    # construction (Pearson correlation ≈ -1).
    polar_text = POLAR_PREFIX + text
    polar_hex = text_to_hex(polar_text)
    polar_frame = -exp_result["attractor"]

    budget = compute_attention_budget(
        salience if salience is not None else SALIENCE_DEFAULT
    )

    pol_result = {
        "state": "CONVERGED",
        "attractor": polar_frame,
        "convergence_ticks": 0,
        "history": [polar_frame],
        "metadata": {
            "rotation_used": 0,
            "attempts": 1,
            "salience": budget.salience,
            "attention_label": budget.label,
            "stability_threshold": budget.stability_threshold,
            "polarity": "inverted",
        },
    }

    # ── 3. Store polar entry in the same chunk ───────────────────────────────
    brick = MemoryBrick.from_evolution_result(pol_result, {"input_text": polar_text})
    store_memory(
        polar_text, pol_result, brick, d,
        chunk=chunk, memory_type="polar",
    )

    # ── 4. Link experience → polar ───────────────────────────────────────────
    _link_polarity(chunk_dir, experience_hex, polar_hex)

    return {
        "experience": exp_result,
        "experience_hex": experience_hex,
        "polar_hex": polar_hex,
        "polar_state": pol_result["state"],
        "link_weight": 1.0,
    }


def get_polar_companion_from_assoc(
    hex_key: str,
    assoc: dict,
    index: dict,
) -> dict | None:
    """Return polar companion info using pre-loaded assoc and index dicts.

    Returns a dict with polar metadata, or None if no link or already neutralized.
    Accepts pre-loaded data to avoid redundant disk reads when batching.
    Accepts both "polarity_link" and legacy "avoidance_link" edge sources.
    """
    neighbors = assoc.get("edges", {}).get(hex_key, {})
    for neighbor_hex, edge in neighbors.items():
        if edge.get("source") not in (EDGE_SOURCE_POLARITY, "avoidance_link"):
            continue
        if is_neutralized(edge):
            continue
        pol_entry = index.get(neighbor_hex, {})
        return {
            "polar_hex": neighbor_hex,
            "text": pol_entry.get("text", POLAR_PREFIX + "?"),
            "weight": polar_weight(edge),
            "decay_count": edge.get("decay_count", edge.get("safe_recall_count", 0)),
        }
    return None


def get_polar_companion(
    experience_hex: str,
    chunk_dir: Path,
) -> dict | None:
    """Return polar companion info if an active (non-neutralized) link exists.

    Returns a dict with polar metadata, or None if no link or already neutralized.
    """
    assoc = _load_associations(chunk_dir)
    index = _load_index(chunk_dir)
    return get_polar_companion_from_assoc(experience_hex, assoc, index)


def apply_polar_decay_in_place(
    experience_hex: str,
    assoc: dict,
) -> float:
    """Increment decay_count on the polarity link in the given assoc dict.

    Mutates assoc in place but does NOT save to disk — caller is responsible.
    Returns new weight, or 0.0 if no active (non-neutralized) link exists.
    Accepts both "polarity_link" and legacy "avoidance_link" edge sources.
    """
    neighbors = assoc.get("edges", {}).get(experience_hex, {})
    for neighbor_hex, edge in neighbors.items():
        if edge.get("source") not in (EDGE_SOURCE_POLARITY, "avoidance_link"):
            continue
        if is_neutralized(edge):
            return 0.0
        old_count = edge.get("decay_count", edge.get("safe_recall_count", 0))
        edge["decay_count"] = old_count + 1
        new_weight = polar_weight(edge)
        edge["weight"] = round(new_weight, 4)
        return new_weight
    return 0.0


def apply_polar_decay(
    experience_hex: str,
    chunk_dir: Path,
) -> float:
    """Increment decay_count on the polarity link and return new weight.

    Returns 0.0 if no active (non-neutralized) link exists.
    """
    assoc = _load_associations(chunk_dir)
    new_weight = apply_polar_decay_in_place(experience_hex, assoc)
    if new_weight > 0.0:
        _save_associations(chunk_dir, assoc)
    return new_weight
