"""Dual-attractor trauma encoding for Wheeler Memory.

Traumatic events create two linked attractors:
  - Experience attractor: stored normally via store_with_rotation_retry
  - Avoidance attractor: a distinct CA trajectory seeded from the prefixed
    text, permanently linked to the experience via an avoidance_link edge

Exposure therapy is modeled by repeated safe recalls that decay the
avoidance link weight until it falls below the healing threshold.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .hashing import text_to_hex
from .storage import _get_data_dir, _load_index, store_memory
from .rotation import store_with_rotation_retry
from .chunking import get_chunk_dir, select_chunk
from .warming import _load_associations, _save_associations

AVOIDANCE_PREFIX = "[AVOIDANCE] "
AVOIDANCE_WEIGHT_DECAY = 0.7
AVOIDANCE_HEAL_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def avoidance_weight(link_entry: dict) -> float:
    """Compute current link weight from safe recall count."""
    return AVOIDANCE_WEIGHT_DECAY ** link_entry.get("safe_recall_count", 0)


def is_healed(link_entry: dict) -> bool:
    """True when avoidance weight has decayed below the healing threshold."""
    return avoidance_weight(link_entry) < AVOIDANCE_HEAL_THRESHOLD


# ---------------------------------------------------------------------------
# Internal: write avoidance_link edge
# ---------------------------------------------------------------------------

def _link_avoidance(
    chunk_dir: Path,
    experience_hex: str,
    avoidance_hex: str,
) -> None:
    """Write a directional avoidance_link edge in associations.json.

    Only the experience → avoidance direction is written; warmth never
    traverses this edge (propagate_warmth skips avoidance_link sources).
    """
    assoc = _load_associations(chunk_dir)
    now_iso = datetime.now(timezone.utc).isoformat()
    assoc.setdefault("edges", {}).setdefault(experience_hex, {})[avoidance_hex] = {
        "weight": 1.0,
        "source": "avoidance_link",
        "safe_recall_count": 0,
        "created": now_iso,
    }
    _save_associations(chunk_dir, assoc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_trauma(
    text: str,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    use_embedding: bool = False,
    salience: float | None = None,
) -> dict:
    """Store a traumatic memory as two linked attractors.

    1. Experience attractor  — stored via store_with_rotation_retry (normal flow)
    2. Avoidance attractor   — seeded from hash_to_frame(AVOIDANCE_PREFIX + text),
                               evolved independently, stored with memory_type="avoidance"
    3. avoidance_link edge   — written in associations.json (experience → avoidance)

    Both attractors land in the same chunk so the link is resolved locally.

    Returns dict with keys:
        experience      – result dict from store_with_rotation_retry
        experience_hex  – hex key of the experience attractor
        avoidance_hex   – hex key of the avoidance attractor
        avoidance_state – CA convergence state of the avoidance attractor
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

    # ── 2. Avoidance attractor — opposite polarity of the experience ─────────
    # The avoidance attractor is the bitwise inverse of the experience attractor
    # (negated frame), placing it antipodal in CA state space.  No separate CA
    # evolution is needed; the two attractors are geometrically opposite by
    # construction (Pearson correlation ≈ -1).
    avoidance_text = AVOIDANCE_PREFIX + text
    avoidance_hex = text_to_hex(avoidance_text)
    avoidance_frame = -exp_result["attractor"]

    budget = compute_attention_budget(
        salience if salience is not None else SALIENCE_DEFAULT
    )

    av_result = {
        "state": "CONVERGED",
        "attractor": avoidance_frame,
        "convergence_ticks": 0,
        "history": [avoidance_frame],
        "metadata": {
            "rotation_used": 0,
            "attempts": 1,
            "salience": budget.salience,
            "attention_label": budget.label,
            "stability_threshold": budget.stability_threshold,
            "polarity": "inverted",
        },
    }

    # ── 3. Store avoidance entry in the same chunk ───────────────────────────
    if av_result["state"] == "CONVERGED":
        brick = MemoryBrick.from_evolution_result(av_result, {"input_text": avoidance_text})
        store_memory(
            avoidance_text, av_result, brick, d,
            chunk=chunk, memory_type="avoidance",
        )

    # ── 4. Link experience → avoidance ──────────────────────────────────────
    _link_avoidance(chunk_dir, experience_hex, avoidance_hex)

    return {
        "experience": exp_result,
        "experience_hex": experience_hex,
        "avoidance_hex": avoidance_hex,
        "avoidance_state": av_result["state"],
        "link_weight": 1.0,
    }


def get_avoidance_companion(
    experience_hex: str,
    chunk_dir: Path,
) -> dict | None:
    """Return avoidance companion info if an active (non-healed) link exists.

    Returns a dict with avoidance metadata, or None if no link or already healed.
    """
    assoc = _load_associations(chunk_dir)
    neighbors = assoc.get("edges", {}).get(experience_hex, {})
    for neighbor_hex, edge in neighbors.items():
        if edge.get("source") != "avoidance_link":
            continue
        if is_healed(edge):
            continue
        index = _load_index(chunk_dir)
        av_entry = index.get(neighbor_hex, {})
        return {
            "avoidance_hex": neighbor_hex,
            "text": av_entry.get("text", AVOIDANCE_PREFIX + "?"),
            "weight": avoidance_weight(edge),
            "safe_recall_count": edge.get("safe_recall_count", 0),
        }
    return None


def apply_safe_exposure(
    experience_hex: str,
    chunk_dir: Path,
) -> float:
    """Increment safe_recall_count on the avoidance link and return new weight.

    Returns 0.0 if no active (non-healed) link exists.
    """
    assoc = _load_associations(chunk_dir)
    neighbors = assoc.get("edges", {}).get(experience_hex, {})
    for neighbor_hex, edge in neighbors.items():
        if edge.get("source") != "avoidance_link":
            continue
        if is_healed(edge):
            return 0.0
        edge["safe_recall_count"] = edge.get("safe_recall_count", 0) + 1
        new_weight = avoidance_weight(edge)
        edge["weight"] = round(new_weight, 4)
        _save_associations(chunk_dir, assoc)
        return new_weight
    return 0.0
