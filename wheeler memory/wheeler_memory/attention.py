"""Attention model — variable tick rates based on salience.

Maps a salience score [0, 1] to an AttentionBudget that controls how many
CA iterations and how tight a convergence threshold a memory gets.

High-salience inputs get more patience (higher max_iters) and tighter
convergence (lower stability_threshold), forming deeper attractors.
Low-salience inputs converge quickly with a looser threshold.

Salience 0.5 produces exactly the current defaults (max_iters=1000,
stability_threshold=1e-4), ensuring backwards compatibility.
"""

import math
from dataclasses import dataclass

from .temperature import (
    SALIENCE_DEFAULT,
    SALIENCE_MAX_ITERS_HIGH,
    SALIENCE_MAX_ITERS_LOW,
    SALIENCE_MAX_ITERS_MED,
    SALIENCE_THRESHOLD_HIGH,
    SALIENCE_THRESHOLD_LOW,
    SALIENCE_THRESHOLD_MED,
)

_SALIENCE_LABELS = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.9,
}


@dataclass(frozen=True)
class AttentionBudget:
    """CA evolution budget derived from salience."""

    max_iters: int
    stability_threshold: float
    salience: float

    @property
    def label(self) -> str:
        """Human-readable label: low / medium / high."""
        if self.salience <= 0.35:
            return "low"
        if self.salience <= 0.7:
            return "medium"
        return "high"


def compute_attention_budget(salience: float) -> AttentionBudget:
    """Map salience [0, 1] to an AttentionBudget.

    Interpolation:
      - max_iters: piecewise linear (low→med for s<0.5, med→high for s≥0.5)
      - stability_threshold: log-linear (spans orders of magnitude)

    Salience is clamped to [0, 1].
    """
    salience = max(0.0, min(1.0, salience))

    # Piecewise linear interpolation for max_iters
    if salience <= 0.5:
        t = salience / 0.5
        max_iters = SALIENCE_MAX_ITERS_LOW + t * (SALIENCE_MAX_ITERS_MED - SALIENCE_MAX_ITERS_LOW)
    else:
        t = (salience - 0.5) / 0.5
        max_iters = SALIENCE_MAX_ITERS_MED + t * (SALIENCE_MAX_ITERS_HIGH - SALIENCE_MAX_ITERS_MED)

    # Log-linear interpolation for threshold
    # salience=0 → THRESHOLD_LOW, salience=0.5 → THRESHOLD_MED, salience=1.0 → THRESHOLD_HIGH
    log_low = math.log(SALIENCE_THRESHOLD_LOW)
    log_med = math.log(SALIENCE_THRESHOLD_MED)
    log_high = math.log(SALIENCE_THRESHOLD_HIGH)
    if salience <= 0.5:
        t = salience / 0.5
        log_thresh = log_low + t * (log_med - log_low)
    else:
        t = (salience - 0.5) / 0.5
        log_thresh = log_med + t * (log_high - log_med)

    return AttentionBudget(
        max_iters=int(round(max_iters)),
        stability_threshold=math.exp(log_thresh),
        salience=salience,
    )


def salience_from_temperature(temperature: float) -> float:
    """Derive salience from a memory's temperature.

    temp 0 → 0.1 (still give some attention)
    temp 1 → 1.0
    Linear mapping: salience = 0.1 + 0.9 * temperature
    """
    return 0.1 + 0.9 * max(0.0, min(1.0, temperature))


def salience_from_label(label: str) -> float:
    """Convert a human label to a salience score.

    "low" → 0.2, "medium" → 0.5, "high" → 0.9.
    Unknown labels return SALIENCE_DEFAULT (0.5).
    """
    return _SALIENCE_LABELS.get(label.lower(), SALIENCE_DEFAULT)
