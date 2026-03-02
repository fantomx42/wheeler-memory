"""Oscillation detection in role-space for epistemic uncertainty.

Cells are classified into discrete roles (local max, slope, local min).
When cells periodically switch between roles, it indicates genuine
epistemic uncertainty -- the system cannot decide the pattern's meaning.
"""

import numpy as np


def get_cell_roles(frame: np.ndarray) -> np.ndarray:
    """Classify each cell as +1 (local max), -1 (local min), or 0 (slope).

    Uses von Neumann (4-neighbor) comparison with wrapping boundaries.
    """
    n_up = np.roll(frame, 1, axis=0)
    n_down = np.roll(frame, -1, axis=0)
    n_left = np.roll(frame, 1, axis=1)
    n_right = np.roll(frame, -1, axis=1)

    is_max = (frame >= n_up) & (frame >= n_down) & (frame >= n_left) & (frame >= n_right)
    is_min = (frame <= n_up) & (frame <= n_down) & (frame <= n_left) & (frame <= n_right)

    roles = np.zeros(frame.shape, dtype=np.int8)
    roles[is_max] = 1
    roles[is_min] = -1
    return roles


def detect_oscillation(history: list[np.ndarray], window: int = 20) -> dict:
    """Detect periodic role-space oscillation in recent evolution history.

    Checks the last `window` frames for periodic patterns where cells
    cycle between roles with period p (2..10). Filters noise by requiring
    at least 1% of cells to be oscillating and requiring actual role changes.

    Returns dict:
      - oscillating: bool
      - period: int or None
      - oscillating_cells: int count
      - cycle_states: list of role matrices for one cycle, or None
    """
    if len(history) < window:
        return {"oscillating": False, "period": None, "oscillating_cells": 0, "cycle_states": None}

    recent = history[-window:]
    role_matrices = np.array([get_cell_roles(f) for f in recent], dtype=np.int8)
    total_cells = role_matrices.shape[1] * role_matrices.shape[2]

    # Check if any cells actually change roles
    role_changes = np.any(role_matrices != role_matrices[0:1], axis=0)
    if not np.any(role_changes):
        return {"oscillating": False, "period": None, "oscillating_cells": 0, "cycle_states": None}

    for p in range(2, 11):
        if p >= window:
            break

        # Check if roles[t] == roles[t+p] for all valid t
        n_checks = window - p
        matches = np.ones(role_matrices.shape[1:], dtype=bool)
        for t in range(n_checks):
            matches &= role_matrices[t] == role_matrices[t + p]

        # Cell must also actually change (not just constant)
        oscillating_mask = matches & role_changes
        n_oscillating = int(np.sum(oscillating_mask))

        if n_oscillating >= total_cells * 0.01:
            cycle_states = [role_matrices[t] for t in range(p)]
            return {
                "oscillating": True,
                "period": p,
                "oscillating_cells": n_oscillating,
                "cycle_states": cycle_states,
            }

    return {"oscillating": False, "period": None, "oscillating_cells": 0, "cycle_states": None}
