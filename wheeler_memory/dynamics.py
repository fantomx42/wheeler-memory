"""Core cellular automata engine for Wheeler Memory.

Implements the 3-state CA dynamics: local max cells push toward +1,
local min cells push toward -1, slope cells flow toward their max neighbor.
"""

import logging

import numpy as np

from .oscillation import detect_oscillation


def apply_ca_dynamics(frame: np.ndarray) -> np.ndarray:
    """Apply a single CA iteration using 3-state logic.

    Update rules (von Neumann neighborhood, wrapping boundaries):
      - Local max (>= all 4 neighbors): delta = (1 - cell) * 0.35
      - Local min (<= all 4 neighbors): delta = (-1 - cell) * 0.35
      - Slope (neither): delta = (max_neighbor - cell) * 0.20
    """
    n_up = np.roll(frame, 1, axis=0)
    n_down = np.roll(frame, -1, axis=0)
    n_left = np.roll(frame, 1, axis=1)
    n_right = np.roll(frame, -1, axis=1)

    is_max = (frame >= n_up) & (frame >= n_down) & (frame >= n_left) & (frame >= n_right)
    is_min = (frame <= n_up) & (frame <= n_down) & (frame <= n_left) & (frame <= n_right)

    neighbors = np.stack([n_up, n_down, n_left, n_right])
    max_neighbor = np.max(neighbors, axis=0)

    delta = np.zeros_like(frame)
    delta = np.where(is_max, (1 - frame) * 0.35, delta)
    delta = np.where(is_min, (-1 - frame) * 0.35, delta)
    delta = np.where(~is_max & ~is_min, (max_neighbor - frame) * 0.20, delta)

    return np.clip(frame + delta, -1, 1)


# GPU dispatch — imported after apply_ca_dynamics is defined to avoid circular
# import (gpu_dynamics imports apply_ca_dynamics from this module).
try:
    from .gpu_dynamics import gpu_available, gpu_evolve_single as _gpu_evolve
    _GPU_READY = gpu_available()
except ImportError:
    _GPU_READY = False
    _gpu_evolve = None


def evolve_and_interpret(
    frame: np.ndarray,
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
) -> dict:
    """Run CA evolution until convergence, oscillation, or chaos.

    Returns dict with keys:
      - state: 'CONVERGED' | 'OSCILLATING' | 'CHAOTIC'
      - attractor: final frame (for CONVERGED)
      - convergence_ticks: number of iterations
      - history: list of all frame copies (for brick construction)
      - metadata: additional info (cycle_period, etc.)
    """
    if _GPU_READY and _gpu_evolve is not None:
        try:
            result = _gpu_evolve(
                frame,
                max_iters=max_iters,
                stability_threshold=stability_threshold,
            )
            # GPU path doesn't store per-tick history; synthesize a minimal
            # 2-frame history so MemoryBrick.save() (np.stack) doesn't crash.
            if not result["history"]:
                result["history"] = [frame.copy(), result["attractor"].copy()]
            return result
        except Exception as e:
            logging.warning("GPU evolution failed, falling back to CPU: %s", e)

    history = [frame.copy()]

    for i in range(max_iters):
        frame_old = frame
        frame = apply_ca_dynamics(frame)
        delta = np.abs(frame - frame_old).mean()

        history.append(frame.copy())

        if delta < stability_threshold:
            return {
                "state": "CONVERGED",
                "attractor": frame,
                "convergence_ticks": i + 1,
                "history": history,
                "metadata": {},
            }

        if i > 50 and i % 10 == 0:
            osc_result = detect_oscillation(history)
            if osc_result["oscillating"]:
                return {
                    "state": "OSCILLATING",
                    "attractor": frame,
                    "convergence_ticks": i + 1,
                    "history": history,
                    "metadata": {
                        "cycle_period": osc_result["period"],
                        "oscillating_cells": osc_result["oscillating_cells"],
                        "cycle_states": osc_result["cycle_states"],
                    },
                }

    return {
        "state": "CHAOTIC",
        "attractor": frame,
        "convergence_ticks": max_iters,
        "history": history,
        "metadata": {},
    }
