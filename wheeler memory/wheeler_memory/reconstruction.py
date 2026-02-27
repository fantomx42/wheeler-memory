"""Reconstructive recall — context-dependent memory reconstruction.

Implements the Darman architecture concept: recalled memories are blended
with the current query context and re-evolved through the CA, producing
a reconstruction that is influenced by what you're currently thinking about.

The same stored memory reconstructs differently depending on the query:
  - Store: "Python has great libraries for data science"
  - Query A: "machine learning tools" → reconstruction biased toward ML
  - Query B: "web development" → reconstruction biased toward web
"""

import numpy as np

from .dynamics import evolve_and_interpret


def reconstruct(
    stored_attractor: np.ndarray,
    query_attractor: np.ndarray,
    alpha: float = 0.3,
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
) -> dict:
    """Blend a stored attractor with a query attractor and re-evolve.

    Args:
        stored_attractor: The stored memory's attractor (64×64).
        query_attractor: The query's attractor (64×64).
        alpha: Reconstruction weight (0=pure memory, 1=pure query).
               Default 0.3 = memory-dominant but context-aware.
        max_iters: Maximum CA iterations for re-evolution.
        stability_threshold: Convergence threshold for re-evolution.

    Returns:
        dict with:
          - attractor: the reconstructed attractor
          - state: convergence state of reconstruction
          - convergence_ticks: ticks to re-converge
          - alpha: the alpha used
          - delta_from_stored: how much the reconstruction differs from stored
          - delta_from_query: how much it differs from query
    """
    stored = stored_attractor.reshape(64, 64).astype(np.float32)
    query = query_attractor.reshape(64, 64).astype(np.float32)

    # Blend: mix = (1 - α) * stored + α * query
    blended = (1.0 - alpha) * stored + alpha * query

    # Re-evolve the blend through CA to find a new stable state
    result = evolve_and_interpret(
        blended, max_iters=max_iters, stability_threshold=stability_threshold
    )

    reconstructed = result["attractor"].flatten()
    stored_flat = stored.flatten()
    query_flat = query.flatten()

    # Measure how different the reconstruction is
    def _pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        norm = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / norm) if norm > 0 else 0.0

    return {
        "attractor": result["attractor"],
        "state": result["state"],
        "convergence_ticks": result["convergence_ticks"],
        "alpha": alpha,
        "correlation_with_stored": _pearson(reconstructed, stored_flat),
        "correlation_with_query": _pearson(reconstructed, query_flat),
        "history": result.get("history"),
        "metadata": result.get("metadata", {}),
    }


def reconstruct_batch(
    stored_attractors: list[np.ndarray],
    query_attractor: np.ndarray,
    alpha: float = 0.3,
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
) -> list[dict]:
    """Reconstruct multiple stored memories against the same query context."""
    return [
        reconstruct(
            stored, query_attractor, alpha,
            max_iters=max_iters, stability_threshold=stability_threshold,
        )
        for stored in stored_attractors
    ]
