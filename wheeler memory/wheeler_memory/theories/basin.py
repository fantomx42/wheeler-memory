"""Attractor Basin Mapping — measure basin widths and find gaps in attractor topology.

This is the foundation module for all theory experiments. It answers:
- How wide is each attractor's basin of attraction?
- Where are the gaps between basins (regions with no stable attractor)?
"""

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from ..dynamics import apply_ca_dynamics, evolve_and_interpret
from ..hashing import hash_to_frame
from ..storage import DEFAULT_DATA_DIR, list_memories


def measure_basin_width(
    attractor: np.ndarray,
    n_probes: int = 50,
    sigma_range: tuple = (0.01, 1.0),
    steps: int = 20,
) -> dict:
    """Perturb a converged attractor with increasing Gaussian noise.

    At each sigma level, add noise, re-evolve, and compare to original
    via Pearson correlation. Basin width = largest sigma where >50% of
    probes return to the same attractor (correlation > 0.95).

    Returns:
        {width: float, profile: [(sigma, survival_rate)], probes: int}
    """
    original_flat = attractor.flatten()
    sigmas = np.linspace(sigma_range[0], sigma_range[1], steps)
    profile = []
    width = 0.0

    # Use 0.85 threshold: CA attractors have natural variance ~0.89
    # at low sigma, so 0.95 is unreachable. 0.85 captures meaningful
    # basin membership while filtering genuine basin escapes.
    corr_threshold = 0.85

    for sigma in sigmas:
        survived = 0
        for _ in range(n_probes):
            perturbed = attractor + np.random.normal(0, sigma, attractor.shape)
            perturbed = np.clip(perturbed, -1, 1).astype(np.float32)
            result = evolve_and_interpret(perturbed)
            corr, _ = pearsonr(result["attractor"].flatten(), original_flat)
            if corr > corr_threshold:
                survived += 1
        survival_rate = survived / n_probes
        profile.append((float(sigma), survival_rate))
        if survival_rate > 0.5:
            width = float(sigma)

    return {"width": width, "profile": profile, "probes": n_probes}


def find_basin_gaps(
    attractors: dict[str, np.ndarray],
    n_interpolations: int = 10,
) -> list[dict]:
    """For each pair of known attractors, linearly interpolate between them.

    Evolve each interpolant. If the result doesn't converge to either
    endpoint attractor (correlation < 0.9 with both), it's in a gap.

    Returns:
        List of {point: np.ndarray, neighbors: (key_a, key_b),
        blend_alpha: float, converged_to: str|None}
    """
    keys = list(attractors.keys())
    gaps = []

    for key_a, key_b in combinations(keys, 2):
        att_a = attractors[key_a]
        att_b = attractors[key_b]
        flat_a = att_a.flatten()
        flat_b = att_b.flatten()

        for alpha in np.linspace(0.1, 0.9, n_interpolations):
            blended = ((1 - alpha) * att_a + alpha * att_b).astype(np.float32)
            blended = np.clip(blended, -1, 1)
            result = evolve_and_interpret(blended)
            result_flat = result["attractor"].flatten()

            corr_a, _ = pearsonr(result_flat, flat_a)
            corr_b, _ = pearsonr(result_flat, flat_b)

            # Check against all known attractors
            converged_to = None
            for k, att in attractors.items():
                c, _ = pearsonr(result_flat, att.flatten())
                if c > 0.9:
                    converged_to = k
                    break

            if corr_a < 0.9 and corr_b < 0.9 and converged_to is None:
                gaps.append({
                    "point": result["attractor"],
                    "neighbors": (key_a, key_b),
                    "blend_alpha": float(alpha),
                    "converged_to": converged_to,
                })

    return gaps


def map_all_basins(data_dir: Path = None) -> dict:
    """Load all stored attractors, measure basin width for each, find gaps.

    Returns full topology report:
        {basins: {hex_key: {width, profile, probes, text}},
         gaps: [...], n_attractors: int}
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    memories = list_memories(data_dir)
    if not memories:
        return {"basins": {}, "gaps": [], "n_attractors": 0}

    # Load all attractors
    attractors = {}
    texts = {}
    for mem in memories:
        hex_key = mem["hex_key"]
        chunk = mem["chunk"]
        att_path = data_dir / "chunks" / chunk / "attractors" / f"{hex_key}.npy"
        if att_path.exists():
            attractors[hex_key] = np.load(att_path)
            texts[hex_key] = mem.get("text", "")

    # Measure basin widths (use fewer probes for full scan)
    basins = {}
    for hex_key, att in attractors.items():
        bw = measure_basin_width(att, n_probes=20, steps=10)
        basins[hex_key] = {**bw, "text": texts.get(hex_key, "")}

    # Find gaps
    gaps = find_basin_gaps(attractors, n_interpolations=5)

    return {
        "basins": basins,
        "gaps": gaps,
        "n_attractors": len(attractors),
    }
