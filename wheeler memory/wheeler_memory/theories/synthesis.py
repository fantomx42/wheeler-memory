"""Novel Frame Synthesis — the Apple Test.

Given a gap in the attractor topology, predict what frame belongs there
without ever having seen content about that concept directly. Tests whether
CA dynamics create meaningful attractor topology independent of input encoding.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from ..brick import MemoryBrick
from ..dynamics import evolve_and_interpret
from ..hashing import hash_to_frame
from ..storage import DEFAULT_DATA_DIR, store_memory
from .basin import find_basin_gaps


def synthesize_from_gap(
    gap_point: np.ndarray,
    neighbor_attractors: list[np.ndarray],
) -> dict:
    """Given a gap point and its neighboring attractors, synthesize a candidate.

    Evolves the gap centroid through CA dynamics. Returns:
        {candidate: np.ndarray, state: str, confidence: float,
         neighbor_correlations: list[float]}
    """
    result = evolve_and_interpret(gap_point)
    candidate = result["attractor"]
    candidate_flat = candidate.flatten()

    neighbor_corrs = []
    for att in neighbor_attractors:
        corr, _ = pearsonr(candidate_flat, att.flatten())
        neighbor_corrs.append(float(corr))

    # Confidence: inverse of max correlation with neighbors
    # (high confidence = candidate is distinct from all neighbors)
    max_corr = max(neighbor_corrs) if neighbor_corrs else 0.0
    confidence = 1.0 - max(0.0, max_corr)

    return {
        "candidate": candidate,
        "state": result["state"],
        "confidence": confidence,
        "neighbor_correlations": neighbor_corrs,
    }


def apple_test(
    domain_items: list[str],
    holdout: str,
    data_dir: Path = None,
) -> dict:
    """The full Apple Test protocol.

    1. Store all domain_items except holdout
    2. Map basins and find gaps
    3. Synthesize candidate frame(s) from largest gap
    4. Expose to holdout text (hash_to_frame + evolve)
    5. Measure: does holdout converge toward candidate (convergence),
       fall to a neighbor (dissolution), or drift (hallucination)?

    Returns:
        {verdict: str, holdout_attractor: np.ndarray, candidate: np.ndarray,
         correlation: float, trajectory: list, log: list[str]}
    """
    # Use a temporary directory to avoid polluting real storage
    if data_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="wheeler_apple_")
        data_dir = Path(tmp_dir)
        cleanup = True
    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False

    log = []
    training_items = [item for item in domain_items if item != holdout]

    # Step 1: Store all training items
    log.append(f"Storing {len(training_items)} training items (holdout: {holdout})")
    attractors = {}
    for item in training_items:
        frame = hash_to_frame(item)
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)
        hex_key = store_memory(item, result, brick, data_dir=data_dir, auto_evict=False)
        attractors[hex_key] = result["attractor"]
        log.append(f"  Stored '{item}' → {hex_key[:12]}... ({result['state']})")

    # Step 2: Find gaps
    log.append(f"Finding gaps between {len(attractors)} attractors...")
    gaps = find_basin_gaps(attractors, n_interpolations=8)
    log.append(f"  Found {len(gaps)} gap points")

    if not gaps:
        log.append("No gaps found — all basins are contiguous. Synthesizing from centroid.")
        # Fallback: blend all attractors as centroid
        all_atts = list(attractors.values())
        centroid = np.mean(all_atts, axis=0).astype(np.float32)
        centroid = np.clip(centroid, -1, 1)
        gaps = [{"point": centroid, "neighbors": (list(attractors.keys())[0],
                                                   list(attractors.keys())[-1]),
                 "blend_alpha": 0.5}]

    # Step 3: Synthesize from largest gap (first gap point as proxy)
    gap = gaps[0]
    neighbor_keys = gap["neighbors"]
    neighbor_atts = [attractors[k] for k in neighbor_keys if k in attractors]
    synth = synthesize_from_gap(gap["point"], neighbor_atts)
    candidate = synth["candidate"]
    log.append(f"Synthesized candidate from gap (state: {synth['state']}, "
               f"confidence: {synth['confidence']:.3f})")

    # Step 4: Expose to holdout
    log.append(f"Exposing holdout '{holdout}' to CA dynamics...")
    holdout_frame = hash_to_frame(holdout)
    holdout_result = evolve_and_interpret(holdout_frame)
    holdout_attractor = holdout_result["attractor"]
    holdout_flat = holdout_attractor.flatten()

    # Build trajectory: correlation with candidate at each history step
    candidate_flat = candidate.flatten()
    trajectory = []
    for step_frame in holdout_result["history"]:
        corr, _ = pearsonr(step_frame.flatten(), candidate_flat)
        trajectory.append(float(corr))

    # Step 5: Verdict
    corr_with_candidate, _ = pearsonr(holdout_flat, candidate_flat)
    corr_with_candidate = float(corr_with_candidate)

    # Check dissolution: does holdout match any existing attractor?
    max_neighbor_corr = 0.0
    dissolved_to = None
    for k, att in attractors.items():
        c, _ = pearsonr(holdout_flat, att.flatten())
        if c > max_neighbor_corr:
            max_neighbor_corr = float(c)
            dissolved_to = k

    if corr_with_candidate > 0.7:
        verdict = "convergence"
        log.append(f"CONVERGENCE: holdout correlates {corr_with_candidate:.3f} with candidate")
    elif max_neighbor_corr > 0.9:
        dissolved_text = next(
            (item for item in training_items
             if hash_to_frame.__module__ and True),  # always true, just get item
            None,
        )
        verdict = "dissolution"
        log.append(f"DISSOLUTION: holdout fell to existing attractor {dissolved_to[:12]}... "
                   f"(corr={max_neighbor_corr:.3f})")
    else:
        verdict = "hallucination"
        log.append(f"HALLUCINATION: holdout drifted (candidate corr={corr_with_candidate:.3f}, "
                   f"max neighbor corr={max_neighbor_corr:.3f})")

    # Cleanup temp dir
    if cleanup:
        shutil.rmtree(data_dir, ignore_errors=True)

    return {
        "verdict": verdict,
        "holdout_attractor": holdout_attractor,
        "candidate": candidate,
        "correlation": corr_with_candidate,
        "max_neighbor_correlation": max_neighbor_corr,
        "dissolved_to": dissolved_to,
        "trajectory": trajectory,
        "log": log,
        "n_gaps": len(gaps),
        "synthesis": synth,
    }


def run_apple_battery() -> dict:
    """Run apple test across three domains.

    - Fruits: holdout=apple
    - Languages: holdout=rust
    - Emotions: holdout=fear

    Returns results for all three with verdicts.
    """
    domains = {
        "fruits": {
            "items": ["apple", "orange", "banana", "grape", "mango",
                       "strawberry", "pear", "peach"],
            "holdout": "apple",
        },
        "languages": {
            "items": ["python", "javascript", "rust", "go", "java",
                       "c", "ruby", "haskell"],
            "holdout": "rust",
        },
        "emotions": {
            "items": ["joy", "anger", "sadness", "fear", "surprise",
                       "disgust", "love", "hope"],
            "holdout": "fear",
        },
    }

    results = {}
    for domain_name, config in domains.items():
        print(f"\n{'='*60}")
        print(f"Apple Test: {domain_name} (holdout: {config['holdout']})")
        print(f"{'='*60}")
        result = apple_test(config["items"], config["holdout"])
        results[domain_name] = {
            "verdict": result["verdict"],
            "correlation": result["correlation"],
            "max_neighbor_correlation": result["max_neighbor_correlation"],
            "n_gaps": result["n_gaps"],
            "log": result["log"],
        }
        for line in result["log"]:
            print(f"  {line}")
        print(f"  VERDICT: {result['verdict']}")

    return results
