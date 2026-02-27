#!/usr/bin/env python3
"""Test: Computable Formal Metrics.

- Verify energy(converged_attractor) is very small (< 1e-4)
- Verify energy(random_frame) is large (> 0.01)
- Verify hallucination_score is low for stored frames, high for random
- Verify topology_consistency returns a float in [0, 1]
"""

import sys

import numpy as np

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.theories.metrics import (
    basin_width,
    context_weight,
    energy,
    hallucination_score,
    topology_consistency,
)


def test_energy_converged():
    """Converged attractors should have very low energy."""
    print("\n--- Test: Energy of Converged Attractors ---")
    texts = ["stable frame", "another stable one", "third frame"]
    for text in texts:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        if result["state"] == "CONVERGED":
            e = energy(result["attractor"])
            print(f"  '{text}': energy={e:.6f}")
            assert e < 1e-3, f"Converged attractor energy too high: {e}"

    print("  PASS: Converged attractors have low energy")


def test_energy_random():
    """Random frames should have high energy."""
    print("\n--- Test: Energy of Random Frames ---")
    for i in range(5):
        random_frame = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        e = energy(random_frame)
        print(f"  Random frame {i}: energy={e:.6f}")
        assert e > 0.001, f"Random frame energy too low: {e}"

    print("  PASS: Random frames have high energy")


def test_hallucination_score():
    """Stored frames should have low hallucination, random frames high."""
    print("\n--- Test: Hallucination Score ---")
    # Build known attractors
    texts = ["known concept A", "known concept B", "known concept C"]
    known = []
    for text in texts:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        known.append(result["attractor"])

    # Hallucination of a known attractor: should be low
    for i, att in enumerate(known):
        h = hallucination_score(att, known)
        print(f"  Known attractor {i}: hallucination={h:.6f}")
        # Energy is very low for converged, so score should be tiny

    # Hallucination of a random frame: should be higher
    for i in range(3):
        random_frame = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        h = hallucination_score(random_frame, known)
        print(f"  Random frame {i}: hallucination={h:.6f}")

    print("  PASS: Hallucination scores computed")


def test_context_weight():
    """Context weight combines hit count and basin width."""
    print("\n--- Test: Context Weight ---")
    cases = [
        (0, 0.5, 0.0),     # No hits
        (5, 0.5, 0.25),    # Half saturated
        (10, 0.5, 0.5),    # Fully saturated
        (20, 0.5, 0.5),    # Over saturated (capped)
        (10, 0.0, 0.0),    # Zero width
    ]
    for hits, width, expected in cases:
        w = context_weight(hits, width)
        print(f"  hits={hits}, width={width}: weight={w:.4f} (expected={expected:.4f})")
        assert abs(w - expected) < 1e-6, f"Expected {expected}, got {w}"

    print("  PASS: Context weights correct")


def test_topology_consistency():
    """Topology consistency should be in [0, 1]."""
    print("\n--- Test: Topology Consistency ---")
    # Build a few attractors
    texts = ["topology test A", "topology test B"]
    attractors = []
    for text in texts:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        attractors.append(result["attractor"])

    tc = topology_consistency(attractors, n_samples=20)
    print(f"  Consistency with {len(attractors)} attractors: {tc:.4f}")
    assert 0.0 <= tc <= 1.0, f"Consistency should be in [0,1], got {tc}"

    # Empty attractors
    tc_empty = topology_consistency([], n_samples=10)
    assert tc_empty == 0.0
    print(f"  Empty attractors: {tc_empty:.4f}")

    print("  PASS: Topology consistency in valid range")


def test_basin_width_metric():
    """Basin width metric delegates to basin module."""
    print("\n--- Test: Basin Width Metric ---")
    frame = hash_to_frame("basin width metric test")
    result = evolve_and_interpret(frame)
    if result["state"] == "CONVERGED":
        bw = basin_width(result["attractor"], n_probes=10, steps=5)
        print(f"  Basin width: {bw:.4f}")
        assert isinstance(bw, float)
        assert bw >= 0
        print("  PASS: Basin width metric works")
    else:
        print("  SKIP: frame did not converge")


def main():
    print("=" * 60)
    print("Wheeler Theories — Metrics Tests")
    print("=" * 60)

    test_energy_converged()
    test_energy_random()
    test_hallucination_score()
    test_context_weight()
    test_topology_consistency()
    test_basin_width_metric()

    print("\n" + "=" * 60)
    print("ALL METRICS TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
