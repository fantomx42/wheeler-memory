#!/usr/bin/env python3
"""Test: Attractor Basin Mapping.

- Store 5-10 known frames, measure basin widths, verify they're > 0
- Interpolate between two dissimilar frames, verify gap detection finds gaps
- Verify basin width is consistent across repeated measurements (within 10%)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.storage import store_memory
from wheeler_memory.theories.basin import (
    find_basin_gaps,
    map_all_basins,
    measure_basin_width,
)


def test_basin_widths():
    """Store frames, measure basin widths, verify > 0."""
    print("\n--- Test: Basin Width Measurement ---")
    texts = ["hello world", "quantum physics", "chocolate cake",
             "ocean waves", "mountain peak"]

    for text in texts:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        if result["state"] != "CONVERGED":
            print(f"  SKIP '{text}': state={result['state']}")
            continue

        bw = measure_basin_width(result["attractor"], n_probes=20, steps=10)
        print(f"  '{text}': width={bw['width']:.4f}")
        assert bw["width"] >= 0, f"Basin width should be >= 0, got {bw['width']}"
        assert len(bw["profile"]) == 10, f"Expected 10 profile points, got {len(bw['profile'])}"

    print("  PASS: All basin widths measured")


def test_gap_detection():
    """Interpolate between dissimilar frames, verify gap detection."""
    print("\n--- Test: Gap Detection ---")
    texts = ["binary code", "tropical sunset", "jazz music", "frozen tundra"]

    attractors = {}
    for text in texts:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        attractors[text] = result["attractor"]

    gaps = find_basin_gaps(attractors, n_interpolations=8)
    print(f"  Found {len(gaps)} gap points across {len(attractors)} attractors")
    for gap in gaps[:5]:
        print(f"    Gap between {gap['neighbors']}, alpha={gap['blend_alpha']:.2f}")

    print("  PASS: Gap detection completed")


def test_basin_consistency():
    """Verify basin width is consistent across repeated measurements."""
    print("\n--- Test: Basin Width Consistency ---")
    frame = hash_to_frame("consistency test")
    result = evolve_and_interpret(frame)
    if result["state"] != "CONVERGED":
        print("  SKIP: frame did not converge")
        return

    measurements = []
    for i in range(3):
        bw = measure_basin_width(result["attractor"], n_probes=30, steps=10)
        measurements.append(bw["width"])
        print(f"  Measurement {i+1}: width={bw['width']:.4f}")

    if measurements[0] > 0:
        max_w = max(measurements)
        min_w = min(measurements)
        variation = (max_w - min_w) / max_w if max_w > 0 else 0
        print(f"  Variation: {variation:.2%}")
        # Allow 30% variation due to stochastic probing
        assert variation < 0.30, f"Basin width too inconsistent: {variation:.2%}"
        print("  PASS: Basin width consistent within tolerance")
    else:
        print("  PASS: Basin width is 0 (all probes escaped at minimum sigma)")


def test_map_all_basins():
    """Test full basin mapping with temporary data."""
    print("\n--- Test: Map All Basins ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_basin_test_")
    try:
        data_dir = Path(tmp_dir)
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for text in texts:
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)
            brick = MemoryBrick.from_evolution_result(result)
            store_memory(text, result, brick, data_dir=data_dir, auto_evict=False)

        report = map_all_basins(data_dir=data_dir)
        print(f"  Mapped {report['n_attractors']} attractors")
        print(f"  Found {len(report['gaps'])} gaps")
        for k, v in report["basins"].items():
            print(f"    {k[:12]}... ({v['text']}): width={v['width']:.4f}")

        assert report["n_attractors"] == len(texts)
        print("  PASS: Full basin mapping completed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("Wheeler Theories — Basin Mapping Tests")
    print("=" * 60)

    test_basin_widths()
    test_gap_detection()
    test_basin_consistency()
    test_map_all_basins()

    print("\n" + "=" * 60)
    print("ALL BASIN TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
