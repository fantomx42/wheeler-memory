#!/usr/bin/env python3
"""Test: Lichtenberg Visualization.

- Store 5 frames, generate static topology plot, verify .png output exists
- Run apple test on fruit domain, generate animation, verify output exists
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.theories.basin import measure_basin_width
from wheeler_memory.theories.lichtenberg import animate_apple_test, plot_topology
from wheeler_memory.theories.synthesis import apple_test


def test_static_topology():
    """Generate static topology plot from known frames."""
    print("\n--- Test: Static Topology Plot ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_lichtenberg_test_")
    try:
        texts = ["red", "blue", "green", "yellow", "purple"]
        attractors = {}
        basin_widths = {}
        hit_counts = {}

        for i, text in enumerate(texts):
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)
            attractors[text] = result["attractor"]
            bw = measure_basin_width(result["attractor"], n_probes=10, steps=5)
            basin_widths[text] = bw["width"]
            hit_counts[text] = (i + 1) * 2  # Synthetic hit counts

        # Without query
        output_path = Path(tmp_dir) / "topology_basic.png"
        fig = plot_topology(attractors, basin_widths, hit_counts,
                            output_path=output_path)
        assert output_path.exists(), f"Output file not created: {output_path}"
        print(f"  Basic plot: {output_path} ({output_path.stat().st_size} bytes)")

        # With query and candidate
        query_frame = hash_to_frame("orange")
        query_result = evolve_and_interpret(query_frame)
        candidate = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        candidate_result = evolve_and_interpret(candidate)

        output_path2 = Path(tmp_dir) / "topology_full.png"
        fig2 = plot_topology(
            attractors, basin_widths, hit_counts,
            query_attractor=query_result["attractor"],
            candidates=[candidate_result["attractor"]],
            output_path=output_path2,
        )
        assert output_path2.exists(), f"Output file not created: {output_path2}"
        print(f"  Full plot: {output_path2} ({output_path2.stat().st_size} bytes)")

        print("  PASS: Static topology plots generated")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_empty_topology():
    """Generate plot with no attractors."""
    print("\n--- Test: Empty Topology Plot ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_empty_topo_")
    try:
        output_path = Path(tmp_dir) / "empty.png"
        fig = plot_topology({}, {}, {}, output_path=output_path)
        assert output_path.exists()
        print(f"  Empty plot: {output_path} ({output_path.stat().st_size} bytes)")
        print("  PASS: Empty topology handled")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_animation():
    """Run apple test and animate the result."""
    print("\n--- Test: Apple Test Animation ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_anim_test_")
    try:
        fruits = ["apple", "orange", "banana", "grape", "mango"]
        result = apple_test(fruits, holdout="apple")

        output_path = Path(tmp_dir) / "apple_test.gif"
        animate_apple_test(result, output_path)

        assert output_path.exists(), f"Animation not created: {output_path}"
        size = output_path.stat().st_size
        print(f"  Animation: {output_path} ({size} bytes)")
        assert size > 0, "Animation file is empty"

        print(f"  Verdict animated: {result['verdict']}")
        print("  PASS: Animation generated")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("Wheeler Theories — Lichtenberg Visualization Tests")
    print("=" * 60)

    test_static_topology()
    test_empty_topology()
    test_animation()

    print("\n" + "=" * 60)
    print("ALL LICHTENBERG TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
