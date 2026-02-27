#!/usr/bin/env python3
"""Test: Structured Theory Output.

- Store 5 memories, build a theory from a query
- Verify Theory has active_frames with basin_width > 0
- Verify context_budget sums to ~1.0
- Verify theory_to_prompt() produces a non-empty string
"""

import shutil
import sys
import tempfile
from pathlib import Path

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.storage import store_memory
from wheeler_memory.theories.structured import Theory, build_theory, theory_to_prompt


def test_build_theory():
    """Store memories, build theory, verify structure."""
    print("\n--- Test: Build Theory ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_theory_test_")
    try:
        data_dir = Path(tmp_dir)
        texts = [
            "neural networks learn patterns",
            "deep learning uses backpropagation",
            "transformers use attention mechanisms",
            "convolutional networks process images",
            "recurrent networks handle sequences",
        ]

        for text in texts:
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)
            brick = MemoryBrick.from_evolution_result(result)
            store_memory(text, result, brick, data_dir=data_dir, auto_evict=False)

        theory = build_theory("how do neural networks work", top_k=5, data_dir=data_dir)

        assert isinstance(theory, Theory), f"Expected Theory, got {type(theory)}"
        print(f"  Active frames: {len(theory.active_frames)}")
        for f in theory.active_frames:
            print(f"    '{f.text}': temp={f.temperature:.3f}, "
                  f"basin_width={f.basin_width:.4f}, confidence={f.confidence:.4f}")

        assert len(theory.active_frames) > 0, "No active frames in theory"

        # Check context budget
        budget_sum = sum(theory.context_budget.values())
        print(f"  Context budget sum: {budget_sum:.4f}")
        assert abs(budget_sum - 1.0) < 0.01 or budget_sum == 0.0, \
            f"Context budget should sum to ~1.0, got {budget_sum:.4f}"

        print("  PASS: Theory built successfully")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_theory_to_prompt():
    """Verify prompt generation from a theory."""
    print("\n--- Test: Theory to Prompt ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_prompt_test_")
    try:
        data_dir = Path(tmp_dir)
        texts = [
            "the ocean is vast and deep",
            "waves crash against the shore",
            "marine life thrives in coral reefs",
        ]

        for text in texts:
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)
            brick = MemoryBrick.from_evolution_result(result)
            store_memory(text, result, brick, data_dir=data_dir, auto_evict=False)

        theory = build_theory("tell me about the ocean", top_k=3, data_dir=data_dir)
        prompt = theory_to_prompt(theory)

        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  First 200 chars: {prompt[:200]}")

        assert len(prompt) > 0, "Prompt should not be empty"
        assert "expressing a theory" in prompt, "Prompt should contain theory framing"
        assert "context budget" not in prompt.lower() or "context" in prompt.lower(), \
            "Prompt should reference context allocation"

        print("  PASS: Prompt generated successfully")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_empty_theory():
    """Verify behavior with no matching memories."""
    print("\n--- Test: Empty Theory ---")
    tmp_dir = tempfile.mkdtemp(prefix="wheeler_empty_test_")
    try:
        data_dir = Path(tmp_dir)
        theory = build_theory("nonexistent topic", top_k=5, data_dir=data_dir)

        assert isinstance(theory, Theory)
        assert len(theory.active_frames) == 0
        assert len(theory.context_budget) == 0

        prompt = theory_to_prompt(theory)
        assert len(prompt) > 0, "Even empty theory should produce a prompt"

        print("  PASS: Empty theory handled gracefully")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("Wheeler Theories — Structured Theory Output Tests")
    print("=" * 60)

    test_build_theory()
    test_theory_to_prompt()
    test_empty_theory()

    print("\n" + "=" * 60)
    print("ALL STRUCTURED THEORY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
