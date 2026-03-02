#!/usr/bin/env python3
"""Integration test for sleep consolidation.

Stores memories, verifies that:
1. select_keyframes preserves seed + attractor
2. min_frames floor — uniform history keeps at least 3 frames
3. consolidate_brick metadata — correct fields, frame reduction, seed/attractor preserved
4. idempotent — consolidate twice, no further reduction
5. skips small history — brick with <5 frames returned unchanged
6. temperature-tiered — hot skipped, warm light, cold aggressive, warm > cold
7. dry_run — report shows savings but brick unchanged on disk
8. re-save integrity — consolidated brick loads correctly, attractor matches
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.consolidation import (
    ConsolidationResult,
    consolidate_brick,
    consolidation_stats,
    select_keyframes,
    sleep_consolidate,
    thresholds_for_temperature,
)
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame, text_to_hex
from wheeler_memory.storage import _load_index, store_memory
from wheeler_memory.temperature import (
    CONSOLIDATION_DELTA_COLD,
    CONSOLIDATION_DELTA_WARM,
    CONSOLIDATION_MIN_FRAMES,
    CONSOLIDATION_MIN_HISTORY,
    CONSOLIDATION_ROLE_COLD,
    CONSOLIDATION_ROLE_WARM,
)

passes = 0
fails = 0


def check(name: str, condition: bool, detail: str = ""):
    global passes, fails
    if condition:
        passes += 1
        print(f"  \u2713 {name}")
    else:
        fails += 1
        msg = f"  \u2717 {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


def make_synthetic_history(n_frames: int, transitions: list[int] | None = None) -> list[np.ndarray]:
    """Create a synthetic evolution history with known transition points.

    Frames are mostly constant with small noise. At transition indices,
    there is a large change to create clear keyframe candidates.
    """
    rng = np.random.RandomState(42)
    base = rng.randn(64, 64).astype(np.float32) * 0.5
    history = []

    for i in range(n_frames):
        if transitions and i in transitions:
            # Large change — new pattern
            base = rng.randn(64, 64).astype(np.float32) * 0.5
            history.append(base.copy())
        else:
            # Tiny noise — barely different from previous
            noise = rng.randn(64, 64).astype(np.float32) * 0.001
            history.append(base + noise)

    return history


def make_uniform_history(n_frames: int) -> list[np.ndarray]:
    """Create a history where all frames are nearly identical."""
    rng = np.random.RandomState(99)
    base = rng.randn(64, 64).astype(np.float32) * 0.5
    return [base + rng.randn(64, 64).astype(np.float32) * 0.0001 for _ in range(n_frames)]


def store_test_memory(text: str, data_dir: Path, chunk: str = "general") -> str:
    """Helper to store a memory and return its hex key."""
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False)


def age_memory(data_dir: Path, chunk: str, hex_key: str, days: float) -> None:
    """Backdate a memory's timestamp and last_accessed by *days*."""
    from datetime import datetime, timedelta, timezone

    chunk_dir = data_dir / "chunks" / chunk
    index = _load_index(chunk_dir)
    if hex_key not in index:
        return
    past = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    index[hex_key]["timestamp"] = past
    index[hex_key]["metadata"]["last_accessed"] = past
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def main():
    global passes, fails

    # ==================================================================
    # Test 1: select_keyframes preserves seed + attractor
    # ==================================================================
    print("\n[1] select_keyframes preserves seed + attractor")

    # 20 frames with transitions at 5, 10, 15
    history = make_synthetic_history(20, transitions=[5, 10, 15])
    kept = select_keyframes(history, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

    check("seed (frame 0) kept", 0 in kept)
    check("attractor (frame 19) kept", 19 in kept)
    check("at least some transition frames kept", len(kept) >= 4,
          f"kept {len(kept)} frames: {kept}")
    check("kept < total", len(kept) < 20, f"kept {len(kept)}")

    # Verify transition frames are captured (at least some of 5, 10, 15)
    transitions_kept = [i for i in [5, 10, 15] if i in kept]
    check("transition frames captured", len(transitions_kept) >= 2,
          f"transitions kept: {transitions_kept}")

    # ==================================================================
    # Test 2: min_frames floor
    # ==================================================================
    print("\n[2] min_frames floor")

    uniform = make_uniform_history(20)
    kept_uniform = select_keyframes(uniform, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

    check(f"uniform history keeps >= {CONSOLIDATION_MIN_FRAMES} frames",
          len(kept_uniform) >= CONSOLIDATION_MIN_FRAMES,
          f"kept {len(kept_uniform)}")
    check("seed kept in uniform", 0 in kept_uniform)
    check("attractor kept in uniform", 19 in kept_uniform)

    # ==================================================================
    # Test 3: consolidate_brick metadata
    # ==================================================================
    print("\n[3] consolidate_brick metadata")

    history_3 = make_synthetic_history(30, transitions=[8, 16, 24])
    brick = MemoryBrick(
        evolution_history=history_3,
        final_attractor=history_3[-1],
        convergence_ticks=30,
        state="CONVERGED",
        metadata={"test": True},
    )

    consolidated = consolidate_brick(brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

    check("frame count reduced",
          len(consolidated.evolution_history) < len(brick.evolution_history),
          f"{len(consolidated.evolution_history)} vs {len(brick.evolution_history)}")
    check("metadata: consolidated=True", consolidated.metadata.get("consolidated") is True)
    check("metadata: consolidated_at set", "consolidated_at" in consolidated.metadata)
    check("metadata: original_frame_count",
          consolidated.metadata.get("original_frame_count") == 30,
          f"got {consolidated.metadata.get('original_frame_count')}")
    check("metadata: retained_frame_count",
          consolidated.metadata.get("retained_frame_count") == len(consolidated.evolution_history))
    check("metadata: frames_pruned",
          consolidated.metadata.get("frames_pruned") == 30 - len(consolidated.evolution_history))
    check("metadata: delta_threshold",
          consolidated.metadata.get("consolidation_delta_threshold") == CONSOLIDATION_DELTA_COLD)
    check("metadata: role_threshold",
          consolidated.metadata.get("consolidation_role_threshold") == CONSOLIDATION_ROLE_COLD)

    # Seed and attractor preserved
    check("seed frame preserved",
          np.array_equal(consolidated.evolution_history[0], brick.evolution_history[0]))
    check("attractor frame preserved",
          np.array_equal(consolidated.evolution_history[-1], brick.evolution_history[-1]))
    check("final_attractor unchanged",
          np.array_equal(consolidated.final_attractor, brick.final_attractor))
    check("original metadata preserved", consolidated.metadata.get("test") is True)

    # ==================================================================
    # Test 4: idempotent
    # ==================================================================
    print("\n[4] idempotent — consolidate twice")

    first = consolidate_brick(brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)
    second = consolidate_brick(first, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

    check("second consolidation returns same object",
          len(second.evolution_history) == len(first.evolution_history),
          f"first={len(first.evolution_history)}, second={len(second.evolution_history)}")
    check("no double-consolidation metadata change",
          second.metadata.get("original_frame_count") == first.metadata.get("original_frame_count"))

    # ==================================================================
    # Test 5: skips small history
    # ==================================================================
    print("\n[5] skips small history")

    small_history = make_synthetic_history(3, transitions=[1])
    small_brick = MemoryBrick(
        evolution_history=small_history,
        final_attractor=small_history[-1],
        convergence_ticks=3,
        state="CONVERGED",
        metadata={},
    )

    result_small = consolidate_brick(small_brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)
    check("small brick unchanged",
          len(result_small.evolution_history) == len(small_history),
          f"got {len(result_small.evolution_history)}")
    check("small brick not marked consolidated",
          not result_small.metadata.get("consolidated"))

    # ==================================================================
    # Test 6: temperature-tiered
    # ==================================================================
    print("\n[6] temperature-tiered thresholds")

    # Hot: skip
    hot_thresh = thresholds_for_temperature(0.8)
    check("hot returns None", hot_thresh is None)

    # Warm
    warm_thresh = thresholds_for_temperature(0.4)
    check("warm returns thresholds", warm_thresh is not None)
    if warm_thresh:
        check("warm delta threshold",
              warm_thresh[0] == CONSOLIDATION_DELTA_WARM,
              f"got {warm_thresh[0]}")
        check("warm role threshold",
              warm_thresh[1] == CONSOLIDATION_ROLE_WARM,
              f"got {warm_thresh[1]}")

    # Cold
    cold_thresh = thresholds_for_temperature(0.1)
    check("cold returns thresholds", cold_thresh is not None)
    if cold_thresh:
        check("cold delta threshold",
              cold_thresh[0] == CONSOLIDATION_DELTA_COLD,
              f"got {cold_thresh[0]}")
        check("cold role threshold",
              cold_thresh[1] == CONSOLIDATION_ROLE_COLD,
              f"got {cold_thresh[1]}")

    # Warm keeps more frames than cold
    history_6 = make_synthetic_history(40, transitions=[5, 10, 15, 20, 25, 30, 35])
    brick_6 = MemoryBrick(
        evolution_history=history_6,
        final_attractor=history_6[-1],
        convergence_ticks=40,
        state="CONVERGED",
        metadata={},
    )

    warm_consolidated = consolidate_brick(brick_6, CONSOLIDATION_DELTA_WARM, CONSOLIDATION_ROLE_WARM)
    cold_consolidated = consolidate_brick(brick_6, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

    warm_frames = len(warm_consolidated.evolution_history)
    cold_frames = len(cold_consolidated.evolution_history)
    check("warm keeps more frames than cold",
          warm_frames >= cold_frames,
          f"warm={warm_frames}, cold={cold_frames}")

    # ==================================================================
    # Test 7: dry_run
    # ==================================================================
    print("\n[7] dry_run")

    tmp7 = Path(tempfile.mkdtemp(prefix="wheeler_consolidation_dry_"))
    try:
        k = store_test_memory("consolidation dry run test", tmp7, chunk="general")
        # Age it to make it cold
        age_memory(tmp7, "general", k, 14)

        brick_path = tmp7 / "chunks" / "general" / "bricks" / f"{k}.npz"
        brick_before = MemoryBrick.load(brick_path)
        frames_before = len(brick_before.evolution_history)

        check("brick has enough frames for consolidation",
              frames_before >= CONSOLIDATION_MIN_HISTORY,
              f"got {frames_before}")

        result = sleep_consolidate(tmp7, dry_run=True)

        # Report should show savings
        if result.memories_consolidated:
            check("dry_run shows consolidation candidates",
                  len(result.memories_consolidated) > 0)

            # But brick should be unchanged on disk
            brick_after = MemoryBrick.load(brick_path)
            check("brick unchanged after dry_run",
                  len(brick_after.evolution_history) == frames_before,
                  f"before={frames_before}, after={len(brick_after.evolution_history)}")
            check("brick not marked consolidated after dry_run",
                  not brick_after.metadata.get("consolidated"))
        else:
            # The CA-generated brick might not have enough delta to consolidate.
            # In that case, verify it was skipped for "no_reduction" reason
            skip_reasons = [m["reason"] for m in result.memories_skipped]
            check("dry_run: brick skipped (no reduction or too few frames)",
                  "no_reduction" in skip_reasons or "too_few_frames" in skip_reasons,
                  f"skip reasons: {skip_reasons}")
            # Still verify brick is unchanged
            brick_after = MemoryBrick.load(brick_path)
            check("brick unchanged after dry_run (skipped)",
                  len(brick_after.evolution_history) == frames_before)
            check("brick not marked consolidated",
                  not brick_after.metadata.get("consolidated"))
    finally:
        shutil.rmtree(tmp7, ignore_errors=True)

    # ==================================================================
    # Test 8: re-save integrity
    # ==================================================================
    print("\n[8] re-save integrity")

    tmp8 = Path(tempfile.mkdtemp(prefix="wheeler_consolidation_resave_"))
    try:
        # Create a brick with known synthetic history that will definitely consolidate
        history_8 = make_synthetic_history(30, transitions=[8, 16, 24])
        brick_8 = MemoryBrick(
            evolution_history=history_8,
            final_attractor=history_8[-1],
            convergence_ticks=30,
            state="CONVERGED",
            metadata={"source": "test"},
        )

        brick_path = tmp8 / "test_brick.npz"
        brick_8.save(brick_path)

        # Consolidate and save
        consolidated = consolidate_brick(brick_8, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)
        consolidated.save(brick_path)

        # Reload and verify
        reloaded = MemoryBrick.load(brick_path)

        check("reloaded frame count matches",
              len(reloaded.evolution_history) == len(consolidated.evolution_history),
              f"expected {len(consolidated.evolution_history)}, got {len(reloaded.evolution_history)}")
        check("reloaded attractor matches",
              np.allclose(reloaded.final_attractor, brick_8.final_attractor))
        check("reloaded seed matches",
              np.allclose(reloaded.evolution_history[0], brick_8.evolution_history[0]))
        check("reloaded last frame matches original attractor",
              np.allclose(reloaded.evolution_history[-1], brick_8.evolution_history[-1]))
        check("reloaded metadata: consolidated",
              reloaded.metadata.get("consolidated") is True)
        check("reloaded metadata: source preserved",
              reloaded.metadata.get("source") == "test")
        check("reloaded state preserved",
              reloaded.state == "CONVERGED")
        check("reloaded convergence_ticks preserved",
              reloaded.convergence_ticks == 30)
    finally:
        shutil.rmtree(tmp8, ignore_errors=True)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  CONSOLIDATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Passed: {passes}")
    print(f"  Failed: {fails}")
    print(f"  Overall: {'PASS' if fails == 0 else 'FAIL'}")
    print(f"{'='*60}")

    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
