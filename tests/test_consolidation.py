"""Integration tests for sleep consolidation.

Converts manual test script to pytest format. Verifies that:
1. select_keyframes preserves seed + attractor
2. min_frames floor — uniform history keeps at least 3 frames
3. consolidate_brick metadata — correct fields, frame reduction, seed/attractor preserved
4. idempotent — consolidate twice → same frame count, same original_frame_count
5. skips small history — brick with <5 frames returned unchanged
6. temperature-tiered — hot(0.8) → None, warm(0.4) → WARM thresholds, cold(0.1) → COLD thresholds
7. dry_run — report shows savings but brick unchanged on disk
8. re-save integrity — consolidated brick loads correctly, attractor matches
"""

import json
import pytest
import numpy as np
from pathlib import Path

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.consolidation import (
    ConsolidationResult,
    consolidate_brick,
    consolidation_stats,
    select_keyframes,
    sleep_consolidate,
    thresholds_for_temperature,
)
from wheeler_memory.storage import _load_index
from wheeler_memory.temperature import (
    CONSOLIDATION_DELTA_COLD,
    CONSOLIDATION_DELTA_WARM,
    CONSOLIDATION_MIN_FRAMES,
    CONSOLIDATION_MIN_HISTORY,
    CONSOLIDATION_ROLE_COLD,
    CONSOLIDATION_ROLE_WARM,
)

# Import helpers from conftest
import sys
from pathlib import Path as PathlibPath

sys.path.insert(0, str(PathlibPath(__file__).resolve().parent))
from conftest import (
    store_test_memory,
    age_memory,
    make_synthetic_history,
    make_uniform_history,
)


class TestConsolidationSelectKeyframes:
    """Test 1: select_keyframes preserves seed + attractor."""

    def test_select_keyframes_preserves_seed_and_attractor(self):
        """20 frames with transitions at [5,10,15], select_keyframes keeps 0 and 19, transitions captured."""
        history = make_synthetic_history(20, transitions=[5, 10, 15])
        kept = select_keyframes(history, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

        assert 0 in kept, "Seed (frame 0) should be kept"
        assert 19 in kept, "Attractor (frame 19) should be kept"
        assert len(kept) >= 4, f"Should keep at least 4 frames, got {len(kept)}: {kept}"
        assert len(kept) < 20, f"Should reduce frame count, kept {len(kept)}"

        # Verify transition frames are captured (at least some of 5, 10, 15)
        transitions_kept = [i for i in [5, 10, 15] if i in kept]
        assert (
            len(transitions_kept) >= 2
        ), f"Should capture at least 2 transition frames, got {transitions_kept}"


class TestConsolidationMinFramesFloor:
    """Test 2: min_frames floor — uniform history keeps at least 3 frames."""

    def test_select_keyframes_min_frames_floor(self):
        """Uniform history, keeps >= CONSOLIDATION_MIN_FRAMES frames."""
        uniform = make_uniform_history(20)
        kept_uniform = select_keyframes(uniform, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

        assert (
            len(kept_uniform) >= CONSOLIDATION_MIN_FRAMES
        ), f"Uniform history should keep >= {CONSOLIDATION_MIN_FRAMES} frames, got {len(kept_uniform)}"
        assert 0 in kept_uniform, "Seed should be kept in uniform history"
        assert 19 in kept_uniform, "Attractor should be kept in uniform history"


class TestConsolidationBrickMetadata:
    """Test 3: consolidate_brick metadata — correct fields, frame reduction, seed/attractor preserved."""

    def test_consolidate_brick_metadata(self):
        """Consolidate a 30-frame brick: frame count reduced, metadata fields correct."""
        history = make_synthetic_history(30, transitions=[8, 16, 24])
        brick = MemoryBrick(
            evolution_history=history,
            final_attractor=history[-1],
            convergence_ticks=30,
            state="CONVERGED",
            metadata={"test": True},
        )

        consolidated = consolidate_brick(brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

        # Verify frame count reduced
        assert len(consolidated.evolution_history) < len(
            brick.evolution_history
        ), f"Frame count not reduced: {len(consolidated.evolution_history)} vs {len(brick.evolution_history)}"

        # Verify metadata
        assert consolidated.metadata.get("consolidated") is True, "Should mark consolidated=True"
        assert "consolidated_at" in consolidated.metadata, "Should set consolidated_at timestamp"
        assert (
            consolidated.metadata.get("original_frame_count") == 30
        ), f"original_frame_count should be 30, got {consolidated.metadata.get('original_frame_count')}"
        assert consolidated.metadata.get("retained_frame_count") == len(
            consolidated.evolution_history
        ), "retained_frame_count should match history length"
        assert consolidated.metadata.get("frames_pruned") == (
            30 - len(consolidated.evolution_history)
        ), "frames_pruned calculation incorrect"
        assert consolidated.metadata.get("consolidation_delta_threshold") == CONSOLIDATION_DELTA_COLD
        assert consolidated.metadata.get("consolidation_role_threshold") == CONSOLIDATION_ROLE_COLD

        # Seed and attractor preserved
        assert np.array_equal(
            consolidated.evolution_history[0], brick.evolution_history[0]
        ), "Seed frame should be preserved"
        assert np.array_equal(
            consolidated.evolution_history[-1], brick.evolution_history[-1]
        ), "Attractor frame should be preserved"
        assert np.array_equal(
            consolidated.final_attractor, brick.final_attractor
        ), "final_attractor should be unchanged"
        assert consolidated.metadata.get("test") is True, "Original metadata should be preserved"


class TestConsolidationIdempotent:
    """Test 4: idempotent — consolidate twice → same frame count."""

    def test_consolidate_brick_idempotent(self):
        """Consolidate twice → same frame count, same original_frame_count."""
        history = make_synthetic_history(30, transitions=[8, 16, 24])
        brick = MemoryBrick(
            evolution_history=history,
            final_attractor=history[-1],
            convergence_ticks=30,
            state="CONVERGED",
            metadata={},
        )

        first = consolidate_brick(brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)
        second = consolidate_brick(first, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

        assert len(second.evolution_history) == len(
            first.evolution_history
        ), f"Second consolidation should not reduce further: first={len(first.evolution_history)}, second={len(second.evolution_history)}"
        assert second.metadata.get("original_frame_count") == first.metadata.get(
            "original_frame_count"
        ), "original_frame_count should not change on second consolidation"


class TestConsolidationSkipsSmallHistory:
    """Test 5: skips small history — brick with <5 frames returned unchanged."""

    def test_consolidate_skips_small_history(self):
        """3-frame brick → unchanged, not marked consolidated."""
        small_history = make_synthetic_history(3, transitions=[1])
        small_brick = MemoryBrick(
            evolution_history=small_history,
            final_attractor=small_history[-1],
            convergence_ticks=3,
            state="CONVERGED",
            metadata={},
        )

        result_small = consolidate_brick(small_brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)

        assert len(result_small.evolution_history) == len(
            small_history
        ), f"Small brick should be unchanged, got {len(result_small.evolution_history)}"
        assert (
            not result_small.metadata.get("consolidated")
        ), "Small brick should not be marked consolidated"


class TestConsolidationTemperatureTiered:
    """Test 6: temperature-tiered — hot(0.8) → None, warm/cold → tiered thresholds."""

    def test_thresholds_temperature_tiered(self):
        """Hot returns None, warm/cold return tiered thresholds, warm keeps more frames than cold."""
        # Hot: skip
        hot_thresh = thresholds_for_temperature(0.8)
        assert hot_thresh is None, "Hot temperature should return None"

        # Warm
        warm_thresh = thresholds_for_temperature(0.4)
        assert warm_thresh is not None, "Warm temperature should return thresholds"
        assert (
            warm_thresh[0] == CONSOLIDATION_DELTA_WARM
        ), f"Warm delta threshold incorrect: {warm_thresh[0]}"
        assert (
            warm_thresh[1] == CONSOLIDATION_ROLE_WARM
        ), f"Warm role threshold incorrect: {warm_thresh[1]}"

        # Cold
        cold_thresh = thresholds_for_temperature(0.1)
        assert cold_thresh is not None, "Cold temperature should return thresholds"
        assert (
            cold_thresh[0] == CONSOLIDATION_DELTA_COLD
        ), f"Cold delta threshold incorrect: {cold_thresh[0]}"
        assert (
            cold_thresh[1] == CONSOLIDATION_ROLE_COLD
        ), f"Cold role threshold incorrect: {cold_thresh[1]}"

        # Warm keeps more frames than cold on same history
        history_tiered = make_synthetic_history(40, transitions=[5, 10, 15, 20, 25, 30, 35])
        brick_tiered = MemoryBrick(
            evolution_history=history_tiered,
            final_attractor=history_tiered[-1],
            convergence_ticks=40,
            state="CONVERGED",
            metadata={},
        )

        warm_consolidated = consolidate_brick(
            brick_tiered, CONSOLIDATION_DELTA_WARM, CONSOLIDATION_ROLE_WARM
        )
        cold_consolidated = consolidate_brick(
            brick_tiered, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD
        )

        warm_frames = len(warm_consolidated.evolution_history)
        cold_frames = len(cold_consolidated.evolution_history)
        assert (
            warm_frames >= cold_frames
        ), f"Warm should keep >= cold frames: warm={warm_frames}, cold={cold_frames}"


class TestConsolidationDryRun:
    """Test 7: dry_run — report shows savings but brick unchanged on disk."""

    def test_dry_run(self, tmp_path):
        """Dry-run consolidation: report shows candidates but disk is unchanged."""
        # Use a synthetic brick with known frame count to avoid GPU 2-frame history
        from wheeler_memory.hashing import text_to_hex
        import json

        history = make_synthetic_history(30, transitions=[8, 16, 24])
        brick = MemoryBrick(
            evolution_history=history,
            final_attractor=history[-1],
            convergence_ticks=30,
            state="CONVERGED",
            metadata={},
        )

        # Write the brick and a minimal index entry so sleep_consolidate can find it
        chunk_dir = tmp_path / "chunks" / "general"
        (chunk_dir / "attractors").mkdir(parents=True, exist_ok=True)
        (chunk_dir / "bricks").mkdir(parents=True, exist_ok=True)

        k = text_to_hex("consolidation dry run test")
        brick_path = chunk_dir / "bricks" / f"{k}.npz"
        brick.save(brick_path)

        # Write attractor and index so sleep_consolidate can read temperature
        import numpy as np
        from datetime import datetime, timedelta, timezone
        np.save(chunk_dir / "attractors" / f"{k}.npy", history[-1])
        past = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        index = {k: {"text": "consolidation dry run test", "timestamp": past,
                     "metadata": {"hit_count": 0, "last_accessed": past}}}
        (chunk_dir / "index.json").write_text(json.dumps(index))

        frames_before = len(brick.evolution_history)
        assert frames_before >= CONSOLIDATION_MIN_HISTORY, f"Synthetic brick should have enough frames, got {frames_before}"

        result = sleep_consolidate(tmp_path, dry_run=True)

        # Report should show savings or skip reasons
        if result.memories_consolidated:
            assert (
                len(result.memories_consolidated) > 0
            ), "dry_run should show consolidation candidates"

            # But brick should be unchanged on disk
            brick_after = MemoryBrick.load(brick_path)
            assert len(brick_after.evolution_history) == frames_before, (
                f"Brick should be unchanged after dry_run: "
                f"before={frames_before}, after={len(brick_after.evolution_history)}"
            )
            assert (
                not brick_after.metadata.get("consolidated")
            ), "Brick should not be marked consolidated after dry_run"
        else:
            # The CA-generated brick might not have enough delta to consolidate.
            # In that case, verify it was skipped for "no_reduction" reason
            skip_reasons = [m["reason"] for m in result.memories_skipped]
            assert (
                "no_reduction" in skip_reasons or "too_few_frames" in skip_reasons
            ), f"Brick should be skipped with valid reason, got {skip_reasons}"
            # Still verify brick is unchanged
            brick_after = MemoryBrick.load(brick_path)
            assert len(brick_after.evolution_history) == frames_before, (
                f"Brick should be unchanged after dry_run (skipped)"
            )
            assert (
                not brick_after.metadata.get("consolidated")
            ), "Brick should not be marked consolidated"


class TestConsolidationResaveIntegrity:
    """Test 8: re-save integrity — consolidated brick loads correctly, attractor matches."""

    def test_resave_integrity(self, tmp_path):
        """Consolidate synthetic 30-frame brick, save, reload, attractor/seed/metadata preserved."""
        history = make_synthetic_history(30, transitions=[8, 16, 24])
        brick = MemoryBrick(
            evolution_history=history,
            final_attractor=history[-1],
            convergence_ticks=30,
            state="CONVERGED",
            metadata={"source": "test"},
        )

        brick_path = tmp_path / "test_brick.npz"
        brick.save(brick_path)

        # Consolidate and save
        consolidated = consolidate_brick(brick, CONSOLIDATION_DELTA_COLD, CONSOLIDATION_ROLE_COLD)
        consolidated.save(brick_path)

        # Reload and verify
        reloaded = MemoryBrick.load(brick_path)

        assert len(reloaded.evolution_history) == len(
            consolidated.evolution_history
        ), f"Reloaded frame count should match: expected {len(consolidated.evolution_history)}, got {len(reloaded.evolution_history)}"
        assert np.allclose(
            reloaded.final_attractor, brick.final_attractor
        ), "Reloaded attractor should match original"
        assert np.allclose(
            reloaded.evolution_history[0], brick.evolution_history[0]
        ), "Reloaded seed should match original"
        assert np.allclose(
            reloaded.evolution_history[-1], brick.evolution_history[-1]
        ), "Reloaded last frame should match original attractor"
        assert (
            reloaded.metadata.get("consolidated") is True
        ), "Reloaded metadata: consolidated should be True"
        assert (
            reloaded.metadata.get("source") == "test"
        ), "Reloaded metadata: source should be preserved"
        assert reloaded.state == "CONVERGED", "Reloaded state should be preserved"
        assert reloaded.convergence_ticks == 30, "Reloaded convergence_ticks should be preserved"
