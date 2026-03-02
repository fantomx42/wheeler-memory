"""Integration tests for MemoryBrick serialization and access."""
import numpy as np
import pytest
from pathlib import Path
from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame


class TestMemoryBrickEvolutionResult:
    """Test MemoryBrick construction from evolution results."""

    def test_from_evolution_result_has_history(self):
        """Brick created from evolution result has non-empty evolution_history."""
        frame = hash_to_frame("test memory 1")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        assert brick.evolution_history is not None
        assert len(brick.evolution_history) > 0

    def test_from_evolution_result_final_attractor(self):
        """Brick's final_attractor matches evolution result."""
        frame = hash_to_frame("test memory 2")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        assert np.array_equal(brick.final_attractor, result["attractor"])

    def test_from_evolution_result_state(self):
        """Brick's state matches evolution result."""
        frame = hash_to_frame("test memory 3")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        assert brick.state == result["state"]

    def test_from_evolution_result_ticks(self):
        """Brick's convergence_ticks matches evolution result."""
        frame = hash_to_frame("test memory 4")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        assert brick.convergence_ticks == result["convergence_ticks"]


class TestMemoryBrickSerialization:
    """Test MemoryBrick save/load roundtrips."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save brick to npz, load back, frame counts match."""
        frame = hash_to_frame("test memory 5")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        npz_path = tmp_path / "test.npz"
        brick.save(npz_path)

        assert npz_path.exists()
        loaded = MemoryBrick.load(npz_path)

        assert len(loaded.evolution_history) == len(brick.evolution_history)

    def test_save_and_load_attractor_matches(self, tmp_path):
        """Loaded brick's final_attractor matches original."""
        frame = hash_to_frame("test memory 6")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        npz_path = tmp_path / "test.npz"
        brick.save(npz_path)
        loaded = MemoryBrick.load(npz_path)

        assert np.array_equal(loaded.final_attractor, brick.final_attractor)

    def test_save_and_load_seed_matches(self, tmp_path):
        """Loaded brick's first frame (seed) matches original."""
        frame = hash_to_frame("test memory 7")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        npz_path = tmp_path / "test.npz"
        brick.save(npz_path)
        loaded = MemoryBrick.load(npz_path)

        assert np.array_equal(loaded.evolution_history[0], brick.evolution_history[0])

    def test_save_and_load_state_matches(self, tmp_path):
        """Loaded brick's state matches original."""
        frame = hash_to_frame("test memory 8")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        npz_path = tmp_path / "test.npz"
        brick.save(npz_path)
        loaded = MemoryBrick.load(npz_path)

        assert loaded.state == brick.state


class TestMemoryBrickFrameAccess:
    """Test frame access via get_frame_at_tick."""

    def test_get_frame_at_tick_valid(self):
        """get_frame_at_tick(0) returns seed frame with correct shape."""
        frame = hash_to_frame("test memory 9")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        seed = brick.get_frame_at_tick(0)
        assert seed.shape == (64, 64)

    def test_get_frame_at_tick_negative(self):
        """get_frame_at_tick(-1) returns last frame."""
        frame = hash_to_frame("test memory 10")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        last = brick.get_frame_at_tick(-1)
        expected = brick.evolution_history[-1]
        assert np.array_equal(last, expected)

    def test_get_frame_at_tick_intermediate(self):
        """get_frame_at_tick accesses intermediate frames correctly."""
        frame = hash_to_frame("test memory 11")
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)

        if len(brick.evolution_history) > 2:
            mid_idx = len(brick.evolution_history) // 2
            mid = brick.get_frame_at_tick(mid_idx)
            expected = brick.evolution_history[mid_idx]
            assert np.array_equal(mid, expected)
