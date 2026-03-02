"""Unit tests for CA dynamics engine."""
import numpy as np
import pytest
from wheeler_memory.dynamics import apply_ca_dynamics, evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame


class TestApplyCADynamics:
    """Tests for apply_ca_dynamics function."""

    def test_apply_ca_dynamics_shape_preserved(self):
        """input 64x64 → output 64x64."""
        frame = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        output = apply_ca_dynamics(frame)
        assert output.shape == (64, 64)

    def test_apply_ca_dynamics_range_preserved(self):
        """output values in [-1, 1]."""
        frame = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        output = apply_ca_dynamics(frame)
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)

    def test_apply_ca_dynamics_local_max_increases(self):
        """center cell = 1, all neighbors = 0 → center is local max, pushed toward +1."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = 1.0  # center >= all neighbors → local max
        output = apply_ca_dynamics(frame)
        # Local max: delta = (1 - cell) * 0.35, center stays near 1.0 (small positive delta)
        assert output[32, 32] >= frame[32, 32]

    def test_apply_ca_dynamics_local_min_decreases(self):
        """center cell = -1, all neighbors = 0 → center is local min, pushed toward -1."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = -1.0  # center <= all neighbors → local min
        output = apply_ca_dynamics(frame)
        # Local min: delta = (-1 - cell) * 0.35, center stays near -1.0 (small negative delta)
        assert output[32, 32] <= frame[32, 32]


class TestEvolveAndInterpret:
    """Tests for evolve_and_interpret function."""

    def test_evolve_result_keys(self):
        """result dict has required keys."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert "state" in result
        assert "attractor" in result
        assert "convergence_ticks" in result
        assert "history" in result

    def test_evolve_attractor_shape(self):
        """result['attractor'] is 64x64."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert result["attractor"].shape == (64, 64)

    def test_evolve_attractor_range(self):
        """attractor values in [-1, 1]."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert np.all(result["attractor"] >= -1.0)
        assert np.all(result["attractor"] <= 1.0)

    def test_evolve_state_valid(self):
        """state is one of 'CONVERGED', 'OSCILLATING', 'CHAOTIC'."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert result["state"] in ("CONVERGED", "OSCILLATING", "CHAOTIC")

    def test_evolve_convergence_ticks_positive(self):
        """convergence_ticks >= 1."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert result["convergence_ticks"] >= 1

    def test_evolve_convergence_ticks_respects_max_iters(self):
        """convergence_ticks <= max_iters."""
        frame = hash_to_frame("test")
        max_iters = 50
        result = evolve_and_interpret(frame, max_iters=max_iters, stability_threshold=1e-4)
        assert result["convergence_ticks"] <= max_iters

    def test_evolve_deterministic(self):
        """same input → same state and same attractor."""
        frame = hash_to_frame("deterministic test")
        result1 = evolve_and_interpret(frame.copy(), max_iters=100, stability_threshold=1e-4)
        result2 = evolve_and_interpret(frame.copy(), max_iters=100, stability_threshold=1e-4)
        assert result1["state"] == result2["state"]
        assert np.array_equal(result1["attractor"], result2["attractor"])

    def test_evolve_typical_input_converges(self):
        """hash_to_frame('test input') → state == 'CONVERGED'."""
        frame = hash_to_frame("test input")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        assert result["state"] == "CONVERGED"

    def test_evolve_history_nonempty(self):
        """history contains at least seed and attractor frames."""
        frame = hash_to_frame("history test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        # GPU path stores [seed, attractor]; CPU path stores every tick.
        # Either way history has at least 2 entries.
        assert len(result["history"]) >= 2

    def test_evolve_first_frame_is_input(self):
        """first frame in history matches input."""
        frame = hash_to_frame("first frame test")
        result = evolve_and_interpret(frame.copy(), max_iters=100, stability_threshold=1e-4)
        assert np.array_equal(result["history"][0], frame)
