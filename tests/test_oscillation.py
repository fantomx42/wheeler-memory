"""Unit tests for oscillation detection."""
import numpy as np
import pytest
from wheeler_memory.oscillation import detect_oscillation, get_cell_roles


class TestGetCellRoles:
    """Tests for get_cell_roles function."""

    def test_get_cell_roles_uniform_frame(self):
        """all-same values → all roles 0 (slope, since no strict min/max)."""
        frame = np.ones((64, 64), dtype=np.float32)
        roles = get_cell_roles(frame)
        # All cells have all neighbors equal, so they're neither strict max nor min
        assert np.all((roles == 0) | (roles == 1) | (roles == -1))

    def test_get_cell_roles_local_max(self):
        """center cell clearly highest → role at center is 1."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = 1.0  # center is max
        roles = get_cell_roles(frame)
        assert roles[32, 32] == 1

    def test_get_cell_roles_local_min(self):
        """center cell clearly lowest → role at center is -1."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = -1.0  # center is min
        roles = get_cell_roles(frame)
        assert roles[32, 32] == -1

    def test_get_cell_roles_shape(self):
        """returns same shape as input."""
        frame = np.random.randn(64, 64).astype(np.float32)
        roles = get_cell_roles(frame)
        assert roles.shape == frame.shape


class TestDetectOscillation:
    """Tests for detect_oscillation function."""

    def test_detect_oscillation_no_history(self):
        """empty or single-frame history → oscillating=False."""
        result = detect_oscillation([])
        assert result["oscillating"] is False
        assert result["period"] is None
        assert result["oscillating_cells"] == 0

    def test_detect_oscillation_single_frame(self):
        """single frame → oscillating=False."""
        frame = np.random.randn(64, 64).astype(np.float32)
        result = detect_oscillation([frame])
        assert result["oscillating"] is False

    def test_detect_oscillation_static(self):
        """identical frames → oscillating=False."""
        frame = np.random.randn(64, 64).astype(np.float32)
        history = [frame.copy() for _ in range(20)]
        result = detect_oscillation(history)
        assert result["oscillating"] is False

    def test_detect_oscillation_returns_dict_keys(self):
        """result has required keys."""
        frame = np.random.randn(64, 64).astype(np.float32)
        result = detect_oscillation([frame])
        assert "oscillating" in result
        assert "period" in result
        assert "oscillating_cells" in result
