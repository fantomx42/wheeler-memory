"""Unit tests for dual-polarity encoding helpers."""
import pytest
from wheeler_memory.polarity import (
    POLAR_WEIGHT_DECAY, POLAR_DECAY_THRESHOLD, is_neutralized, polar_weight,
)


class TestPolarWeight:
    """Tests for polar_weight function."""

    def test_polar_weight_zero_decays(self):
        """{'decay_count': 0} → 1.0."""
        weight = polar_weight({"decay_count": 0})
        assert abs(weight - 1.0) < 1e-10

    def test_polar_weight_one_decay(self):
        """{'decay_count': 1} → 0.7."""
        weight = polar_weight({"decay_count": 1})
        assert abs(weight - POLAR_WEIGHT_DECAY) < 1e-10

    def test_polar_weight_three_decays(self):
        """{'decay_count': 3} → 0.7^3 ≈ 0.343."""
        weight = polar_weight({"decay_count": 3})
        expected = POLAR_WEIGHT_DECAY ** 3
        assert abs(weight - expected) < 1e-10

    def test_polar_weight_legacy_field(self):
        """{'safe_recall_count': 2} → 0.7^2 (backward compat)."""
        weight = polar_weight({"safe_recall_count": 2})
        expected = POLAR_WEIGHT_DECAY ** 2
        assert abs(weight - expected) < 1e-10

    def test_polar_weight_no_count(self):
        """missing decay_count → defaults to 0."""
        weight = polar_weight({})
        assert abs(weight - 1.0) < 1e-10


class TestIsNeutralized:
    """Tests for is_neutralized function."""

    def test_is_neutralized_fresh(self):
        """decay_count=0 → False (weight=1.0 > threshold)."""
        result = is_neutralized({"decay_count": 0})
        assert result is False

    def test_is_neutralized_after_many_decays(self):
        """decay_count=10 → True (0.7^10 ≈ 0.028 < 0.1)."""
        result = is_neutralized({"decay_count": 10})
        assert result is True

    def test_polar_weight_decays_below_threshold(self):
        """find N where 0.7^N < 0.1 (N=7 → 0.082)."""
        # 0.7^6 ≈ 0.118 (still above threshold)
        # 0.7^7 ≈ 0.083 (below threshold)
        assert is_neutralized({"decay_count": 6}) is False
        assert is_neutralized({"decay_count": 7}) is True
