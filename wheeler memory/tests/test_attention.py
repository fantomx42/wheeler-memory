"""Unit tests for the attention model (variable tick rates)."""
import pytest
import numpy as np
from wheeler_memory.attention import (
    AttentionBudget, compute_attention_budget, salience_from_label, salience_from_temperature,
)
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.temperature import (
    SALIENCE_DEFAULT,
    SALIENCE_MAX_ITERS_LOW, SALIENCE_MAX_ITERS_MED, SALIENCE_MAX_ITERS_HIGH,
    SALIENCE_THRESHOLD_LOW, SALIENCE_THRESHOLD_MED, SALIENCE_THRESHOLD_HIGH,
)


class TestBudgetAnchors:
    """Tests for budget computation at anchor points."""

    def test_budget_anchor_low(self):
        """salience=0.0 → max_iters=LOW, threshold=THRESHOLD_LOW."""
        b = compute_attention_budget(0.0)
        assert b.max_iters == SALIENCE_MAX_ITERS_LOW
        assert abs(b.stability_threshold - SALIENCE_THRESHOLD_LOW) < 1e-10

    def test_budget_anchor_med(self):
        """salience=0.5 → max_iters=MED, threshold=THRESHOLD_MED."""
        b = compute_attention_budget(0.5)
        assert b.max_iters == SALIENCE_MAX_ITERS_MED
        assert abs(b.stability_threshold - SALIENCE_THRESHOLD_MED) < 1e-10

    def test_budget_anchor_high(self):
        """salience=1.0 → max_iters=HIGH, threshold=THRESHOLD_HIGH."""
        b = compute_attention_budget(1.0)
        assert b.max_iters == SALIENCE_MAX_ITERS_HIGH
        assert abs(b.stability_threshold - SALIENCE_THRESHOLD_HIGH) < 1e-10


class TestBudgetMonotonicity:
    """Tests for monotonicity of budget values."""

    def test_budget_max_iters_nondecreasing(self):
        """saliences 0.0 to 1.0 in 0.1 steps → max_iters non-decreasing."""
        saliences = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        budgets = [compute_attention_budget(s) for s in saliences]
        for i in range(len(budgets) - 1):
            assert budgets[i].max_iters <= budgets[i + 1].max_iters

    def test_budget_threshold_nonincreasing(self):
        """saliences 0.0 to 1.0 in 0.1 steps → threshold non-increasing."""
        saliences = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        budgets = [compute_attention_budget(s) for s in saliences]
        for i in range(len(budgets) - 1):
            assert budgets[i].stability_threshold >= budgets[i + 1].stability_threshold


class TestBudgetClamping:
    """Tests for salience clamping."""

    def test_clamp_negative(self):
        """salience=-0.5 → salience=0.0, max_iters matches salience=0."""
        b_neg = compute_attention_budget(-0.5)
        assert b_neg.salience == 0.0
        b_zero = compute_attention_budget(0.0)
        assert b_neg.max_iters == b_zero.max_iters

    def test_clamp_over_one(self):
        """salience=1.5 → salience=1.0, max_iters matches salience=1."""
        b_over = compute_attention_budget(1.5)
        assert b_over.salience == 1.0
        b_one = compute_attention_budget(1.0)
        assert b_over.max_iters == b_one.max_iters


class TestLabelConversion:
    """Tests for label-to-salience conversion."""

    def test_label_low(self):
        """salience_from_label('low') == 0.2."""
        assert salience_from_label("low") == 0.2

    def test_label_medium(self):
        """salience_from_label('medium') == 0.5."""
        assert salience_from_label("medium") == 0.5

    def test_label_high(self):
        """salience_from_label('high') == 0.9."""
        assert salience_from_label("high") == 0.9

    def test_label_case_insensitive(self):
        """salience_from_label('HIGH') == 0.9."""
        assert salience_from_label("HIGH") == 0.9

    def test_label_unknown_returns_default(self):
        """salience_from_label('bogus') == SALIENCE_DEFAULT."""
        assert salience_from_label("bogus") == SALIENCE_DEFAULT

    def test_budget_label_property(self):
        """compute_attention_budget(...).label matches salience range."""
        assert compute_attention_budget(0.2).label == "low"
        assert compute_attention_budget(0.5).label == "medium"
        assert compute_attention_budget(0.9).label == "high"


class TestTemperatureToSalience:
    """Tests for temperature-to-salience conversion."""

    def test_salience_from_temperature_endpoints(self):
        """temp=0.0 → 0.1, temp=1.0 → 1.0."""
        assert abs(salience_from_temperature(0.0) - 0.1) < 1e-9
        assert abs(salience_from_temperature(1.0) - 1.0) < 1e-9

    def test_salience_from_temperature_monotonic(self):
        """temps 0.0..1.0 → monotonically non-decreasing."""
        temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        sal_values = [salience_from_temperature(t) for t in temps]
        for i in range(len(sal_values) - 1):
            assert sal_values[i] <= sal_values[i + 1]


class TestBackwardsCompat:
    """Tests for backwards compatibility with old defaults."""

    def test_backwards_compat_default_matches_explicit(self):
        """evolve with default params == evolve with max_iters=1000, threshold=1e-4."""
        frame = hash_to_frame("backwards compatibility test")
        result_default = evolve_and_interpret(frame.copy())
        result_explicit = evolve_and_interpret(frame.copy(), max_iters=1000, stability_threshold=1e-4)
        assert result_default["state"] == result_explicit["state"]
        assert result_default["convergence_ticks"] == result_explicit["convergence_ticks"]
        if result_default["state"] == "CONVERGED":
            delta = np.abs(result_default["attractor"] - result_explicit["attractor"]).max()
            assert delta < 1e-10

    def test_default_budget_values(self):
        """compute_attention_budget(SALIENCE_DEFAULT).max_iters == 1000, threshold == 1e-4."""
        b = compute_attention_budget(SALIENCE_DEFAULT)
        assert b.max_iters == 1000
        assert abs(b.stability_threshold - 1e-4) < 1e-12


class TestHighSalienceDepth:
    """Tests for high salience using more computation."""

    def test_high_salience_uses_more_ticks(self):
        """high budget >= ticks than low budget on same input."""
        frame = hash_to_frame("attention depth test")
        budget_low = compute_attention_budget(0.2)
        budget_high = compute_attention_budget(0.9)
        result_low = evolve_and_interpret(
            frame.copy(),
            max_iters=budget_low.max_iters,
            stability_threshold=budget_low.stability_threshold,
        )
        result_high = evolve_and_interpret(
            frame.copy(),
            max_iters=budget_high.max_iters,
            stability_threshold=budget_high.stability_threshold,
        )
        assert result_high["convergence_ticks"] >= result_low["convergence_ticks"]
