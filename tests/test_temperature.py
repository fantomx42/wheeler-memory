"""Unit tests for temperature computation."""
from datetime import datetime, timedelta, timezone
import pytest
from wheeler_memory.temperature import (
    HALF_LIFE_DAYS, HIT_SATURATION, TIER_HOT, TIER_WARM,
    WARMTH_FLOOR, WARMTH_HALF_LIFE_DAYS,
    compute_temperature, compute_warmth, effective_temperature, temperature_tier,
)


class TestComputeTemperature:
    """Tests for compute_temperature function."""

    def test_compute_temperature_zero_hits(self):
        """0 hits, just accessed: base=0.3, decay≈1, result≈0.3."""
        now = datetime.now(timezone.utc)
        temp = compute_temperature(0, now, now=now)
        assert abs(temp - 0.3) < 0.01

    def test_compute_temperature_saturated_hits(self):
        """10 hits (saturated), just accessed: base=1.0."""
        now = datetime.now(timezone.utc)
        temp = compute_temperature(HIT_SATURATION, now, now=now)
        assert abs(temp - 1.0) < 0.01

    def test_compute_temperature_half_life(self):
        """10 hits, 7 days ago (half-life): result ≈ 0.5."""
        now = datetime.now(timezone.utc)
        seven_days_ago = now - timedelta(days=7)
        temp = compute_temperature(HIT_SATURATION, seven_days_ago, now=now)
        # base = 1.0, decay ≈ 0.5 (at half-life)
        assert abs(temp - 0.5) < 0.01

    def test_compute_temperature_two_weeks(self):
        """0 hits, 14 days ago: result ≈ 0.3 * 0.25."""
        now = datetime.now(timezone.utc)
        two_weeks_ago = now - timedelta(days=14)
        temp = compute_temperature(0, two_weeks_ago, now=now)
        # base = 0.3, decay = 2^(-14/7) = 0.25
        expected = 0.3 * 0.25
        assert abs(temp - expected) < 0.01

    def test_compute_temperature_accepts_datetime(self):
        """Accepts datetime object, not just ISO string."""
        now = datetime.now(timezone.utc)
        temp = compute_temperature(0, now, now=now)
        assert isinstance(temp, float)
        assert 0 <= temp <= 1.0

    def test_compute_temperature_accepts_iso_string(self):
        """Accepts ISO-8601 string."""
        now = datetime.now(timezone.utc)
        temp = compute_temperature(0, now.isoformat(), now=now)
        assert isinstance(temp, float)
        assert 0 <= temp <= 1.0


class TestTemperatureTier:
    """Tests for temperature_tier function."""

    def test_temperature_tier_hot(self):
        """0.8 → 'hot'."""
        assert temperature_tier(0.8) == "hot"

    def test_temperature_tier_warm(self):
        """0.4 → 'warm'."""
        assert temperature_tier(0.4) == "warm"

    def test_temperature_tier_cold(self):
        """0.1 → 'cold'."""
        assert temperature_tier(0.1) == "cold"

    def test_temperature_tier_boundary_hot(self):
        """Exactly 0.6 (TIER_HOT) → 'hot'."""
        assert temperature_tier(TIER_HOT) == "hot"

    def test_temperature_tier_boundary_warm(self):
        """Exactly 0.3 (TIER_WARM) → 'warm'."""
        assert temperature_tier(TIER_WARM) == "warm"


class TestComputeWarmth:
    """Tests for compute_warmth function."""

    def test_compute_warmth_at_t0(self):
        """boost=0.05, just applied → ≈0.05."""
        now = datetime.now(timezone.utc)
        warmth = compute_warmth(0.05, now, now=now)
        assert abs(warmth - 0.05) < 0.01

    def test_compute_warmth_at_t1d(self):
        """boost=0.05, 1 day ago → ≈0.025 (half-life 1 day)."""
        now = datetime.now(timezone.utc)
        one_day_ago = now - timedelta(days=1)
        warmth = compute_warmth(0.05, one_day_ago, now=now)
        # At half-life, value should be 0.5 * boost
        assert abs(warmth - 0.025) < 0.01

    def test_compute_warmth_at_t10d(self):
        """boost=0.05, 10 days ago → 0.0 (below WARMTH_FLOOR)."""
        now = datetime.now(timezone.utc)
        ten_days_ago = now - timedelta(days=10)
        warmth = compute_warmth(0.05, ten_days_ago, now=now)
        # After 10 days with 1-day half-life, value is 0.05 * 2^(-10) ≈ 0.00004
        # This is below WARMTH_FLOOR, so should return 0.0
        assert warmth == 0.0

    def test_compute_warmth_accepts_iso_string(self):
        """Accepts ISO string for applied_at."""
        now = datetime.now(timezone.utc)
        warmth = compute_warmth(0.05, now.isoformat(), now=now)
        assert abs(warmth - 0.05) < 0.01


class TestEffectiveTemperature:
    """Tests for effective_temperature function."""

    def test_effective_temperature_adds_warmth(self):
        """base + warmth = effective."""
        now = datetime.now(timezone.utc)
        base_temp = compute_temperature(0, now, now=now)
        warmth_boost = 0.05
        eff_temp = effective_temperature(0, now, warmth_boost=warmth_boost,
                                        warmth_applied_at=now, now=now)
        expected = base_temp + warmth_boost
        assert abs(eff_temp - expected) < 0.01

    def test_effective_temperature_capped_at_1(self):
        """high hits + high warmth → capped at 1.0."""
        now = datetime.now(timezone.utc)
        eff_temp = effective_temperature(
            HIT_SATURATION, now,
            warmth_boost=0.5,
            warmth_applied_at=now,
            now=now
        )
        assert eff_temp == 1.0

    def test_effective_temperature_no_warmth(self):
        """no warmth_boost → equals base temp."""
        now = datetime.now(timezone.utc)
        base_temp = compute_temperature(0, now, now=now)
        eff_temp = effective_temperature(0, now, now=now)
        assert abs(eff_temp - base_temp) < 0.01
