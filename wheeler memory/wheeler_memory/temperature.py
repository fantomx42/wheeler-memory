"""Temperature dynamics for Wheeler Memory.

Pure computation module — no I/O. Memories have a temperature reflecting
how recently and frequently they're accessed:

    temp = base_from_hits * decay_from_time

    base_from_hits = min(1.0, 0.3 + 0.7 * (hit_count / HIT_SATURATION))
    decay_from_time = 2 ^ (-days_since_last_access / HALF_LIFE_DAYS)

Tiers: hot >= 0.6, warm >= 0.3, cold < 0.3
"""

from datetime import datetime, timezone

HALF_LIFE_DAYS = 7.0
HIT_SATURATION = 10
TIER_HOT = 0.6
TIER_WARM = 0.3
TIER_FADING = 0.05          # Below this → brick eligible for deletion
TIER_DEAD = 0.01            # Below this → full eviction
MAX_ATTRACTORS = 10_000     # Global capacity across all chunks
EVICTION_RATIO = 0.10       # Remove bottom 10% when over capacity
MIN_AGE_DAYS = 1.0          # Never evict memories younger than 1 day

# Associative warming constants
WARMTH_HALF_LIFE_DAYS = 1.0   # Fast decay — warmth is short-term priming
WARMTH_HOP1 = 0.05            # Boost for direct neighbors
WARMTH_HOP2 = 0.025           # Boost for neighbors-of-neighbors
MAX_WARMTH = 0.15             # Cap to prevent runaway accumulation
WARMTH_FLOOR = 0.001          # Below this, warmth is garbage-collected


def compute_temperature(
    hit_count: int,
    last_accessed: str | datetime,
    now: datetime | None = None,
) -> float:
    """Compute temperature from access count and recency.

    Args:
        hit_count: Number of times this memory has been recalled.
        last_accessed: ISO-8601 timestamp or datetime of last access.
        now: Current time (defaults to utcnow).

    Returns:
        Temperature in [0, 1].
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if isinstance(last_accessed, str):
        last_accessed = datetime.fromisoformat(last_accessed)

    days_since = max(0.0, (now - last_accessed).total_seconds() / 86400.0)

    base_from_hits = min(1.0, 0.3 + 0.7 * (hit_count / HIT_SATURATION))
    decay_from_time = 2.0 ** (-days_since / HALF_LIFE_DAYS)

    # Round to 4 decimals to avoid float noise at tier boundaries
    # (handles seconds-level gaps between store and immediate recall)
    return round(base_from_hits * decay_from_time, 4)


def temperature_tier(temp: float) -> str:
    """Classify temperature into hot / warm / cold."""
    if temp >= TIER_HOT:
        return "hot"
    if temp >= TIER_WARM:
        return "warm"
    return "cold"


def temperature_tier_detailed(temp: float) -> str:
    """Classify temperature into hot / warm / cold / fading / dead."""
    if temp >= TIER_HOT:
        return "hot"
    if temp >= TIER_WARM:
        return "warm"
    if temp >= TIER_FADING:
        return "cold"
    if temp >= TIER_DEAD:
        return "fading"
    return "dead"


def ensure_access_fields(entry: dict, creation_timestamp: str) -> dict:
    """Backfill hit_count and last_accessed on legacy entries.

    Mutates and returns *entry* so callers can chain.
    """
    meta = entry.setdefault("metadata", {})
    if "hit_count" not in meta:
        meta["hit_count"] = 0
    if "last_accessed" not in meta:
        meta["last_accessed"] = creation_timestamp
    return entry


def compute_warmth(
    boost: float,
    applied_at: str | datetime,
    now: datetime | None = None,
) -> float:
    """Compute decayed warmth boost.

    Warmth decays with a 1-day half-life. Returns 0.0 if decayed below floor.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if isinstance(applied_at, str):
        applied_at = datetime.fromisoformat(applied_at)
    days_since = max(0.0, (now - applied_at).total_seconds() / 86400.0)
    decayed = boost * 2.0 ** (-days_since / WARMTH_HALF_LIFE_DAYS)
    return round(decayed, 4) if decayed >= WARMTH_FLOOR else 0.0


def effective_temperature(
    hit_count: int,
    last_accessed: str | datetime,
    warmth_boost: float = 0.0,
    warmth_applied_at: str | datetime | None = None,
    now: datetime | None = None,
) -> float:
    """Compute temperature + decayed warmth, capped at 1.0."""
    temp = compute_temperature(hit_count, last_accessed, now=now)
    if warmth_boost > 0.0 and warmth_applied_at is not None:
        temp += compute_warmth(warmth_boost, warmth_applied_at, now=now)
    return min(temp, 1.0)


def bump_access(entry: dict) -> dict:
    """Increment hit_count and update last_accessed to now.

    Mutates and returns *entry*.
    """
    meta = entry.setdefault("metadata", {})
    meta["hit_count"] = meta.get("hit_count", 0) + 1
    meta["last_accessed"] = datetime.now(timezone.utc).isoformat()
    return entry
