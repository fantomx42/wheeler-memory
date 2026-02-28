"""Integration tests for associative warming."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wheeler_memory.temperature import (
    MAX_WARMTH,
    WARMTH_FLOOR,
    WARMTH_HOP1,
    WARMTH_HOP2,
    compute_temperature,
    compute_warmth,
    effective_temperature,
)
from wheeler_memory.warming import (
    build_co_recall_associations,
    load_associations,
    load_warmth,
    propagate_warmth,
)
from wheeler_memory.storage import list_memories, recall_memory

# Import conftest helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import store_test_memory


class TestComputeWarmthDecay:
    """Test pure compute_warmth decay (no disk I/O)."""

    def test_warmth_at_t0(self):
        """compute_warmth at t=0 equals boost."""
        now = datetime.now(timezone.utc)
        w0 = compute_warmth(0.05, now.isoformat(), now=now)

        assert abs(w0 - 0.05) < 1e-4

    def test_warmth_at_t1d(self):
        """compute_warmth after 1 day (half-life) is ~0.025."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=1)
        w1 = compute_warmth(0.05, past.isoformat(), now=now)

        assert abs(w1 - 0.025) < 1e-4

    def test_warmth_at_t3d(self):
        """compute_warmth after 3 days is ~0.00625."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=3)
        w3 = compute_warmth(0.05, past.isoformat(), now=now)

        assert abs(w3 - 0.00625) < 1e-4

    def test_warmth_at_t10d(self):
        """compute_warmth after 10 days (below floor) is 0.0."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=10)
        w10 = compute_warmth(0.05, past.isoformat(), now=now)

        assert w10 == 0.0


class TestEffectiveTemperature:
    """Test pure effective_temperature computation (no disk I/O)."""

    def test_effective_equals_base_plus_warmth(self):
        """effective_temperature matches base + warmth."""
        now = datetime.now(timezone.utc)
        base = compute_temperature(5, now.isoformat(), now=now)
        eff = effective_temperature(
            5,
            now.isoformat(),
            warmth_boost=0.05,
            warmth_applied_at=now.isoformat(),
            now=now,
        )

        assert abs(eff - (base + 0.05)) < 1e-4

    def test_effective_capped_at_1(self):
        """effective_temperature is capped at 1.0."""
        now = datetime.now(timezone.utc)
        eff_cap = effective_temperature(
            10,
            now.isoformat(),
            warmth_boost=0.15,
            warmth_applied_at=now.isoformat(),
            now=now,
        )

        assert eff_cap <= 1.0

    def test_effective_no_warmth_equals_base(self):
        """effective_temperature with no warmth equals base temp."""
        now = datetime.now(timezone.utc)
        base = compute_temperature(5, now.isoformat(), now=now)
        eff_none = effective_temperature(5, now.isoformat(), now=now)

        assert abs(eff_none - base) < 1e-4


class TestStoreTimeAssociations:
    """Test associations formed at store time (disk I/O)."""

    def test_store_time_edges_sha256_mode(self, tmp_path):
        """Store 5 memories in SHA-256 mode → 0 store-time edges (expected)."""
        texts = [
            "fix the python debug error in the login module",
            "debug the python authentication bug",
            "buy groceries milk eggs bread",
            "quantum entanglement violates Bell inequalities",
            "resolve python import error in tests",
        ]

        for text in texts:
            store_test_memory(text, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)
        edges = assoc.get("edges", {})
        total_edges = sum(len(v) for v in edges.values()) // 2

        # SHA-256 attractors have near-zero correlation, so no edges expected
        assert total_edges == 0

    def test_edges_are_bidirectional(self, tmp_path):
        """All edges in association graph are bidirectional."""
        texts = [
            "memory one",
            "memory two",
            "memory three",
        ]

        for text in texts:
            store_test_memory(text, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)
        edges = assoc.get("edges", {})

        for a, neighbors in edges.items():
            for b in neighbors:
                assert a in edges.get(b, {}), f"Edge {a}-{b} is not bidirectional"


class TestCoRecallAssociations:
    """Test co-recall associations formed during recall."""

    def test_co_recall_may_add_edges(self, tmp_path):
        """After recall returning multiple results, edges may increase."""
        texts = [
            "memory association one",
            "memory association two",
            "memory association three",
            "memory association four",
        ]

        for text in texts:
            store_test_memory(text, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        edges_before = sum(
            len(v) for v in load_associations(chunk_dir).get("edges", {}).values()
        ) // 2

        # Recall should form co-recall edges
        results = recall_memory("memory association", top_k=3, data_dir=tmp_path)

        edges_after = sum(
            len(v) for v in load_associations(chunk_dir).get("edges", {}).values()
        ) // 2

        assert len(results) > 0
        assert edges_after >= edges_before


class TestWarmthPropagation:
    """Test warmth propagation to neighbors (disk I/O)."""

    def test_propagate_warmth_forms_warmth(self, tmp_path):
        """Create edge manually, propagate → warmed dict non-empty."""
        text1 = "test warmth 1"
        text2 = "test warmth 2"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)

        # Manually create an edge
        assoc.setdefault("edges", {}).setdefault(key1, {})[key2] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        assoc.setdefault("edges", {}).setdefault(key2, {})[key1] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        # Propagate warmth from key1
        warmed = propagate_warmth(chunk_dir, [key1])

        assert len(warmed) > 0

    def test_hop1_boost_value(self, tmp_path):
        """Direct neighbor receives WARMTH_HOP1 boost."""
        text1 = "test hop1 1"
        text2 = "test hop1 2"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)

        # Create bidirectional edge
        assoc.setdefault("edges", {}).setdefault(key1, {})[key2] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        assoc.setdefault("edges", {}).setdefault(key2, {})[key1] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        warmed = propagate_warmth(chunk_dir, [key1])

        assert key2 in warmed
        assert abs(warmed[key2] - WARMTH_HOP1) < 1e-4

    def test_warmth_persisted(self, tmp_path):
        """After propagate_warmth, load_warmth returns non-empty dict."""
        text1 = "test persist 1"
        text2 = "test persist 2"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)

        # Create edge
        assoc.setdefault("edges", {}).setdefault(key1, {})[key2] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        assoc.setdefault("edges", {}).setdefault(key2, {})[key1] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        propagate_warmth(chunk_dir, [key1])
        warmth_on_disk = load_warmth(chunk_dir)

        assert len(warmth_on_disk) > 0


class TestWarmthInListMemories:
    """Test that warmed memories show elevated temperature in list_memories."""

    def test_warmed_memory_temp_above_base(self, tmp_path):
        """After warmth propagation, effective temp > base temp for warmed key."""
        text1 = "test list warmth 1"
        text2 = "test list warmth 2"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)

        # Create edge
        assoc.setdefault("edges", {}).setdefault(key1, {})[key2] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        assoc.setdefault("edges", {}).setdefault(key2, {})[key1] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        propagate_warmth(chunk_dir, [key1])
        memories = list_memories(data_dir=tmp_path)

        # Find the warmed memory (key2)
        warmed_mem = next((m for m in memories if m["hex_key"] == key2), None)
        assert warmed_mem is not None

        # Compute base temperature (hit_count=0, just stored)
        base = compute_temperature(
            warmed_mem["metadata"]["hit_count"],
            warmed_mem["metadata"]["last_accessed"],
        )

        # Effective should be higher due to warmth
        assert warmed_mem["temperature"] > base


class TestWarmthCap:
    """Test that warmth accumulation is capped at MAX_WARMTH."""

    def test_warmth_cap(self, tmp_path):
        """After multiple propagations, warmth capped at MAX_WARMTH."""
        text1 = "test cap 1"
        text2 = "test cap 2"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)

        # Create edge
        assoc.setdefault("edges", {}).setdefault(key1, {})[key2] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        assoc.setdefault("edges", {}).setdefault(key2, {})[key1] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        # Propagate multiple times to accumulate warmth
        for _ in range(10):
            propagate_warmth(chunk_dir, [key1])

        warmth_data = load_warmth(chunk_dir)

        for hk, entry in warmth_data.items():
            assert entry["boost"] <= MAX_WARMTH + 1e-4


class TestFiredExclusion:
    """Test that fired memories don't get warmed by each other."""

    def test_fired_exclusion(self, tmp_path):
        """Propagating with a pair → neither member is warmed."""
        text1 = "test fired 1"
        text2 = "test fired 2"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        chunk_dir = tmp_path / "chunks" / "general"
        assoc = load_associations(chunk_dir)

        # Create edge
        assoc.setdefault("edges", {}).setdefault(key1, {})[key2] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        assoc.setdefault("edges", {}).setdefault(key2, {})[key1] = {
            "weight": 0.7,
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        # Fire both keys (they won't warm each other)
        warmed = propagate_warmth(chunk_dir, [key1, key2])

        assert key1 not in warmed
        assert key2 not in warmed
