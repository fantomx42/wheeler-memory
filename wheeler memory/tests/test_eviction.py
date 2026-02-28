"""Integration tests for memory eviction and forgetting.

Converts manual test script to pytest format. Stores memories, simulates aging,
and verifies that:
1. score_memories ordering — coldest first
2. fade_cold_memories — brick deleted, attractor + index remain
3. evict_dead_memories — all artifacts removed
4. Association cleanup — evicted memory's edges removed, others preserved
5. forget_memory targeted — immediate removal
6. Hot/warm protection — recalled memories survive eviction
7. MIN_AGE_DAYS protection — brand-new cold memory not evicted
8. dry_run — report is correct but nothing deleted
9. Capacity eviction — monkeypatch MAX_ATTRACTORS=5, store 8, coldest evicted
"""

import json
import pytest
from pathlib import Path
from unittest import mock

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.eviction import (
    EvictionResult,
    evict_dead_memories,
    evict_for_capacity,
    fade_cold_memories,
    forget_memory,
    score_memories,
    sweep_and_evict,
)
from wheeler_memory.hashing import text_to_hex
from wheeler_memory.storage import _load_index, recall_memory
from wheeler_memory.temperature import TIER_DEAD, TIER_FADING
from wheeler_memory.warming import build_co_recall_associations, get_neighbors

# Import helpers from conftest
import sys
from pathlib import Path as PathlibPath

sys.path.insert(0, str(PathlibPath(__file__).resolve().parent))
from conftest import store_test_memory, age_memory


class TestEvictionScoreMemories:
    """Test 1: score_memories ordering — coldest first."""

    def test_score_memories_ordering(self, tmp_path):
        """Store 5 memories with ages 0/7/14/30/60 days, verify coldest first."""
        texts_and_ages = [
            ("eviction test memory zero days", 0),
            ("eviction test memory seven days", 7),
            ("eviction test memory fourteen days", 14),
            ("eviction test memory thirty days", 30),
            ("eviction test memory sixty days", 60),
        ]
        keys = []
        for text, age in texts_and_ages:
            k = store_test_memory(text, tmp_path, chunk="general")
            if age > 0:
                age_memory(tmp_path, "general", k, age)
            keys.append(k)

        scored = score_memories(tmp_path)

        # Verify 5 entries stored
        assert len(scored) == 5, f"Expected 5 entries, got {len(scored)}"

        # Coldest first — the oldest (60 days) should be first
        assert (
            scored[0]["temperature"] <= scored[-1]["temperature"]
        ), f"Coldest first check failed: first={scored[0]['temperature']:.6f}, last={scored[-1]['temperature']:.6f}"

        # The 60-day-old should be the coldest
        sixty_day_key = keys[4]
        assert (
            scored[0]["hex_key"] == sixty_day_key
        ), f"60-day memory not coldest: expected {sixty_day_key[:8]}..., got {scored[0]['hex_key'][:8]}..."


class TestEvictionFadeColdMemories:
    """Test 2: fade_cold_memories — brick deleted, attractor + index remain."""

    def test_fade_deletes_brick_preserves_attractor(self, tmp_path):
        """30-day memory gets faded; brick gone, attractor+index remain, still recallable."""
        texts_and_ages = [
            ("eviction test memory zero days", 0),
            ("eviction test memory seven days", 7),
            ("eviction test memory fourteen days", 14),
            ("eviction test memory thirty days", 30),
            ("eviction test memory sixty days", 60),
        ]
        keys = []
        for text, age in texts_and_ages:
            k = store_test_memory(text, tmp_path, chunk="general")
            if age > 0:
                age_memory(tmp_path, "general", k, age)
            keys.append(k)

        # The 30-day-old should have temp < TIER_FADING
        scored = score_memories(tmp_path)
        thirty_key = keys[3]
        thirty_temp = next(m["temperature"] for m in scored if m["hex_key"] == thirty_key)
        assert thirty_temp < TIER_FADING, f"30-day memory temp not < TIER_FADING: {thirty_temp:.6f}"

        brick_path = tmp_path / "chunks" / "general" / "bricks" / f"{thirty_key}.npz"
        assert brick_path.exists(), "Brick should exist before fade"

        faded = fade_cold_memories(tmp_path)
        assert len(faded) > 0, f"Fade should return entries, got {len(faded)}"

        assert not brick_path.exists(), "Brick should be deleted after fade"

        # Attractor and index should remain
        att_path = tmp_path / "chunks" / "general" / "attractors" / f"{thirty_key}.npy"
        assert att_path.exists(), "Attractor should survive fade"

        index = _load_index(tmp_path / "chunks" / "general")
        assert thirty_key in index, "Index entry should survive fade"

        # Memory should still be recallable (attractor exists)
        results = recall_memory(
            "eviction test memory thirty days", top_k=5, data_dir=tmp_path, chunk="general"
        )
        found_keys = [r["hex_key"] for r in results]
        assert thirty_key in found_keys, "Faded memory should still be recallable"

        # Re-age memories that were bumped by the recall above
        for text, age in texts_and_ages:
            if age > 0:
                k = text_to_hex(text)
                age_memory(tmp_path, "general", k, age)


class TestEvictionEvictDeadMemories:
    """Test 3: evict_dead_memories — all artifacts removed."""

    def test_evict_dead_removes_all_artifacts(self, tmp_path):
        """60-day memory gets evicted; all files gone, not recallable."""
        texts_and_ages = [
            ("eviction test memory zero days", 0),
            ("eviction test memory seven days", 7),
            ("eviction test memory fourteen days", 14),
            ("eviction test memory thirty days", 30),
            ("eviction test memory sixty days", 60),
        ]
        keys = []
        for text, age in texts_and_ages:
            k = store_test_memory(text, tmp_path, chunk="general")
            if age > 0:
                age_memory(tmp_path, "general", k, age)
            keys.append(k)

        # Verify 60-day is dead
        scored = score_memories(tmp_path)
        sixty_key = keys[4]
        sixty_temp = next(m["temperature"] for m in scored if m["hex_key"] == sixty_key)
        assert sixty_temp < TIER_DEAD, f"60-day memory temp not < TIER_DEAD: {sixty_temp:.6f}"

        evicted = evict_dead_memories(tmp_path)
        assert len(evicted) > 0, f"Evict should return entries, got {len(evicted)}"

        # All artifacts should be gone
        att_path_60 = tmp_path / "chunks" / "general" / "attractors" / f"{sixty_key}.npy"
        brick_path_60 = tmp_path / "chunks" / "general" / "bricks" / f"{sixty_key}.npz"
        assert not att_path_60.exists(), "Attractor should be removed"
        assert not brick_path_60.exists(), "Brick should be removed"

        index = _load_index(tmp_path / "chunks" / "general")
        assert sixty_key not in index, "Index entry should be removed"

        # Not recallable
        results = recall_memory(
            "eviction test memory sixty days", top_k=5, data_dir=tmp_path, chunk="general"
        )
        found_keys = [r["hex_key"] for r in results]
        assert sixty_key not in found_keys, "Evicted memory should not be recallable"


class TestEvictionAssociationCleanup:
    """Test 4: Association cleanup — evicted memory's edges removed, others preserved."""

    def test_association_cleanup_on_evict(self, tmp_path):
        """3 memories with co-recall edges, evict k3, k3's edges gone, k1-k2 edge preserved."""
        k1 = store_test_memory("assoc test memory alpha", tmp_path, chunk="general")
        k2 = store_test_memory("assoc test memory beta", tmp_path, chunk="general")
        k3 = store_test_memory("assoc test memory gamma", tmp_path, chunk="general")

        # Create co-recall edges between all three
        chunk_dir = tmp_path / "chunks" / "general"
        build_co_recall_associations(chunk_dir, [k1, k2, k3])

        # Verify edges exist
        n1_before = get_neighbors(chunk_dir, k1)
        assert len(n1_before) >= 1, f"k1 should have neighbors, got {len(n1_before)}"

        # Age k3 to make it dead, then evict
        age_memory(tmp_path, "general", k3, 60)
        evict_dead_memories(tmp_path)

        # k3's edges should be gone
        n3_after = get_neighbors(chunk_dir, k3)
        assert len(n3_after) == 0, f"Evicted memory k3 should have no neighbors, got {len(n3_after)}"

        # k1 should no longer reference k3
        n1_after = get_neighbors(chunk_dir, k1)
        assert k3 not in n1_after, f"k1 should no longer reference evicted k3"

        # k1 and k2 edge should still exist
        assert k2 in n1_after, f"k1-k2 edge should be preserved, k1 neighbors: {list(n1_after.keys())}"


class TestEvictionForgetMemory:
    """Test 5: forget_memory targeted — immediate removal."""

    def test_forget_memory_targeted(self, tmp_path):
        """Store, forget_memory(key), index entry and attractor gone."""
        k = store_test_memory("forget me immediately", tmp_path, chunk="general")
        chunk_dir = tmp_path / "chunks" / "general"

        assert k in _load_index(chunk_dir), "Memory should exist before forget"

        found = forget_memory(k, tmp_path)
        assert found is True, "forget_memory should return True"
        assert k not in _load_index(chunk_dir), "Index entry should be removed"

        att = chunk_dir / "attractors" / f"{k}.npy"
        assert not att.exists(), "Attractor should be removed"


class TestEvictionHotProtection:
    """Test 6: Hot/warm protection — recalled memories survive eviction."""

    def test_hot_memory_protected(self, tmp_path):
        """Recall 10x to make hot, age 5 days, evict_dead → not evicted."""
        k = store_test_memory("hot memory survives eviction", tmp_path, chunk="general")

        # Recall it 10 times to make it hot
        for _ in range(10):
            recall_memory(
                "hot memory survives eviction", top_k=1, data_dir=tmp_path, chunk="general"
            )

        # Age it moderately
        age_memory(tmp_path, "general", k, 5)

        # Re-score: index was updated by recalls, but we backdated timestamp only.
        # The hit_count=10 keeps temperature warm even after 5 days.
        scored = score_memories(tmp_path)
        mem = next((m for m in scored if m["hex_key"] == k), None)
        assert mem is not None and mem["temperature"] >= 0.3, (
            f"Recalled memory should be warm or hot, got temp={mem['temperature']:.4f}"
            if mem
            else "Memory not found in scored list"
        )

        # Try to evict — should not touch warm/hot
        evict_dead_memories(tmp_path)
        assert k in _load_index(tmp_path / "chunks" / "general"), "Hot memory should not be evicted"


class TestEvictionMinAgeProtection:
    """Test 7: MIN_AGE_DAYS protection — brand-new cold memory not evicted."""

    def test_min_age_protection(self, tmp_path):
        """Brand-new memory, fade+evict → not touched."""
        k = store_test_memory("brand new cold memory", tmp_path, chunk="general")

        # Even if we somehow had a very low temperature, MIN_AGE_DAYS protects it
        # A brand-new memory has temp=0.3 so wouldn't be evicted anyway,
        # but let's verify the age check works by trying fade/evict
        faded = fade_cold_memories(tmp_path)
        evicted = evict_dead_memories(tmp_path)

        assert k not in [m["hex_key"] for m in faded], "New memory should not be faded"
        assert k not in [m["hex_key"] for m in evicted], "New memory should not be evicted"
        assert k in _load_index(tmp_path / "chunks" / "general"), "New memory should still be in index"


class TestEvictionDryRun:
    """Test 8: dry_run — report is correct but nothing deleted."""

    def test_dry_run(self, tmp_path):
        """Age to dead, sweep_and_evict(dry_run=True), report has items but memory still in index."""
        k = store_test_memory("dry run test memory", tmp_path, chunk="general")
        age_memory(tmp_path, "general", k, 60)  # Make it dead

        result = sweep_and_evict(tmp_path, dry_run=True)

        # Report should show items to evict
        total_items = len(result.bricks_deleted) + len(result.memories_evicted)
        assert total_items > 0, f"Dry run report should have items, got {total_items}"

        # But nothing should be deleted
        assert k in _load_index(
            tmp_path / "chunks" / "general"
        ), "Memory should still be in index after dry_run"

        att = tmp_path / "chunks" / "general" / "attractors" / f"{k}.npy"
        assert att.exists(), "Attractor should still exist after dry_run"


class TestEvictionCapacityEviction:
    """Test 9: Capacity eviction — monkeypatch MAX_ATTRACTORS=5, store 8, coldest evicted."""

    def test_capacity_eviction(self, tmp_path):
        """Monkeypatch MAX_ATTRACTORS=5, store 8 memories, evict_for_capacity, oldest evicted."""
        cap_keys = []
        for i in range(8):
            k = store_test_memory(f"capacity test memory {i}", tmp_path, chunk="general")
            # Age them progressively: 2, 4, 6, 8, 10, 12, 14, 16 days
            age_memory(tmp_path, "general", k, (i + 1) * 2)
            cap_keys.append(k)

        assert len(score_memories(tmp_path)) == 8, "Expected 8 memories stored"

        # Monkeypatch MAX_ATTRACTORS to 5
        with mock.patch("wheeler_memory.eviction.MAX_ATTRACTORS", 5):
            evicted = evict_for_capacity(tmp_path)

        assert len(evicted) > 0, f"Expected some memories evicted, got {len(evicted)}"

        remaining = score_memories(tmp_path)
        assert len(remaining) < 8, f"Expected remaining < 8, got {len(remaining)}"

        # The evicted should be the coldest (oldest)
        evicted_keys = {m["hex_key"] for m in evicted}
        # The oldest key (16 days) should have been evicted
        assert cap_keys[-1] in evicted_keys, "Oldest memory (16 days) should have been evicted"
