#!/usr/bin/env python3
"""Integration test for eviction / forgetting.

Stores memories, simulates aging, and verifies that:
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
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

# Ensure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.eviction import (
    EvictionResult,
    evict_dead_memories,
    evict_for_capacity,
    fade_cold_memories,
    forget_memory,
    score_memories,
    sweep_and_evict,
)
from wheeler_memory.hashing import hash_to_frame, text_to_hex
from wheeler_memory.storage import _load_index, recall_memory, store_memory
from wheeler_memory.temperature import (
    TIER_DEAD,
    TIER_FADING,
    compute_temperature,
)
from wheeler_memory.warming import (
    build_co_recall_associations,
    get_neighbors,
    load_associations,
)

passes = 0
fails = 0


def check(name: str, condition: bool, detail: str = ""):
    global passes, fails
    if condition:
        passes += 1
        print(f"  \u2713 {name}")
    else:
        fails += 1
        msg = f"  \u2717 {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


def store_test_memory(text: str, data_dir: Path, chunk: str = "general") -> str:
    """Helper to store a memory and return its hex key."""
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False)


def age_memory(data_dir: Path, chunk: str, hex_key: str, days: float) -> None:
    """Backdate a memory's timestamp and last_accessed by *days*."""
    chunk_dir = data_dir / "chunks" / chunk
    index = _load_index(chunk_dir)
    if hex_key not in index:
        return
    past = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    index[hex_key]["timestamp"] = past
    index[hex_key]["metadata"]["last_accessed"] = past
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def main():
    global passes, fails
    tmp = Path(tempfile.mkdtemp(prefix="wheeler_eviction_test_"))
    print(f"Test data dir: {tmp}")

    try:
        # ==================================================================
        # Test 1: score_memories ordering
        # ==================================================================
        print("\n[1] score_memories ordering")

        texts_and_ages = [
            ("eviction test memory zero days", 0),
            ("eviction test memory seven days", 7),
            ("eviction test memory fourteen days", 14),
            ("eviction test memory thirty days", 30),
            ("eviction test memory sixty days", 60),
        ]
        keys = []
        for text, age in texts_and_ages:
            k = store_test_memory(text, tmp, chunk="general")
            if age > 0:
                age_memory(tmp, "general", k, age)
            keys.append(k)

        scored = score_memories(tmp)
        check("scored has 5 entries", len(scored) == 5, f"got {len(scored)}")

        # Coldest first — the oldest (60 days) should be first
        if len(scored) >= 2:
            check("sorted coldest-first",
                  scored[0]["temperature"] <= scored[-1]["temperature"],
                  f"first={scored[0]['temperature']:.6f}, last={scored[-1]['temperature']:.6f}")

        # The 60-day-old should be the coldest
        sixty_day_key = keys[4]
        check("60-day memory is coldest",
              scored[0]["hex_key"] == sixty_day_key,
              f"coldest is {scored[0]['hex_key'][:8]}..., expected {sixty_day_key[:8]}...")

        # ==================================================================
        # Test 2: fade_cold_memories
        # ==================================================================
        print("\n[2] fade_cold_memories")

        # The 30-day-old (base=0.3) should have temp ≈ 0.3 * 2^(-30/7) ≈ 0.0053
        thirty_key = keys[3]
        thirty_temp = next(m["temperature"] for m in scored if m["hex_key"] == thirty_key)
        check("30-day memory temp < TIER_FADING",
              thirty_temp < TIER_FADING,
              f"temp={thirty_temp:.6f}")

        brick_path = tmp / "chunks" / "general" / "bricks" / f"{thirty_key}.npz"
        check("brick exists before fade", brick_path.exists())

        faded = fade_cold_memories(tmp)
        check("fade returned entries", len(faded) > 0, f"got {len(faded)}")

        check("brick deleted after fade", not brick_path.exists())

        # Attractor and index should remain
        att_path = tmp / "chunks" / "general" / "attractors" / f"{thirty_key}.npy"
        check("attractor survives fade", att_path.exists())

        index = _load_index(tmp / "chunks" / "general")
        check("index entry survives fade", thirty_key in index)

        # Memory should still be recallable (attractor exists)
        results = recall_memory("eviction test memory thirty days", top_k=5, data_dir=tmp, chunk="general")
        found_keys = [r["hex_key"] for r in results]
        check("faded memory still recallable", thirty_key in found_keys)

        # Re-age memories that were bumped by the recall above
        for text, age in texts_and_ages:
            if age > 0:
                k = text_to_hex(text)
                age_memory(tmp, "general", k, age)

        # ==================================================================
        # Test 3: evict_dead_memories
        # ==================================================================
        print("\n[3] evict_dead_memories")

        sixty_key = keys[4]
        scored = score_memories(tmp)
        sixty_temp = next(m["temperature"] for m in scored if m["hex_key"] == sixty_key)
        check("60-day memory temp < TIER_DEAD",
              sixty_temp < TIER_DEAD,
              f"temp={sixty_temp:.6f}")

        evicted = evict_dead_memories(tmp)
        check("evict returned entries", len(evicted) > 0, f"got {len(evicted)}")

        # All artifacts should be gone
        att_path_60 = tmp / "chunks" / "general" / "attractors" / f"{sixty_key}.npy"
        brick_path_60 = tmp / "chunks" / "general" / "bricks" / f"{sixty_key}.npz"
        check("attractor removed", not att_path_60.exists())
        check("brick removed", not brick_path_60.exists())

        index = _load_index(tmp / "chunks" / "general")
        check("index entry removed", sixty_key not in index)

        # Not recallable
        results = recall_memory("eviction test memory sixty days", top_k=5, data_dir=tmp, chunk="general")
        found_keys = [r["hex_key"] for r in results]
        check("evicted memory not recallable", sixty_key not in found_keys)

        # ==================================================================
        # Test 4: Association cleanup
        # ==================================================================
        print("\n[4] Association cleanup")

        # Fresh dir for this test
        tmp4 = Path(tempfile.mkdtemp(prefix="wheeler_eviction_assoc_"))
        try:
            k1 = store_test_memory("assoc test memory alpha", tmp4, chunk="general")
            k2 = store_test_memory("assoc test memory beta", tmp4, chunk="general")
            k3 = store_test_memory("assoc test memory gamma", tmp4, chunk="general")

            # Create co-recall edges between all three
            chunk_dir4 = tmp4 / "chunks" / "general"
            build_co_recall_associations(chunk_dir4, [k1, k2, k3])

            # Verify edges exist
            n1_before = get_neighbors(chunk_dir4, k1)
            check("k1 has neighbors before eviction", len(n1_before) >= 1,
                  f"got {len(n1_before)}")

            # Age k3 to make it dead, then evict
            age_memory(tmp4, "general", k3, 60)
            evict_dead_memories(tmp4)

            # k3's edges should be gone
            n3_after = get_neighbors(chunk_dir4, k3)
            check("evicted memory has no neighbors", len(n3_after) == 0,
                  f"got {len(n3_after)}")

            # k1 should no longer reference k3
            n1_after = get_neighbors(chunk_dir4, k1)
            check("k1 no longer references evicted k3", k3 not in n1_after)

            # k1 and k2 edge should still exist
            check("k1-k2 edge preserved", k2 in n1_after,
                  f"k1 neighbors: {list(n1_after.keys())}")
        finally:
            shutil.rmtree(tmp4, ignore_errors=True)

        # ==================================================================
        # Test 5: forget_memory targeted
        # ==================================================================
        print("\n[5] forget_memory targeted")

        tmp5 = Path(tempfile.mkdtemp(prefix="wheeler_eviction_forget_"))
        try:
            k = store_test_memory("forget me immediately", tmp5, chunk="general")
            chunk_dir5 = tmp5 / "chunks" / "general"

            check("memory exists before forget", k in _load_index(chunk_dir5))

            found = forget_memory(k, tmp5)
            check("forget_memory returned True", found)
            check("index entry removed", k not in _load_index(chunk_dir5))

            att = chunk_dir5 / "attractors" / f"{k}.npy"
            check("attractor removed", not att.exists())
        finally:
            shutil.rmtree(tmp5, ignore_errors=True)

        # ==================================================================
        # Test 6: Hot/warm protection
        # ==================================================================
        print("\n[6] Hot/warm protection")

        tmp6 = Path(tempfile.mkdtemp(prefix="wheeler_eviction_hot_"))
        try:
            k = store_test_memory("hot memory survives eviction", tmp6, chunk="general")
            # Recall it 10 times to make it hot
            for _ in range(10):
                recall_memory("hot memory survives eviction", top_k=1, data_dir=tmp6, chunk="general")

            # Age it moderately
            age_memory(tmp6, "general", k, 5)

            # Re-score: index was updated by recalls, but we backdated timestamp only.
            # The hit_count=10 keeps temperature warm even after 5 days.
            scored = score_memories(tmp6)
            mem = next((m for m in scored if m["hex_key"] == k), None)
            check("recalled memory is warm or hot",
                  mem is not None and mem["temperature"] >= 0.3,
                  f"temp={mem['temperature']:.4f}" if mem else "not found")

            # Try to evict — should not touch warm/hot
            evicted = evict_dead_memories(tmp6)
            check("hot memory not evicted", k in _load_index(tmp6 / "chunks" / "general"))
        finally:
            shutil.rmtree(tmp6, ignore_errors=True)

        # ==================================================================
        # Test 7: MIN_AGE_DAYS protection
        # ==================================================================
        print("\n[7] MIN_AGE_DAYS protection")

        tmp7 = Path(tempfile.mkdtemp(prefix="wheeler_eviction_age_"))
        try:
            # Store a memory but don't age it — it's brand new
            k = store_test_memory("brand new cold memory", tmp7, chunk="general")

            # Even if we somehow had a very low temperature, MIN_AGE_DAYS protects it
            # A brand-new memory has temp=0.3 so wouldn't be evicted anyway,
            # but let's verify the age check works by trying fade/evict
            faded = fade_cold_memories(tmp7)
            evicted = evict_dead_memories(tmp7)
            check("new memory not faded", k not in [m["hex_key"] for m in faded])
            check("new memory not evicted", k not in [m["hex_key"] for m in evicted])
            check("new memory still in index", k in _load_index(tmp7 / "chunks" / "general"))
        finally:
            shutil.rmtree(tmp7, ignore_errors=True)

        # ==================================================================
        # Test 8: dry_run
        # ==================================================================
        print("\n[8] dry_run")

        tmp8 = Path(tempfile.mkdtemp(prefix="wheeler_eviction_dry_"))
        try:
            k = store_test_memory("dry run test memory", tmp8, chunk="general")
            age_memory(tmp8, "general", k, 60)  # Make it dead

            result = sweep_and_evict(tmp8, dry_run=True)

            # Report should show items to evict
            total_items = len(result.bricks_deleted) + len(result.memories_evicted)
            check("dry_run report has items", total_items > 0, f"got {total_items}")

            # But nothing should be deleted
            check("memory still in index after dry_run",
                  k in _load_index(tmp8 / "chunks" / "general"))

            att = tmp8 / "chunks" / "general" / "attractors" / f"{k}.npy"
            check("attractor still exists after dry_run", att.exists())
        finally:
            shutil.rmtree(tmp8, ignore_errors=True)

        # ==================================================================
        # Test 9: Capacity eviction
        # ==================================================================
        print("\n[9] Capacity eviction (MAX_ATTRACTORS=5)")

        tmp9 = Path(tempfile.mkdtemp(prefix="wheeler_eviction_cap_"))
        try:
            # Store 8 memories with varying ages
            cap_keys = []
            for i in range(8):
                k = store_test_memory(f"capacity test memory {i}", tmp9, chunk="general")
                # Age them progressively: 2, 4, 6, 8, 10, 12, 14, 16 days
                age_memory(tmp9, "general", k, (i + 1) * 2)
                cap_keys.append(k)

            check("8 memories stored", len(score_memories(tmp9)) == 8)

            # Monkeypatch MAX_ATTRACTORS to 5
            with mock.patch("wheeler_memory.eviction.MAX_ATTRACTORS", 5):
                evicted = evict_for_capacity(tmp9)

            check("some memories evicted for capacity", len(evicted) > 0,
                  f"evicted {len(evicted)}")

            remaining = score_memories(tmp9)
            check("remaining count reduced", len(remaining) < 8,
                  f"remaining={len(remaining)}")

            # The evicted should be the coldest (oldest)
            evicted_keys = {m["hex_key"] for m in evicted}
            # The oldest key (16 days) should have been evicted
            check("oldest memory was evicted", cap_keys[-1] in evicted_keys)
        finally:
            shutil.rmtree(tmp9, ignore_errors=True)

        # ==================================================================
        # Summary
        # ==================================================================
        print(f"\n{'='*60}")
        print(f"  EVICTION TEST RESULTS")
        print(f"{'='*60}")
        print(f"  Passed: {passes}")
        print(f"  Failed: {fails}")
        print(f"  Overall: {'PASS' if fails == 0 else 'FAIL'}")
        print(f"{'='*60}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
