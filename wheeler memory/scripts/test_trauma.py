#!/usr/bin/env python3
"""Integration tests for dual-attractor trauma encoding.

Tests:
 1. Store trauma pair — both memories stored, trauma.json correct
 2. Index metadata — trauma_pair_id and trauma_role set
 3. Association edge — trauma_pair source edge created
 4. Auto recall — avoidance injected with trauma_avoidance=True
 5. Safe exposure — avoidance NOT injected, safe_exposure_count incremented
 6. Suppression decay — decays after threshold, floors at 0.05
 7. Suppress mode — no trauma processing, no counter changes
 8. Therapy status — correct status labels and fields
 9. Pair removal — trauma.json cleaned, index metadata cleared, memories survive
10. Eviction cleanup — evicting a member removes the pair
11. Cross-chunk pairs — experience and avoidance in different chunks
12. List pairs — multiple pairs returned with correct statuses
13. Backward compatibility — non-trauma recall unchanged
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.eviction import forget_memory
from wheeler_memory.hashing import hash_to_frame, text_to_hex
from wheeler_memory.storage import recall_memory, store_memory
from wheeler_memory.temperature import (
    TRAUMA_DECAY_PER_SAFE_EXPOSURE,
    TRAUMA_INITIAL_SUPPRESSION,
    TRAUMA_SAFE_EXPOSURE_THRESHOLD,
    TRAUMA_SUPPRESSION_FLOOR,
)
from wheeler_memory.trauma import (
    _load_trauma,
    check_trauma_activation,
    inject_avoidance_results,
    list_trauma_pairs,
    record_trauma_activation,
    remove_trauma_pair,
    store_trauma_pair,
    therapy_status,
)
from wheeler_memory.warming import load_associations

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
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk)


def main():
    global passes, fails
    tmp = Path(tempfile.mkdtemp(prefix="wheeler_trauma_test_"))
    print(f"Test data dir: {tmp}")

    try:
        # ==================================================================
        # Test 1: Store trauma pair
        # ==================================================================
        print("\n[1] Store trauma pair")

        result = store_trauma_pair(
            "the car crash",
            "driving on highways",
            data_dir=tmp,
            chunk="general",
        )
        pair_id = result["pair_id"]
        exp_hex = result["experience_hex"]
        avoid_hex = result["avoidance_hex"]

        check("pair_id is 12 chars", len(pair_id) == 12, f"got {len(pair_id)}")
        check("experience hex returned", len(exp_hex) > 0)
        check("avoidance hex returned", len(avoid_hex) > 0)

        trauma = _load_trauma(tmp)
        check("pair exists in trauma.json", pair_id in trauma["pairs"])

        pair = trauma["pairs"][pair_id]
        check("suppression is 1.0", pair["suppression_strength"] == TRAUMA_INITIAL_SUPPRESSION,
              f"got {pair['suppression_strength']}")
        check("activation_count is 0", pair["activation_count"] == 0)
        check("safe_exposure_count is 0", pair["safe_exposure_count"] == 0)

        # ==================================================================
        # Test 2: Index metadata
        # ==================================================================
        print("\n[2] Index metadata")

        chunk_dir = tmp / "chunks" / "general"
        index = json.loads((chunk_dir / "index.json").read_text())

        exp_meta = index.get(exp_hex, {}).get("metadata", {})
        check("experience has trauma_pair_id", exp_meta.get("trauma_pair_id") == pair_id,
              f"got {exp_meta.get('trauma_pair_id')}")
        check("experience has trauma_role=experience", exp_meta.get("trauma_role") == "experience",
              f"got {exp_meta.get('trauma_role')}")

        avoid_meta = index.get(avoid_hex, {}).get("metadata", {})
        check("avoidance has trauma_pair_id", avoid_meta.get("trauma_pair_id") == pair_id,
              f"got {avoid_meta.get('trauma_pair_id')}")
        check("avoidance has trauma_role=avoidance", avoid_meta.get("trauma_role") == "avoidance",
              f"got {avoid_meta.get('trauma_role')}")

        # ==================================================================
        # Test 3: Association edge
        # ==================================================================
        print("\n[3] Association edge")

        assoc = load_associations(chunk_dir)
        edges = assoc.get("edges", {})
        exp_edges = edges.get(exp_hex, {})
        has_trauma_edge = any(
            e.get("source") == "trauma_pair"
            for e in exp_edges.values()
        )
        check("trauma_pair edge exists from experience", has_trauma_edge)

        if avoid_hex in exp_edges:
            edge = exp_edges[avoid_hex]
            check("edge has trauma_pair_id", edge.get("trauma_pair_id") == pair_id,
                  f"got {edge.get('trauma_pair_id')}")

        # ==================================================================
        # Test 4: Auto recall — avoidance injected
        # ==================================================================
        print("\n[4] Auto recall — avoidance injected")

        results = recall_memory(
            "the car crash", top_k=5, data_dir=tmp, chunk="general",
            trauma_mode="auto",
        )
        avoidance_results = [r for r in results if r.get("trauma_avoidance")]
        check("avoidance injected in auto mode", len(avoidance_results) > 0,
              f"got {len(avoidance_results)} avoidance results")

        if avoidance_results:
            ar = avoidance_results[0]
            check("avoidance has trauma_pair_id", ar.get("trauma_pair_id") == pair_id)
            check("avoidance has suppression_strength", ar.get("suppression_strength") == 1.0,
                  f"got {ar.get('suppression_strength')}")

        # ==================================================================
        # Test 5: Safe exposure — no injection, counter incremented
        # ==================================================================
        print("\n[5] Safe exposure mode")

        # Reset counters by re-reading
        trauma_before = _load_trauma(tmp)
        act_before = trauma_before["pairs"][pair_id]["activation_count"]

        results_safe = recall_memory(
            "the car crash", top_k=5, data_dir=tmp, chunk="general",
            trauma_mode="safe",
        )
        avoidance_safe = [r for r in results_safe if r.get("trauma_avoidance")]
        check("avoidance NOT injected in safe mode", len(avoidance_safe) == 0,
              f"got {len(avoidance_safe)}")

        trauma_after = _load_trauma(tmp)
        safe_count = trauma_after["pairs"][pair_id]["safe_exposure_count"]
        check("safe_exposure_count incremented", safe_count > 0,
              f"got {safe_count}")

        # ==================================================================
        # Test 6: Suppression decay
        # ==================================================================
        print("\n[6] Suppression decay")

        # Need to do enough safe exposures to pass threshold
        # Already did 1 safe exposure (test 5) + 1 auto (test 4)
        # Need TRAUMA_SAFE_EXPOSURE_THRESHOLD + 1 more safe exposures
        trauma_data = _load_trauma(tmp)
        current_safe = trauma_data["pairs"][pair_id]["safe_exposure_count"]
        needed = TRAUMA_SAFE_EXPOSURE_THRESHOLD - current_safe + 1
        for _ in range(max(0, needed) + 1):
            record_trauma_activation(pair_id, tmp, safe_context=True)

        trauma_data = _load_trauma(tmp)
        suppression = trauma_data["pairs"][pair_id]["suppression_strength"]
        check("suppression decayed below 1.0", suppression < TRAUMA_INITIAL_SUPPRESSION,
              f"got {suppression}")

        # Keep going until we approach floor
        for _ in range(50):
            record_trauma_activation(pair_id, tmp, safe_context=True)

        trauma_data = _load_trauma(tmp)
        suppression = trauma_data["pairs"][pair_id]["suppression_strength"]
        check("suppression floors at TRAUMA_SUPPRESSION_FLOOR",
              abs(suppression - TRAUMA_SUPPRESSION_FLOOR) < 1e-6,
              f"got {suppression}")

        # ==================================================================
        # Test 7: Suppress mode — no processing
        # ==================================================================
        print("\n[7] Suppress mode")

        # Reset a fresh pair for this test
        result2 = store_trauma_pair(
            "being attacked by a dog",
            "fear of all dogs",
            data_dir=tmp,
            chunk="general",
        )
        pair_id2 = result2["pair_id"]

        trauma_before = _load_trauma(tmp)
        act_before = trauma_before["pairs"][pair_id2]["activation_count"]

        results_suppress = recall_memory(
            "being attacked by a dog", top_k=5, data_dir=tmp, chunk="general",
            trauma_mode="suppress",
        )
        avoidance_suppress = [r for r in results_suppress if r.get("trauma_avoidance")]
        check("avoidance NOT injected in suppress mode", len(avoidance_suppress) == 0)

        trauma_after = _load_trauma(tmp)
        act_after = trauma_after["pairs"][pair_id2]["activation_count"]
        check("activation_count unchanged in suppress mode", act_after == act_before,
              f"before={act_before}, after={act_after}")

        # ==================================================================
        # Test 8: Therapy status
        # ==================================================================
        print("\n[8] Therapy status")

        # pair_id (from test 1) is resolved by now
        st = therapy_status(pair_id, tmp)
        check("resolved pair has status=resolved", st["status"] == "resolved",
              f"got {st['status']}")
        check("status has suppression_strength field", "suppression_strength" in st)
        check("status has experience field", "experience" in st)
        check("status has avoidance field", "avoidance" in st)
        check("status has therapy_history", "therapy_history" in st)

        # pair_id2 is still active
        st2 = therapy_status(pair_id2, tmp)
        check("active pair has status=active", st2["status"] == "active",
              f"got {st2['status']}")

        # ==================================================================
        # Test 9: Pair removal
        # ==================================================================
        print("\n[9] Pair removal")

        # Store a fresh pair just for removal
        result3 = store_trauma_pair(
            "falling from a height",
            "fear of ladders",
            data_dir=tmp,
            chunk="general",
        )
        pair_id3 = result3["pair_id"]
        exp_hex3 = result3["experience_hex"]
        avoid_hex3 = result3["avoidance_hex"]

        removed = remove_trauma_pair(pair_id3, tmp)
        check("remove returned True", removed)

        trauma = _load_trauma(tmp)
        check("pair removed from trauma.json", pair_id3 not in trauma["pairs"])
        check("exp hex removed from index_by_hex",
              trauma["index_by_hex"].get(exp_hex3) != pair_id3)
        check("avoid hex removed from index_by_hex",
              trauma["index_by_hex"].get(avoid_hex3) != pair_id3)

        # Check index metadata cleared
        index = json.loads((chunk_dir / "index.json").read_text())
        if exp_hex3 in index:
            check("experience index metadata cleared",
                  "trauma_pair_id" not in index[exp_hex3].get("metadata", {}))
        if avoid_hex3 in index:
            check("avoidance index metadata cleared",
                  "trauma_pair_id" not in index[avoid_hex3].get("metadata", {}))

        # Memories still exist
        check("experience memory survives removal",
              (chunk_dir / "attractors" / f"{exp_hex3}.npy").exists())
        check("avoidance memory survives removal",
              (chunk_dir / "attractors" / f"{avoid_hex3}.npy").exists())

        # ==================================================================
        # Test 10: Eviction cleanup
        # ==================================================================
        print("\n[10] Eviction cleanup")

        result4 = store_trauma_pair(
            "house fire emergency",
            "fear of smoke smell",
            data_dir=tmp,
            chunk="general",
        )
        pair_id4 = result4["pair_id"]
        exp_hex4 = result4["experience_hex"]

        trauma_before = _load_trauma(tmp)
        check("pair exists before eviction", pair_id4 in trauma_before["pairs"])

        # Evict the experience memory
        forget_memory(exp_hex4, tmp)

        trauma_after = _load_trauma(tmp)
        check("pair removed after evicting experience",
              pair_id4 not in trauma_after["pairs"])

        # ==================================================================
        # Test 11: Cross-chunk pairs
        # ==================================================================
        print("\n[11] Cross-chunk pairs")

        result5 = store_trauma_pair(
            "the python code crash that deleted data",
            "fear of running python scripts",
            data_dir=tmp,
        )
        pair_id5 = result5["pair_id"]
        exp_chunk5 = result5["experience_chunk"]
        avoid_chunk5 = result5["avoidance_chunk"]

        trauma = _load_trauma(tmp)
        check("cross-chunk pair stored", pair_id5 in trauma["pairs"])
        check("experience chunk recorded", trauma["pairs"][pair_id5]["experience"]["chunk"] == exp_chunk5)
        check("avoidance chunk recorded", trauma["pairs"][pair_id5]["avoidance"]["chunk"] == avoid_chunk5)

        # ==================================================================
        # Test 12: List pairs
        # ==================================================================
        print("\n[12] List pairs")

        pairs = list_trauma_pairs(tmp)
        check("list returns multiple pairs", len(pairs) >= 2,
              f"got {len(pairs)}")

        statuses = {p["status"] for p in pairs}
        check("list includes status field", len(statuses) > 0)

        pair_ids_listed = {p["pair_id"] for p in pairs}
        # pair_id2 should still be there (active)
        check("active pair appears in list", pair_id2 in pair_ids_listed,
              f"pair_id2={pair_id2}, listed={pair_ids_listed}")

        # ==================================================================
        # Test 13: Backward compatibility — non-trauma recall unchanged
        # ==================================================================
        print("\n[13] Backward compatibility")

        # Store a normal (non-trauma) memory
        normal_hex = store_test_memory("regular memory about cooking pasta", tmp, chunk="general")

        results_normal = recall_memory(
            "cooking pasta", top_k=5, data_dir=tmp, chunk="general",
            trauma_mode="auto",
        )
        # Should get results without any trauma processing errors
        check("non-trauma recall returns results", len(results_normal) > 0)

        # Check none of the normal results have trauma markers (unless a trauma
        # memory happened to match by correlation)
        normal_only = [r for r in results_normal
                       if r["hex_key"] == normal_hex]
        if normal_only:
            r = normal_only[0]
            check("normal memory has no trauma_avoidance flag",
                  not r.get("trauma_avoidance", False))

        # ==================================================================
        # Summary
        # ==================================================================
        print(f"\n{'='*60}")
        print(f"  TRAUMA ENCODING TEST RESULTS")
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
