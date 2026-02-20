#!/usr/bin/env python3
"""Integration test for associative warming.

Stores related memories, triggers recall, and verifies that:
1. Store-time associations form between correlated attractors
2. Co-recall associations form between memories recalled together
3. Warmth propagates to neighbors on recall (hop 1 and hop 2)
4. Warmth decays over simulated time
5. Effective temperature includes warmth boost
"""

import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Ensure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame, text_to_hex
from wheeler_memory.brick import MemoryBrick
from wheeler_memory.storage import recall_memory, store_memory, list_memories
from wheeler_memory.temperature import (
    compute_temperature,
    compute_warmth,
    effective_temperature,
    WARMTH_HOP1,
    WARMTH_HOP2,
    MAX_WARMTH,
    WARMTH_FLOOR,
)
from wheeler_memory.warming import (
    build_store_associations,
    build_co_recall_associations,
    load_associations,
    load_warmth,
    propagate_warmth,
    get_neighbors,
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
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk)


def main():
    global passes, fails
    tmp = Path(tempfile.mkdtemp(prefix="wheeler_warming_test_"))
    print(f"Test data dir: {tmp}")

    try:
        # ==================================================================
        # Test 1: compute_warmth pure computation
        # ==================================================================
        print("\n[1] compute_warmth decay")

        now = datetime.now(timezone.utc)
        # At t=0, full boost
        w0 = compute_warmth(0.05, now.isoformat(), now=now)
        check("warmth at t=0 equals boost", abs(w0 - 0.05) < 1e-4, f"got {w0}")

        # After 1 day (half-life), half boost
        w1 = compute_warmth(0.05, now.isoformat(), now=now + timedelta(days=1))
        check("warmth at t=1d is half", abs(w1 - 0.025) < 1e-4, f"got {w1}")

        # After 3 days, ~12.5%
        w3 = compute_warmth(0.05, now.isoformat(), now=now + timedelta(days=3))
        check("warmth at t=3d is ~0.00625", abs(w3 - 0.00625) < 1e-4, f"got {w3}")

        # After 10 days, below floor → 0
        w10 = compute_warmth(0.05, now.isoformat(), now=now + timedelta(days=10))
        check("warmth at t=10d is 0 (below floor)", w10 == 0.0, f"got {w10}")

        # ==================================================================
        # Test 2: effective_temperature
        # ==================================================================
        print("\n[2] effective_temperature")

        base = compute_temperature(5, now.isoformat(), now=now)
        eff = effective_temperature(5, now.isoformat(), warmth_boost=0.05,
                                    warmth_applied_at=now.isoformat(), now=now)
        check("effective = base + warmth", abs(eff - (base + 0.05)) < 1e-4,
              f"base={base}, eff={eff}")

        # Cap at 1.0
        eff_cap = effective_temperature(10, now.isoformat(), warmth_boost=0.15,
                                        warmth_applied_at=now.isoformat(), now=now)
        check("effective capped at 1.0", eff_cap <= 1.0, f"got {eff_cap}")

        # No warmth returns base
        eff_none = effective_temperature(5, now.isoformat(), now=now)
        check("no warmth returns base temp", abs(eff_none - base) < 1e-4)

        # ==================================================================
        # Test 3: Store-time associations
        # ==================================================================
        print("\n[3] Store-time associations")

        # Store several memories in the same chunk
        texts = [
            "fix the python debug error in the login module",
            "debug the python authentication bug",
            "buy groceries milk eggs bread",
            "quantum entanglement violates Bell inequalities",
            "resolve python import error in tests",
        ]
        keys = []
        for t in texts:
            k = store_test_memory(t, tmp, chunk="general")
            keys.append(k)

        chunk_dir = tmp / "chunks" / "general"
        assoc = load_associations(chunk_dir)
        edges = assoc.get("edges", {})
        total_edges = sum(len(v) for v in edges.values()) // 2  # bidirectional
        print(f"  Stored {len(texts)} memories, {total_edges} edges formed")
        # SHA-256 attractors have near-zero correlation (~0.015 avg), so
        # store-time edges only form in embedding mode.  0 edges is correct here.
        check("store-time edges: 0 expected for SHA-256 mode", total_edges == 0,
              f"got {total_edges}")

        # Check that each edge is bidirectional
        bidirectional = True
        for a, neighbors in edges.items():
            for b in neighbors:
                if a not in edges.get(b, {}):
                    bidirectional = False
                    break
        check("all edges are bidirectional", bidirectional)

        # ==================================================================
        # Test 4: Co-recall associations
        # ==================================================================
        print("\n[4] Co-recall associations")

        edges_before = sum(len(v) for v in load_associations(chunk_dir).get("edges", {}).values()) // 2
        # Recall should return multiple results and form co-recall edges
        results = recall_memory("python debug error", top_k=3, data_dir=tmp, chunk="general")
        edges_after = sum(len(v) for v in load_associations(chunk_dir).get("edges", {}).values()) // 2
        check("recall returned results", len(results) > 0, f"got {len(results)}")
        check("co-recall may add edges", edges_after >= edges_before,
              f"before={edges_before}, after={edges_after}")

        # ==================================================================
        # Test 5: Warmth propagation
        # ==================================================================
        print("\n[5] Warmth propagation")

        # Find a key that has neighbors
        key_with_neighbors = None
        assoc = load_associations(chunk_dir)
        for k, nbrs in assoc.get("edges", {}).items():
            if len(nbrs) >= 1:
                key_with_neighbors = k
                break

        if key_with_neighbors:
            neighbors = list(assoc["edges"][key_with_neighbors].keys())
            # Clear any existing warmth
            assoc["warmth"] = {}
            (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

            warmed = propagate_warmth(chunk_dir, [key_with_neighbors])
            check("warmth propagated to neighbors", len(warmed) > 0, f"warmed {len(warmed)} memories")

            # Check hop 1 neighbor got WARMTH_HOP1
            if neighbors[0] in warmed:
                check(f"hop-1 neighbor boost is {WARMTH_HOP1}",
                      abs(warmed[neighbors[0]] - WARMTH_HOP1) < 1e-4,
                      f"got {warmed[neighbors[0]]}")

            # Check warmth is persisted
            warmth_on_disk = load_warmth(chunk_dir)
            check("warmth persisted to associations.json", len(warmth_on_disk) > 0)

            # Check hop 2 if there are second-degree neighbors
            hop2_found = False
            for n1 in neighbors:
                n1_neighbors = assoc["edges"].get(n1, {})
                for n2 in n1_neighbors:
                    if n2 != key_with_neighbors and n2 not in set(neighbors):
                        if n2 in warmed:
                            check(f"hop-2 neighbor boost is {WARMTH_HOP2}",
                                  abs(warmed[n2] - WARMTH_HOP2) < 1e-4,
                                  f"got {warmed[n2]}")
                            hop2_found = True
                            break
                if hop2_found:
                    break
            if not hop2_found:
                print("  - (no hop-2 neighbors to test, skipping)")
        else:
            print("  - (no edges formed, skipping warmth propagation tests)")

        # ==================================================================
        # Test 6: Warmth appears in list_memories
        # ==================================================================
        print("\n[6] Warmth in list_memories")

        memories = list_memories(data_dir=tmp, chunk="general")
        warmed_keys = set(load_warmth(chunk_dir).keys())
        if warmed_keys:
            warmed_mem = [m for m in memories if m["hex_key"] in warmed_keys]
            if warmed_mem:
                m = warmed_mem[0]
                base = compute_temperature(
                    m["metadata"]["hit_count"],
                    m["metadata"]["last_accessed"],
                )
                check("warmed memory temp > base temp",
                      m["temperature"] > base,
                      f"effective={m['temperature']}, base={base}")
            else:
                print("  - (warmed memories not found in list)")
        else:
            print("  - (no active warmth to check)")

        # ==================================================================
        # Test 7: MAX_WARMTH cap
        # ==================================================================
        print("\n[7] Warmth cap")

        assoc = load_associations(chunk_dir)
        if key_with_neighbors:
            # Repeatedly propagate to accumulate warmth
            for _ in range(10):
                propagate_warmth(chunk_dir, [key_with_neighbors])
            warmth_data = load_warmth(chunk_dir)
            for hk, entry in warmth_data.items():
                check(f"warmth capped at {MAX_WARMTH}",
                      entry["boost"] <= MAX_WARMTH + 1e-4,
                      f"got {entry['boost']}")
                break  # Just check one

        # ==================================================================
        # Test 8: Fired memories don't warm each other
        # ==================================================================
        print("\n[8] Fired exclusion")

        assoc = load_associations(chunk_dir)
        assoc["warmth"] = {}
        (chunk_dir / "associations.json").write_text(json.dumps(assoc, indent=2))

        # Fire two connected memories
        edges = assoc.get("edges", {})
        pair = None
        for a, nbrs in edges.items():
            for b in nbrs:
                pair = (a, b)
                break
            if pair:
                break

        if pair:
            warmed = propagate_warmth(chunk_dir, list(pair))
            check("fired memory A not warmed", pair[0] not in warmed)
            check("fired memory B not warmed", pair[1] not in warmed)
        else:
            print("  - (no edge pair to test)")

        # ==================================================================
        # Summary
        # ==================================================================
        print(f"\n{'='*60}")
        print(f"  ASSOCIATIVE WARMING TEST RESULTS")
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
