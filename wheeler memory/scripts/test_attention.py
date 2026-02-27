"""Tests for the attention model (variable tick rates).

Run: python scripts/test_attention.py
"""

import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from wheeler_memory.attention import (
    AttentionBudget,
    compute_attention_budget,
    salience_from_label,
    salience_from_temperature,
)
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.temperature import (
    SALIENCE_DEFAULT,
    SALIENCE_MAX_ITERS_HIGH,
    SALIENCE_MAX_ITERS_LOW,
    SALIENCE_MAX_ITERS_MED,
    SALIENCE_THRESHOLD_HIGH,
    SALIENCE_THRESHOLD_LOW,
    SALIENCE_THRESHOLD_MED,
)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


# ── Test 1: Budget computation at anchor points ──────────────────────

print("\n1. Budget computation at anchor points")

b0 = compute_attention_budget(0.0)
check("salience=0.0 max_iters", b0.max_iters == SALIENCE_MAX_ITERS_LOW,
      f"got {b0.max_iters}, expected {SALIENCE_MAX_ITERS_LOW}")
check("salience=0.0 threshold", abs(b0.stability_threshold - SALIENCE_THRESHOLD_LOW) < 1e-10,
      f"got {b0.stability_threshold}, expected {SALIENCE_THRESHOLD_LOW}")

b5 = compute_attention_budget(0.5)
check("salience=0.5 max_iters", b5.max_iters == SALIENCE_MAX_ITERS_MED,
      f"got {b5.max_iters}, expected {SALIENCE_MAX_ITERS_MED}")
check("salience=0.5 threshold", abs(b5.stability_threshold - SALIENCE_THRESHOLD_MED) < 1e-10,
      f"got {b5.stability_threshold}, expected {SALIENCE_THRESHOLD_MED}")

b1 = compute_attention_budget(1.0)
check("salience=1.0 max_iters", b1.max_iters == SALIENCE_MAX_ITERS_HIGH,
      f"got {b1.max_iters}, expected {SALIENCE_MAX_ITERS_HIGH}")
check("salience=1.0 threshold", abs(b1.stability_threshold - SALIENCE_THRESHOLD_HIGH) < 1e-10,
      f"got {b1.stability_threshold}, expected {SALIENCE_THRESHOLD_HIGH}")

# ── Test 2: Monotonicity ─────────────────────────────────────────────

print("\n2. Monotonicity")

saliences = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
budgets = [compute_attention_budget(s) for s in saliences]

iters_monotonic = all(
    budgets[i].max_iters <= budgets[i + 1].max_iters
    for i in range(len(budgets) - 1)
)
check("max_iters non-decreasing", iters_monotonic)

thresh_monotonic = all(
    budgets[i].stability_threshold >= budgets[i + 1].stability_threshold
    for i in range(len(budgets) - 1)
)
check("threshold non-increasing", thresh_monotonic)

# ── Test 3: Clamping ─────────────────────────────────────────────────

print("\n3. Clamping")

b_neg = compute_attention_budget(-0.5)
check("salience < 0 clamped", b_neg.salience == 0.0,
      f"got {b_neg.salience}")
check("salience < 0 matches salience=0", b_neg.max_iters == b0.max_iters)

b_over = compute_attention_budget(1.5)
check("salience > 1 clamped", b_over.salience == 1.0,
      f"got {b_over.salience}")
check("salience > 1 matches salience=1", b_over.max_iters == b1.max_iters)

# ── Test 4: Label conversion ─────────────────────────────────────────

print("\n4. Label conversion")

check("low → 0.2", salience_from_label("low") == 0.2)
check("medium → 0.5", salience_from_label("medium") == 0.5)
check("high → 0.9", salience_from_label("high") == 0.9)
check("HIGH (case insensitive) → 0.9", salience_from_label("HIGH") == 0.9)
check("unknown → default", salience_from_label("bogus") == SALIENCE_DEFAULT)

# Also check AttentionBudget.label property
check("label property low", compute_attention_budget(0.2).label == "low")
check("label property medium", compute_attention_budget(0.5).label == "medium")
check("label property high", compute_attention_budget(0.9).label == "high")

# ── Test 5: Temperature to salience ──────────────────────────────────

print("\n5. Temperature to salience")

check("temp=0 → 0.1", abs(salience_from_temperature(0.0) - 0.1) < 1e-9)
check("temp=1 → 1.0", abs(salience_from_temperature(1.0) - 1.0) < 1e-9)

temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sal_values = [salience_from_temperature(t) for t in temps]
temp_monotonic = all(sal_values[i] <= sal_values[i + 1] for i in range(len(sal_values) - 1))
check("temperature→salience monotonic", temp_monotonic)

# ── Test 6: Backwards compatibility ──────────────────────────────────

print("\n6. Backwards compatibility")

frame = hash_to_frame("backwards compatibility test input")

# Default call (no extra params) should use threshold=1e-4, max_iters=1000
result_default = evolve_and_interpret(frame.copy())

# Explicit call with old defaults
result_explicit = evolve_and_interpret(frame.copy(), max_iters=1000, stability_threshold=1e-4)

check("default matches explicit state",
      result_default["state"] == result_explicit["state"])
check("default matches explicit ticks",
      result_default["convergence_ticks"] == result_explicit["convergence_ticks"])
if result_default["state"] == "CONVERGED":
    delta = np.abs(result_default["attractor"] - result_explicit["attractor"]).max()
    check("default matches explicit attractor", delta < 1e-10,
          f"max delta={delta}")

# Budget at default salience should produce these values
b_default = compute_attention_budget(SALIENCE_DEFAULT)
check("default budget max_iters=1000", b_default.max_iters == 1000,
      f"got {b_default.max_iters}")
check("default budget threshold=1e-4", abs(b_default.stability_threshold - 1e-4) < 1e-12,
      f"got {b_default.stability_threshold}")

# ── Test 7: High salience deeper attractor ────────────────────────────

print("\n7. High salience deeper attractor")

test_frame = hash_to_frame("attention depth test")

budget_low = compute_attention_budget(0.2)
budget_high = compute_attention_budget(0.9)

result_low = evolve_and_interpret(
    test_frame.copy(),
    max_iters=budget_low.max_iters,
    stability_threshold=budget_low.stability_threshold,
)
result_high = evolve_and_interpret(
    test_frame.copy(),
    max_iters=budget_high.max_iters,
    stability_threshold=budget_high.stability_threshold,
)

# High salience should use at least as many ticks (tighter threshold)
check("high salience >= ticks than low",
      result_high["convergence_ticks"] >= result_low["convergence_ticks"],
      f"high={result_high['convergence_ticks']}, low={result_low['convergence_ticks']}")

# Both should converge for this well-behaved input
check("low salience converges", result_low["state"] == "CONVERGED",
      f"got {result_low['state']}")
check("high salience converges", result_high["state"] == "CONVERGED",
      f"got {result_high['state']}")

# ── Test 8: Salience metadata stored ──────────────────────────────────

print("\n8. Salience metadata stored")

import tempfile
from wheeler_memory.rotation import store_with_rotation_retry

with tempfile.TemporaryDirectory() as tmpdir:
    result = store_with_rotation_retry(
        "salience metadata test input",
        data_dir=tmpdir,
        salience=0.9,
    )
    meta = result["metadata"]
    check("salience in metadata", "salience" in meta,
          f"keys: {list(meta.keys())}")
    check("salience value correct", meta.get("salience") == 0.9,
          f"got {meta.get('salience')}")
    check("attention_label in metadata", "attention_label" in meta)
    check("attention_label is high", meta.get("attention_label") == "high",
          f"got {meta.get('attention_label')}")
    check("stability_threshold in metadata", "stability_threshold" in meta)

# ── Summary ───────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
