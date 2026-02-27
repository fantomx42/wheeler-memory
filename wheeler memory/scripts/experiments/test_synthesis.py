#!/usr/bin/env python3
"""Test: Novel Frame Synthesis (Apple Test).

- Run apple test on fruit domain with apple held out
- Verify verdict is one of {convergence, dissolution, hallucination}
- Log full trajectory regardless of outcome
- Run all three domains, record and print results table
"""

import sys

from wheeler_memory.theories.synthesis import apple_test, run_apple_battery


VALID_VERDICTS = {"convergence", "dissolution", "hallucination"}


def test_single_apple_test():
    """Run apple test on fruit domain."""
    print("\n--- Test: Single Apple Test (Fruits) ---")
    fruits = ["apple", "orange", "banana", "grape", "mango",
              "strawberry", "pear", "peach"]

    result = apple_test(fruits, holdout="apple")

    for line in result["log"]:
        print(f"  {line}")

    assert result["verdict"] in VALID_VERDICTS, \
        f"Invalid verdict: {result['verdict']}"
    assert result["holdout_attractor"] is not None, "Missing holdout attractor"
    assert result["candidate"] is not None, "Missing candidate"
    assert len(result["trajectory"]) > 0, "Empty trajectory"

    print(f"\n  Verdict: {result['verdict']}")
    print(f"  Correlation with candidate: {result['correlation']:.4f}")
    print(f"  Max neighbor correlation: {result['max_neighbor_correlation']:.4f}")
    print(f"  Trajectory length: {len(result['trajectory'])} steps")
    print(f"  Gaps found: {result['n_gaps']}")
    print("  PASS: Apple test produced valid verdict")


def test_apple_battery():
    """Run all three domain tests."""
    print("\n--- Test: Full Apple Battery ---")
    results = run_apple_battery()

    print("\n  Results Summary:")
    print(f"  {'Domain':<12} {'Verdict':<15} {'Candidate Corr':<16} {'Max Neighbor':<14} {'Gaps'}")
    print(f"  {'-'*12} {'-'*15} {'-'*16} {'-'*14} {'-'*5}")

    for domain, res in results.items():
        print(f"  {domain:<12} {res['verdict']:<15} {res['correlation']:<16.4f} "
              f"{res['max_neighbor_correlation']:<14.4f} {res['n_gaps']}")
        assert res["verdict"] in VALID_VERDICTS, \
            f"Invalid verdict for {domain}: {res['verdict']}"

    print("\n  PASS: All domains produced valid verdicts")


def main():
    print("=" * 60)
    print("Wheeler Theories — Synthesis (Apple Test) Tests")
    print("=" * 60)

    test_single_apple_test()
    test_apple_battery()

    print("\n" + "=" * 60)
    print("ALL SYNTHESIS TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
