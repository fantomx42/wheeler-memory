"""Slow: attractor diversity validation.

Tests that Wheeler Memory attractors are genuinely distinct across diverse inputs —
the core correctness guarantee.
"""
import numpy as np
import pytest
from scipy.stats import pearsonr

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame


TEST_INPUTS = [
    "Fix authentication bug in login flow",
    "Deploy Kubernetes cluster on AWS",
    "Buy groceries: milk, eggs, bread",
    "Schedule dentist appointment for Thursday",
    "Quantum entanglement violates Bell inequalities",
    "The mitochondria is the powerhouse of the cell",
    "Review pull request #42 for memory leaks",
    "Plan birthday party for next Saturday",
    "Configure NGINX reverse proxy with TLS",
    "Water the garden every morning at 7am",
    "Implement binary search tree in Rust",
    "Book flight to Tokyo for March conference",
    "Dark matter comprises 27% of the universe",
    "Refactor database schema for multi-tenancy",
    "Practice piano scales for 30 minutes daily",
    "Debug segfault in GPU kernel launch",
    "Write unit tests for payment processing",
    "Organize closet by season and color",
    "Black holes emit Hawking radiation",
    "Compile FFmpeg with hardware acceleration",
]


@pytest.mark.slow
class TestAttractorDiversity:
    """Higher-level tests for attractor diversity and convergence."""

    def test_attractors_are_diverse(self):
        """Attractors from diverse inputs show low average off-diagonal correlation.

        Evolves all TEST_INPUTS, computes pairwise Pearson correlations on
        flattened attractors. Asserts:
          - avg off-diagonal correlation < 0.5
          - max off-diagonal correlation < 0.85
        """
        attractors = []
        for text in TEST_INPUTS:
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
            attractors.append(result["attractor"].flatten())

        n = len(attractors)
        correlations = []

        # Compute pairwise correlations
        for i in range(n):
            for j in range(i + 1, n):
                # Use scipy's pearsonr for robust correlation computation
                corr, _ = pearsonr(attractors[i], attractors[j])
                correlations.append(corr)

        correlations = np.array(correlations)

        # Check diversity properties
        avg_corr = np.mean(np.abs(correlations))
        max_corr = np.max(np.abs(correlations))

        assert avg_corr < 0.5, (
            f"Average attractor correlation {avg_corr:.4f} should be < 0.5 "
            f"(attractors are not diverse enough)"
        )
        assert max_corr < 0.85, (
            f"Max attractor correlation {max_corr:.4f} should be < 0.85 "
            f"(some attractors are too similar)"
        )

    def test_attractors_all_converge_or_oscillate(self):
        """All TEST_INPUTS produce state "CONVERGED" or "OSCILLATING", not "CHAOTIC".

        This is the core correctness guarantee: the CA should reach a stable
        attractor for semantically meaningful inputs.
        """
        states = []
        for text in TEST_INPUTS:
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
            states.append(result["state"])

        chaotic_count = sum(1 for s in states if s == "CHAOTIC")
        assert chaotic_count == 0, (
            f"Expected no CHAOTIC states, but got {chaotic_count}/{len(TEST_INPUTS)} "
            f"(states: {states})"
        )

        # All should be either CONVERGED or OSCILLATING
        for state in states:
            assert state in ("CONVERGED", "OSCILLATING"), (
                f"State '{state}' not in ('CONVERGED', 'OSCILLATING')"
            )

    def test_same_input_same_attractor(self):
        """Evolving the same input twice gives identical attractors.

        Tests determinism: seed→evolution→attractor is deterministic.
        """
        test_text = "Determinism test for Wheeler Memory"

        frame1 = hash_to_frame(test_text)
        result1 = evolve_and_interpret(frame1, max_iters=1000, stability_threshold=1e-4)
        attractor1 = result1["attractor"]

        frame2 = hash_to_frame(test_text)
        result2 = evolve_and_interpret(frame2, max_iters=1000, stability_threshold=1e-4)
        attractor2 = result2["attractor"]

        # Must be exactly identical (bit-for-bit)
        assert np.array_equal(attractor1, attractor2), (
            f"Same input produced different attractors on repeated evolution"
        )

        # Also check state consistency
        assert result1["state"] == result2["state"], (
            f"Same input produced different convergence states: "
            f"{result1['state']} vs {result2['state']}"
        )
