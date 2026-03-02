"""Slow: reconstructive recall (Darman architecture).

Tests the reconstructive recall mechanism where stored and query attractors
are blended and re-evolved through the CA to produce context-dependent
reconstructions.
"""
import numpy as np
import pytest

from wheeler_memory.reconstruction import reconstruct
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame


@pytest.mark.slow
class TestReconstruction:
    """Higher-level tests for reconstructive recall."""

    def test_reconstruct_returns_keys(self):
        """Result dict has required keys.

        The reconstruct() function should return:
          - "attractor": the reconstructed 64x64 frame
          - "state": convergence state of reconstruction
          - "convergence_ticks": number of iterations to re-converge
          - "alpha": the blending weight used
          - "correlation_with_stored": Pearson correlation to stored attractor
          - "correlation_with_query": Pearson correlation to query attractor
        """
        stored_frame = hash_to_frame("stored memory about Python libraries")
        query_frame = hash_to_frame("query about machine learning")

        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        query_result = evolve_and_interpret(query_frame, max_iters=1000)

        stored_attractor = stored_result["attractor"]
        query_attractor = query_result["attractor"]

        result = reconstruct(stored_attractor, query_attractor, alpha=0.3)

        # Check all required keys are present
        required_keys = {
            "attractor",
            "state",
            "convergence_ticks",
            "alpha",
            "correlation_with_stored",
            "correlation_with_query",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys in reconstruction result. "
            f"Expected {required_keys}, got {set(result.keys())}"
        )

    def test_reconstruct_attractor_shape(self):
        """Reconstructed attractor is 64×64."""
        stored_frame = hash_to_frame("stored memory")
        query_frame = hash_to_frame("query context")

        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        query_result = evolve_and_interpret(query_frame, max_iters=1000)

        result = reconstruct(
            stored_result["attractor"],
            query_result["attractor"],
            alpha=0.3,
        )

        assert result["attractor"].shape == (64, 64), (
            f"Expected shape (64, 64), got {result['attractor'].shape}"
        )

    def test_reconstruct_attractor_range(self):
        """Reconstructed attractor values are in [-1, 1]."""
        stored_frame = hash_to_frame("stored memory")
        query_frame = hash_to_frame("query context")

        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        query_result = evolve_and_interpret(query_frame, max_iters=1000)

        result = reconstruct(
            stored_result["attractor"],
            query_result["attractor"],
            alpha=0.3,
        )

        attractor = result["attractor"]
        assert np.all(attractor >= -1.0), (
            f"Some values < -1.0: min={attractor.min()}"
        )
        assert np.all(attractor <= 1.0), (
            f"Some values > 1.0: max={attractor.max()}"
        )

    def test_reconstruct_correlation_bounds(self):
        """Both correlation_with_stored and correlation_with_query are in [-1, 1]."""
        stored_frame = hash_to_frame("stored memory")
        query_frame = hash_to_frame("query context")

        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        query_result = evolve_and_interpret(query_frame, max_iters=1000)

        result = reconstruct(
            stored_result["attractor"],
            query_result["attractor"],
            alpha=0.3,
        )

        corr_stored = result["correlation_with_stored"]
        corr_query = result["correlation_with_query"]

        assert -1.0 <= corr_stored <= 1.0, (
            f"correlation_with_stored {corr_stored} not in [-1, 1]"
        )
        assert -1.0 <= corr_query <= 1.0, (
            f"correlation_with_query {corr_query} not in [-1, 1]"
        )

    def test_reconstruct_memory_dominant(self):
        """With default alpha=0.3 (memory-dominant): correlation_with_stored > correlation_with_query.

        The reconstruction should be biased toward the stored memory when alpha < 0.5.
        """
        stored_frame = hash_to_frame("stored memory about Python libraries")
        query_frame = hash_to_frame("query about web development (very different)")

        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        query_result = evolve_and_interpret(query_frame, max_iters=1000)

        result = reconstruct(
            stored_result["attractor"],
            query_result["attractor"],
            alpha=0.3,  # Memory-dominant
        )

        corr_stored = result["correlation_with_stored"]
        corr_query = result["correlation_with_query"]

        assert corr_stored > corr_query, (
            f"With alpha=0.3 (memory-dominant), expected correlation_with_stored "
            f"({corr_stored:.4f}) > correlation_with_query ({corr_query:.4f})"
        )

    def test_reconstruct_query_dominant(self):
        """With alpha=0.7 (query-dominant): correlation_with_query > correlation_with_stored.

        The reconstruction should be biased toward the query when alpha > 0.5.
        """
        stored_frame = hash_to_frame("stored memory about Python libraries")
        query_frame = hash_to_frame("query about web development")

        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        query_result = evolve_and_interpret(query_frame, max_iters=1000)

        result = reconstruct(
            stored_result["attractor"],
            query_result["attractor"],
            alpha=0.7,  # Query-dominant
        )

        corr_stored = result["correlation_with_stored"]
        corr_query = result["correlation_with_query"]

        assert corr_query > corr_stored, (
            f"With alpha=0.7 (query-dominant), expected correlation_with_query "
            f"({corr_query:.4f}) > correlation_with_stored ({corr_stored:.4f})"
        )

    def test_reconstruct_context_dependent(self):
        """Same stored attractor + two different queries → different reconstructions.

        The Darman architecture: reconstruction varies with query context.
        The same stored memory reconstructs differently when blended with
        different query contexts.
        """
        # One stored memory
        stored_frame = hash_to_frame("stored memory: Python has great libraries")
        stored_result = evolve_and_interpret(stored_frame, max_iters=1000)
        stored_attractor = stored_result["attractor"]

        # Two very different queries
        query_frame_a = hash_to_frame("query context A: machine learning and data science")
        query_result_a = evolve_and_interpret(query_frame_a, max_iters=1000)

        query_frame_b = hash_to_frame("query context B: web development and REST APIs")
        query_result_b = evolve_and_interpret(query_frame_b, max_iters=1000)

        # Reconstruct stored memory with both queries
        recon_a = reconstruct(stored_attractor, query_result_a["attractor"], alpha=0.3)
        recon_b = reconstruct(stored_attractor, query_result_b["attractor"], alpha=0.3)

        # The reconstructions should be different
        assert not np.array_equal(recon_a["attractor"], recon_b["attractor"]), (
            "Expected different reconstructions for same stored memory + different queries. "
            "The Darman architecture requires context-dependent reconstruction."
        )
