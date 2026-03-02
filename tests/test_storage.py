"""Integration tests for attractor storage and recall."""
import sys
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame, text_to_hex
from wheeler_memory.storage import list_memories, recall_memory, store_memory

# Import conftest helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import store_test_memory


class TestStoreReturnsHexKey:
    """Test that store_memory returns a valid hex key."""

    def test_store_returns_hex_key(self, tmp_path):
        """store_test_memory returns a 64-char hex string."""
        key = store_test_memory("test storage 1", tmp_path)

        assert isinstance(key, str)
        assert len(key) == 64
        # All chars should be hex digits
        assert all(c in "0123456789abcdef" for c in key)


class TestStoreCreatesFiles:
    """Test that store_memory creates attractor and brick files."""

    def test_store_creates_attractor_file(self, tmp_path):
        """After store, .npy attractor file exists."""
        text = "test storage 2"
        key = store_test_memory(text, tmp_path)

        attractor_path = tmp_path / "chunks" / "general" / "attractors" / f"{key}.npy"
        assert attractor_path.exists()

    def test_store_creates_brick_file(self, tmp_path):
        """After store, .npz brick file exists."""
        text = "test storage 3"
        key = store_test_memory(text, tmp_path)

        brick_path = tmp_path / "chunks" / "general" / "bricks" / f"{key}.npz"
        assert brick_path.exists()

    def test_store_creates_index_entry(self, tmp_path):
        """After store, index.json contains the hex key."""
        text = "test storage 4"
        key = store_test_memory(text, tmp_path)

        index_path = tmp_path / "chunks" / "general" / "index.json"
        assert index_path.exists()

        import json
        index = json.loads(index_path.read_text())
        assert key in index
        assert index[key]["text"] == text


class TestRecallReturnsResults:
    """Test that recall_memory returns valid results."""

    def test_recall_returns_results(self, tmp_path):
        """Store a memory, recall with same text → results list non-empty."""
        text = "test storage 5"
        store_test_memory(text, tmp_path)

        results = recall_memory(text, top_k=5, data_dir=tmp_path)
        assert len(results) > 0

    def test_recall_top_result_matches_stored(self, tmp_path):
        """Top recall result hex_key matches stored key."""
        text = "test storage 6"
        key = store_test_memory(text, tmp_path)

        results = recall_memory(text, top_k=5, data_dir=tmp_path)
        assert len(results) > 0
        assert results[0]["hex_key"] == key

    def test_recall_result_has_required_keys(self, tmp_path):
        """Recall result dict has required keys."""
        text = "test storage 7"
        store_test_memory(text, tmp_path)

        results = recall_memory(text, top_k=5, data_dir=tmp_path)
        assert len(results) > 0

        result = results[0]
        required_keys = {"hex_key", "text", "similarity", "temperature"}
        assert required_keys.issubset(result.keys())

    def test_recall_similarity_in_bounds(self, tmp_path):
        """Recall similarity is in [-1, 1]."""
        text = "test storage 8"
        store_test_memory(text, tmp_path)

        results = recall_memory(text, top_k=5, data_dir=tmp_path)
        assert len(results) > 0

        for result in results:
            assert -1.0 <= result["similarity"] <= 1.0

    def test_recall_top_k_respected(self, tmp_path):
        """Recall with top_k=2 returns at most 2 results."""
        texts = ["test storage 9a", "test storage 9b", "test storage 9c", "test storage 9d"]
        for text in texts:
            store_test_memory(text, tmp_path)

        results = recall_memory("test storage", top_k=2, data_dir=tmp_path)
        assert len(results) <= 2


class TestListMemories:
    """Test list_memories functionality."""

    def test_list_memories_returns_stored(self, tmp_path):
        """Store 2 memories, list_memories returns both."""
        text1 = "test storage 10a"
        text2 = "test storage 10b"
        key1 = store_test_memory(text1, tmp_path)
        key2 = store_test_memory(text2, tmp_path)

        memories = list_memories(data_dir=tmp_path)
        keys = {m["hex_key"] for m in memories}

        assert key1 in keys
        assert key2 in keys

    def test_list_memories_has_temperature(self, tmp_path):
        """Each memory in list has temperature key."""
        text = "test storage 11"
        store_test_memory(text, tmp_path)

        memories = list_memories(data_dir=tmp_path)
        assert len(memories) > 0

        for mem in memories:
            assert "temperature" in mem


class TestStoreIdempotent:
    """Test that storing the same text twice is idempotent."""

    def test_store_idempotent(self, tmp_path):
        """Storing same text twice doesn't create duplicate index entries."""
        text = "test storage 12"
        key1 = store_test_memory(text, tmp_path)
        key2 = store_test_memory(text, tmp_path)

        # Both keys should be identical (same hash)
        assert key1 == key2

        # Index should have exactly one entry for this text
        import json
        index_path = tmp_path / "chunks" / "general" / "index.json"
        index = json.loads(index_path.read_text())
        assert len(index) == 1
        assert text in [v["text"] for v in index.values()]
