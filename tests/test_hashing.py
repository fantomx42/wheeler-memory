"""Unit tests for SHA-256 hashing and frame generation."""
import numpy as np
import pytest
from wheeler_memory.hashing import hash_to_frame, text_to_hex


class TestTextToHex:
    """Tests for text_to_hex function."""

    def test_text_to_hex_deterministic(self):
        """Same input produces same hex output twice."""
        hex1 = text_to_hex("test input")
        hex2 = text_to_hex("test input")
        assert hex1 == hex2

    def test_text_to_hex_different_inputs(self):
        """Different inputs produce different hex outputs."""
        hex1 = text_to_hex("test1")
        hex2 = text_to_hex("test2")
        assert hex1 != hex2

    def test_text_to_hex_is_hex_string(self):
        """Result is a 64-character lowercase hex string."""
        result = text_to_hex("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestHashToFrame:
    """Tests for hash_to_frame function."""

    def test_hash_to_frame_shape(self):
        """Returns 64x64 array."""
        frame = hash_to_frame("test")
        assert frame.shape == (64, 64)

    def test_hash_to_frame_dtype(self):
        """Array dtype is float32."""
        frame = hash_to_frame("test")
        assert frame.dtype == np.float32

    def test_hash_to_frame_range(self):
        """All values are in [-1, 1]."""
        frame = hash_to_frame("test")
        assert np.all(frame >= -1.0)
        assert np.all(frame <= 1.0)

    def test_hash_to_frame_deterministic(self):
        """Same input produces identical frames."""
        frame1 = hash_to_frame("test input")
        frame2 = hash_to_frame("test input")
        assert np.array_equal(frame1, frame2)

    def test_hash_to_frame_different_inputs(self):
        """Different texts produce different frames."""
        frame1 = hash_to_frame("test1")
        frame2 = hash_to_frame("test2")
        assert not np.array_equal(frame1, frame2)
