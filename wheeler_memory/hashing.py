"""Deterministic text-to-frame hashing using SHA-256."""

import hashlib
import numpy as np


def text_to_hex(text: str) -> str:
    """Return SHA-256 hex digest of text. Used as file/index key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_to_frame(text: str, size: int = 64) -> np.ndarray:
    """Convert text to a deterministic 64x64 frame via SHA-256 seeded RNG.

    Uses SHA-256 hash as seed for numpy PCG64 generator, then fills
    a size x size frame with uniform(-1, 1) values.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform(-1.0, 1.0, size=(size, size)).astype(np.float32)
