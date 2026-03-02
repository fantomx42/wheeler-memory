"""Embedding-based text-to-frame conversion using sentence-transformers.

Instead of SHA-256 hashing (which destroys semantic similarity), this module
uses a sentence embedding model to convert text to a 384-dim vector, then
projects it into a 64×64 CA frame via a fixed random projection matrix.

This means similar text → similar frames → similar attractors, enabling
fuzzy recall through Wheeler Memory's Pearson correlation search.
"""

import hashlib
import numpy as np

# Lazy-loaded model
_model = None
_projection_matrix = None

# Model config
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
FRAME_SIZE = 64
FRAME_CELLS = FRAME_SIZE * FRAME_SIZE  # 4096
PROJECTION_SEED = 0xDEAD_BEEF  # fixed seed for reproducible projection


def embed_available() -> bool:
    """Check if sentence-transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_model():
    """Lazy-load the sentence transformer model (cached after first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from .hardware import get_optimal_device
        
        device = get_optimal_device()
        print(f"Loading embedding model on {device}...")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def _get_projection_matrix() -> np.ndarray:
    """Return the fixed 384→4096 random projection matrix.

    Uses Johnson-Lindenstrauss random projection to approximately
    preserve cosine distances from embedding space to frame space.
    The matrix is generated once from a fixed seed and cached.
    """
    global _projection_matrix
    if _projection_matrix is None:
        rng = np.random.Generator(np.random.PCG64(PROJECTION_SEED))
        # Gaussian random projection preserves distances
        _projection_matrix = rng.standard_normal(
            (EMBED_DIM, FRAME_CELLS)
        ).astype(np.float32) / np.sqrt(FRAME_CELLS)
    return _projection_matrix


def embed_text(text: str) -> np.ndarray:
    """Convert text to a 384-dim sentence embedding vector."""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.astype(np.float32)


def embed_to_frame(text: str, size: int = FRAME_SIZE) -> np.ndarray:
    """Convert text to a 64×64 CA frame via sentence embedding + projection.

    1. Encode text → 384-dim embedding (via sentence-transformers)
    2. Project 384 → 4096 via fixed random matrix (preserves distances)
    3. Apply tanh to map to (-1, 1) range
    4. Reshape to 64×64

    Similar text produces similar frames, enabling fuzzy CA recall.
    """
    embedding = embed_text(text)
    proj = _get_projection_matrix()

    # Project: (384,) @ (384, 4096) → (4096,)
    frame_flat = embedding @ proj

    # Scale to make tanh spread values well across (-1, 1)
    # The raw projection has std ≈ 1/sqrt(4096) * sqrt(384) ≈ 0.3
    # Multiply by ~3 so tanh covers most of its range
    frame_flat = np.tanh(frame_flat * 3.0)

    return frame_flat.reshape(size, size).astype(np.float32)


def embed_text_batch(texts: list[str]) -> np.ndarray:
    """Batch-encode multiple texts to embeddings. Returns (N, 384)."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def embed_to_frame_batch(texts: list[str], size: int = FRAME_SIZE) -> list[np.ndarray]:
    """Batch-convert texts to CA frames via embedding + projection."""
    embeddings = embed_text_batch(texts)  # (N, 384)
    proj = _get_projection_matrix()        # (384, 4096)

    frames_flat = embeddings @ proj        # (N, 4096)
    frames_flat = np.tanh(frames_flat * 3.0)

    return [
        frames_flat[i].reshape(size, size).astype(np.float32)
        for i in range(len(texts))
    ]


def text_to_embed_hex(text: str) -> str:
    """Stable file key for an embedded text.

    Uses SHA-256 of the embedding vector bytes so that:
    - Same text always gets the same key
    - Different text gets different keys
    - Independent of the text's own hash
    """
    embedding = embed_text(text)
    return hashlib.sha256(embedding.tobytes()).hexdigest()
