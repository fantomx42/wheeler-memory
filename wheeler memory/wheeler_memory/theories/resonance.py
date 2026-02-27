"""Corpus Resonance — query a directory of raw files using CA dynamics only.

No embeddings. No cosine similarity. Resonance is determined by whether
content moves Wheeler's CA dynamics toward or away from the active query frame.
Cost scales with query complexity, not corpus size.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from ..dynamics import apply_ca_dynamics, evolve_and_interpret
from ..hashing import hash_to_frame


@dataclass
class ResonanceResult:
    total_chunks: int = 0
    resonant_chunks: int = 0
    skipped_chunks: int = 0
    resonant_files: list[dict] = field(default_factory=list)  # [{path, chunk_text, resonance_score}]
    cost_ratio: float = 0.0  # resonant / total


def _is_text_file(path: Path) -> bool:
    """Quick check if a file is likely text."""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            f.read(512)
        return True
    except (UnicodeDecodeError, PermissionError, IsADirectoryError):
        return False


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _detect_resonance(
    chunk_frame: np.ndarray,
    query_attractor_flat: np.ndarray,
    early_exit_ticks: int,
) -> tuple[bool, float]:
    """Run CA steps and check if correlation with query is increasing.

    Returns (is_resonant, final_correlation).
    """
    state = chunk_frame.copy()
    correlations = []

    for _ in range(early_exit_ticks):
        state = apply_ca_dynamics(state)
        corr, _ = pearsonr(state.flatten(), query_attractor_flat)
        correlations.append(float(corr))

    if len(correlations) < 4:
        return False, correlations[-1] if correlations else 0.0

    # Check if correlation is trending upward over the window
    # Compare first quarter average to last quarter average
    q = len(correlations) // 4
    first_quarter = np.mean(correlations[:q])
    last_quarter = np.mean(correlations[-q:])
    trend = last_quarter - first_quarter

    return trend > 0, correlations[-1]


def query_corpus(
    query: str,
    corpus_dir: Path,
    chunk_size: int = 512,
    threshold: float = 0.1,
    early_exit_ticks: int = 50,
) -> ResonanceResult:
    """Query a corpus directory using CA dynamics resonance.

    1. Seed query frame, evolve to attractor
    2. For each file chunk: seed frame, run early_exit_ticks CA steps
       - If correlation moves TOWARD query attractor -> resonant
       - If flat or decreasing -> skip
    3. Return only resonant chunks

    No embeddings. No cosine similarity. Resonance = CA dynamics only.
    """
    corpus_dir = Path(corpus_dir)
    result = ResonanceResult()

    # Evolve query to attractor
    query_frame = hash_to_frame(query)
    query_result = evolve_and_interpret(query_frame)
    query_att_flat = query_result["attractor"].flatten()

    # Walk corpus
    for file_path in sorted(corpus_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if not _is_text_file(file_path):
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        chunks = _chunk_text(text, chunk_size)

        for chunk_text in chunks:
            result.total_chunks += 1
            chunk_frame = hash_to_frame(chunk_text)

            is_resonant, final_corr = _detect_resonance(
                chunk_frame, query_att_flat, early_exit_ticks
            )

            if is_resonant:
                result.resonant_chunks += 1
                result.resonant_files.append({
                    "path": str(file_path),
                    "chunk_text": chunk_text[:200],  # Truncate for log
                    "resonance_score": final_corr,
                })
            else:
                result.skipped_chunks += 1

    if result.total_chunks > 0:
        result.cost_ratio = result.resonant_chunks / result.total_chunks

    return result
