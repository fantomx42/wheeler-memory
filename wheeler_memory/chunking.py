"""Chunked memory routing — brain-inspired domain-specific storage.

Memories are routed to chunks like "code", "hardware", "daily_tasks" via
keyword substring matching.  Each chunk has its own attractors/, bricks/,
and index.json on disk under ~/.wheeler_memory/chunks/<name>/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

CHUNK_KEYWORDS: dict[str, list[str]] = {
    "code": [
        "python", "rust", "code", "bug", "debug", "compile", "function",
        "class", "import", "git", "commit", "api", "server", "deploy",
        "docker", "test", "refactor", "script", "variable", "error",
        "exception", "lint", "cargo", "npm", "pip", "branch", "merge",
        "syntax", "frontend", "backend", "database", "sql", "html", "css",
        "javascript", "typescript",
    ],
    "hardware": [
        "printer", "3d print", "solder", "circuit", "arduino", "raspberry",
        "gpio", "wire", "pcb", "resistor", "capacitor", "motor", "sensor",
        "voltage", "ampere", "oscilloscope", "multimeter", "firmware",
        "hardware", "cnc", "laser", "filament", "nozzle", "extruder",
        "bambu", "ender", "stepper",
    ],
    "daily_tasks": [
        "grocery", "groceries", "dentist", "doctor", "appointment",
        "schedule", "meeting", "call", "email", "buy", "pick up",
        "todo", "errand", "laundry", "clean", "cook", "dinner",
        "lunch", "breakfast", "workout", "exercise", "gym",
    ],
    "science": [
        "physics", "chemistry", "biology", "math", "equation", "theorem",
        "hypothesis", "experiment", "quantum", "relativity", "entropy",
        "molecule", "atom", "cell", "genome", "evolution", "neuron",
        "calculus", "algebra", "statistics", "probability",
    ],
    "meta": [
        "wheeler", "memory system", "attractor", "brick", "cellular automata",
        "ca dynamics", "rotation", "convergence", "oscillation", "chunk",
    ],
}

DEFAULT_CHUNK = "general"


def select_chunk(text: str) -> str:
    """Pick the single best chunk for storing *text*.

    Returns the chunk name with the most keyword hits, or DEFAULT_CHUNK
    when nothing matches.
    """
    lower = text.lower()
    best_chunk = DEFAULT_CHUNK
    best_hits = 0

    for chunk, keywords in CHUNK_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits = hits
            best_chunk = chunk

    return best_chunk


def select_recall_chunks(query: str, max_chunks: int = 3) -> list[str]:
    """Pick chunks to search when recalling *query*.

    Returns all matching chunks (up to *max_chunks*) plus "general".
    """
    lower = query.lower()
    scored: list[tuple[str, int]] = []

    for chunk, keywords in CHUNK_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > 0:
            scored.append((chunk, hits))

    scored.sort(key=lambda t: t[1], reverse=True)
    selected = [name for name, _ in scored[:max_chunks]]

    if DEFAULT_CHUNK not in selected:
        selected.append(DEFAULT_CHUNK)

    return selected


def get_chunk_dir(data_dir: Path, chunk: str) -> Path:
    """Return (and create) the directory subtree for *chunk*."""
    chunk_dir = data_dir / "chunks" / chunk
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "attractors").mkdir(exist_ok=True)
    (chunk_dir / "bricks").mkdir(exist_ok=True)
    return chunk_dir


def list_existing_chunks(data_dir: Path) -> list[str]:
    """Scan disk for populated chunk directories."""
    chunks_root = data_dir / "chunks"
    if not chunks_root.exists():
        return []
    return sorted(
        d.name for d in chunks_root.iterdir()
        if d.is_dir() and (d / "index.json").exists()
    )


def find_brick_across_chunks(hex_key: str, data_dir: Path) -> Path | None:
    """Search all chunks for a brick file matching *hex_key*."""
    chunks_root = data_dir / "chunks"
    if not chunks_root.exists():
        return None
    for chunk_dir in chunks_root.iterdir():
        if not chunk_dir.is_dir():
            continue
        brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"
        if brick_path.exists():
            return brick_path
    return None


def touch_chunk_metadata(chunk_dir: Path, stored: bool = False) -> None:
    """Update per-chunk stats (last access, store count)."""
    meta_path = chunk_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"created": datetime.now(timezone.utc).isoformat(), "store_count": 0}

    meta["last_accessed"] = datetime.now(timezone.utc).isoformat()
    if stored:
        meta["store_count"] = meta.get("store_count", 0) + 1

    meta_path.write_text(json.dumps(meta, indent=2))
