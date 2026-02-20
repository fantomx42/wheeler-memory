"""Attractor storage, indexing, and recall by Pearson correlation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .brick import MemoryBrick
from .chunking import (
    DEFAULT_CHUNK,
    get_chunk_dir,
    list_existing_chunks,
    select_chunk,
    select_recall_chunks,
    touch_chunk_metadata,
)
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex
from .temperature import (
    bump_access,
    compute_temperature,
    effective_temperature,
    ensure_access_fields,
    temperature_tier,
)
from .warming import (
    build_co_recall_associations,
    build_store_associations,
    load_warmth,
    propagate_warmth,
)

# Lazy import for embedding (optional dependency)
def _get_embed_to_frame():
    from .embedding import embed_to_frame
    return embed_to_frame

DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"


def _get_data_dir(data_dir: str | Path | None = None) -> Path:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_index(chunk_dir: Path, index: dict) -> None:
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def store_memory(
    text: str,
    result: dict,
    brick: MemoryBrick,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> str:
    """Save attractor, brick, and index entry for a memory.

    Returns the hex hash key used for storage.
    """
    d = _get_data_dir(data_dir)
    if chunk is None:
        chunk = select_chunk(text)

    chunk_dir = get_chunk_dir(d, chunk)
    hex_key = text_to_hex(text)

    np.save(chunk_dir / "attractors" / f"{hex_key}.npy", result["attractor"])
    brick.save(chunk_dir / "bricks" / f"{hex_key}.npz")

    index = _load_index(chunk_dir)
    now_iso = datetime.now(timezone.utc).isoformat()
    base_metadata = result.get("metadata", {})
    base_metadata["hit_count"] = 0
    base_metadata["last_accessed"] = now_iso
    index[hex_key] = {
        "text": text,
        "state": result["state"],
        "convergence_ticks": result["convergence_ticks"],
        "timestamp": now_iso,
        "metadata": base_metadata,
        "chunk": chunk,
    }
    _save_index(chunk_dir, index)
    touch_chunk_metadata(chunk_dir, stored=True)
    build_store_associations(chunk_dir, hex_key)
    return hex_key


def recall_memory(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    temperature_boost: float = 0.0,
    use_embedding: bool = False,
    reconstruct: bool = False,
    reconstruct_alpha: float = 0.3,
) -> list[dict]:
    """Recall stored memories by Pearson correlation with the query's attractor.

    Searches across matched chunks, merges results sorted by effective similarity.
    When temperature_boost > 0, hotter memories get a ranking bonus.
    When use_embedding is True, uses sentence embedding instead of SHA-256 hash
    for the query frame, enabling fuzzy semantic recall.
    When reconstruct is True, each recalled memory's attractor is blended with
    the query attractor and re-evolved through the CA (Darman architecture).
    """
    d = _get_data_dir(data_dir)

    if chunk is not None:
        chunks_to_search = [chunk]
    else:
        chunks_to_search = select_recall_chunks(text)
        # Also include any existing chunks not already selected
        existing = list_existing_chunks(d)
        for c in existing:
            if c not in chunks_to_search:
                chunks_to_search.append(c)

    if use_embedding:
        embed_fn = _get_embed_to_frame()
        query_frame = embed_fn(text)
    else:
        query_frame = hash_to_frame(text)
    query_result = evolve_and_interpret(query_frame)
    query_flat = query_result["attractor"].flatten()

    results = []
    for c in chunks_to_search:
        chunk_dir = d / "chunks" / c
        if not chunk_dir.exists():
            continue
        index = _load_index(chunk_dir)
        if not index:
            continue

        touch_chunk_metadata(chunk_dir)
        warmth_data = load_warmth(chunk_dir)

        for hex_key, meta in index.items():
            attractor_path = chunk_dir / "attractors" / f"{hex_key}.npy"
            if not attractor_path.exists():
                continue

            ensure_access_fields(meta, meta["timestamp"])

            attractor = np.load(attractor_path)
            corr, _ = pearsonr(query_flat, attractor.flatten())
            sim = float(corr)

            w = warmth_data.get(hex_key, {})
            temp = effective_temperature(
                meta["metadata"]["hit_count"],
                meta["metadata"]["last_accessed"],
                warmth_boost=w.get("boost", 0.0),
                warmth_applied_at=w.get("applied_at"),
            )
            tier = temperature_tier(temp)
            effective = sim + temperature_boost * temp

            results.append({
                "hex_key": hex_key,
                "text": meta["text"],
                "similarity": sim,
                "temperature": temp,
                "temperature_tier": tier,
                "effective_similarity": effective,
                "state": meta["state"],
                "convergence_ticks": meta["convergence_ticks"],
                "timestamp": meta["timestamp"],
                "chunk": c,
            })

    results.sort(key=lambda r: r["effective_similarity"], reverse=True)
    top_results = results[:top_k]

    # Reconstructive recall: blend each result with query context
    if reconstruct and top_results:
        from .reconstruction import reconstruct as _reconstruct
        query_att = query_result["attractor"]
        for r in top_results:
            # Load the stored attractor for reconstruction
            chunk_dir = d / "chunks" / r["chunk"]
            att_path = chunk_dir / "attractors" / f"{r['hex_key']}.npy"
            stored_att = np.load(att_path)
            recon = _reconstruct(stored_att, query_att, alpha=reconstruct_alpha)
            r["reconstructed_attractor"] = recon["attractor"]
            r["reconstruction_state"] = recon["state"]
            r["reconstruction_ticks"] = recon["convergence_ticks"]
            r["reconstruction_alpha"] = recon["alpha"]
            r["correlation_with_stored"] = recon["correlation_with_stored"]
            r["correlation_with_query"] = recon["correlation_with_query"]

    _bump_recalled_memories(d, top_results)

    return top_results


def _bump_recalled_memories(data_dir: Path, results: list[dict]) -> None:
    """Increment hit_count and update last_accessed for recalled memories."""
    # Group results by chunk to minimise index loads
    by_chunk: dict[str, list[str]] = {}
    for r in results:
        by_chunk.setdefault(r["chunk"], []).append(r["hex_key"])

    for chunk_name, hex_keys in by_chunk.items():
        chunk_dir = data_dir / "chunks" / chunk_name
        index = _load_index(chunk_dir)
        changed = False
        for hk in hex_keys:
            if hk in index:
                bump_access(index[hk])
                changed = True
        if changed:
            _save_index(chunk_dir, index)
        propagate_warmth(chunk_dir, hex_keys)
        if len(hex_keys) >= 2:
            build_co_recall_associations(chunk_dir, hex_keys)


def list_memories(
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> list[dict]:
    """List all stored memories from the index."""
    d = _get_data_dir(data_dir)

    if chunk is not None:
        chunks_to_list = [chunk]
    else:
        chunks_to_list = list_existing_chunks(d)

    all_memories = []
    for c in chunks_to_list:
        chunk_dir = d / "chunks" / c
        if not chunk_dir.exists():
            continue
        index = _load_index(chunk_dir)
        warmth_data = load_warmth(chunk_dir)
        for k, v in index.items():
            ensure_access_fields(v, v["timestamp"])
            w = warmth_data.get(k, {})
            temp = effective_temperature(
                v["metadata"]["hit_count"],
                v["metadata"]["last_accessed"],
                warmth_boost=w.get("boost", 0.0),
                warmth_applied_at=w.get("applied_at"),
            )
            all_memories.append({
                "hex_key": k,
                "chunk": c,
                "temperature": temp,
                "temperature_tier": temperature_tier(temp),
                **v,
            })

    return all_memories
