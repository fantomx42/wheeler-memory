"""CLI: Store text as a Wheeler Memory attractor.

Usage:
    wheeler-store "some text to remember"
    echo "piped text" | wheeler-store -
    wheeler-store --chunk code "fix the bug"
"""

import argparse
import sys

from wheeler_memory import store_with_rotation_retry
from wheeler_memory.attention import salience_from_label
from wheeler_memory.chunking import select_chunk


def _embed_store(text, chunk, data_dir, salience=None):
    """Store using embedding-based frame generation."""
    import time
    from wheeler_memory.attention import compute_attention_budget
    from wheeler_memory.embedding import embed_to_frame
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.storage import store_memory
    from wheeler_memory.temperature import SALIENCE_DEFAULT

    budget = compute_attention_budget(
        salience if salience is not None else SALIENCE_DEFAULT
    )

    start = time.time()
    frame = embed_to_frame(text)
    result = evolve_and_interpret(
        frame, max_iters=budget.max_iters,
        stability_threshold=budget.stability_threshold,
    )
    wall_time = time.time() - start

    result["metadata"]["rotation_used"] = 0
    result["metadata"]["attempts"] = 1
    result["metadata"]["wall_time_seconds"] = wall_time
    result["metadata"]["frame_mode"] = "embedding"
    result["metadata"]["salience"] = budget.salience
    result["metadata"]["attention_label"] = budget.label
    result["metadata"]["stability_threshold"] = budget.stability_threshold

    if result["state"] == "CONVERGED":
        brick = MemoryBrick.from_evolution_result(result, {"input_text": text})
        store_memory(text, result, brick, data_dir, chunk=chunk)

    return result


def main():
    parser = argparse.ArgumentParser(description="Store text as Wheeler Memory")
    parser.add_argument("text", help="Text to store (use '-' for stdin)")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--chunk", default=None, help="Target chunk (default: auto-route)")
    parser.add_argument("--embed", action="store_true", help="Use sentence embedding instead of SHA-256 hash")
    parser.add_argument(
        "--salience", choices=["low", "medium", "high"], default=None,
        help="Attention level: low (fast/loose), medium (default), high (deep/tight)",
    )
    parser.add_argument(
        "--trauma", action="store_true",
        help="Store as a traumatic memory (creates experience + avoidance attractors linked by avoidance_link edge)",
    )
    args = parser.parse_args()

    text = sys.stdin.read().strip() if args.text == "-" else args.text

    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    auto = args.chunk is None
    chunk = args.chunk if args.chunk else select_chunk(text)
    sal = salience_from_label(args.salience) if args.salience else None

    if args.trauma:
        from wheeler_memory.trauma import store_trauma
        trauma_result = store_trauma(
            text, data_dir=args.data_dir, chunk=chunk,
            use_embedding=args.embed, salience=sal,
        )
        exp = trauma_result["experience"]
        state = exp["state"]
        ticks = exp["convergence_ticks"]
        angle = exp["metadata"].get("rotation_used", 0)
        attempts = exp["metadata"].get("attempts", 1)
        wall = exp["metadata"].get("wall_time_seconds", 0)
        salience_label = exp["metadata"].get("attention_label", "medium")
        chunk_label = f"{chunk} (auto)" if auto else chunk
        print(f"Chunk:          {chunk_label}")
        print(f"State:          {state} (experience)")
        print(f"Avoidance:      {trauma_result['avoidance_state']}")
        print(f"Ticks:          {ticks}")
        print(f"Rotation:       {angle}° (attempt {attempts})")
        print(f"Salience:       {salience_label}")
        print(f"Time:           {wall:.3f}s")
        print(f"Experience key: {trauma_result['experience_hex']}")
        print(f"Avoidance key:  {trauma_result['avoidance_hex']}")
        if state == "CONVERGED":
            print("Trauma stored successfully (experience + avoidance attractors).")
        elif state == "FAILED_ALL_ROTATIONS":
            print("Warning: experience attractor failed to converge on all rotations.", file=sys.stderr)
        return

    if args.embed:
        result = _embed_store(text, chunk, args.data_dir, salience=sal)
    else:
        result = store_with_rotation_retry(
            text, data_dir=args.data_dir, chunk=chunk, salience=sal,
        )

    state = result["state"]
    ticks = result["convergence_ticks"]
    angle = result["metadata"].get("rotation_used", 0)
    attempts = result["metadata"].get("attempts", 1)
    wall = result["metadata"].get("wall_time_seconds", 0)

    salience_label = result["metadata"].get("attention_label", "medium")
    chunk_label = f"{chunk} (auto)" if auto else chunk
    print(f"Chunk:    {chunk_label}")
    print(f"State:    {state}")
    print(f"Ticks:    {ticks}")
    print(f"Rotation: {angle}° (attempt {attempts})")
    print(f"Salience: {salience_label}")
    print(f"Time:     {wall:.3f}s")

    if state == "CONVERGED":
        print("Memory stored successfully.")
    elif state == "FAILED_ALL_ROTATIONS":
        print("Warning: failed to converge on all rotations.", file=sys.stderr)


if __name__ == "__main__":
    main()
