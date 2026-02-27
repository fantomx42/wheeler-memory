"""CLI: Recall similar memories by Pearson correlation.

Usage:
    wheeler-recall "query text" --top-k 5
    wheeler-recall --chunk code "python bug"
    wheeler-recall --temperature-boost 0.1 "python bug"
"""

import argparse

from wheeler_memory import recall_memory
from wheeler_memory.attention import salience_from_label


def main():
    parser = argparse.ArgumentParser(description="Recall similar Wheeler memories")
    parser.add_argument("query", help="Query text to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--chunk", default=None, help="Search only this chunk (default: auto-select)")
    parser.add_argument(
        "--temperature-boost", type=float, default=0.0,
        help="Boost ranking by temperature (0.0 = no boost, default: 0.0)",
    )
    parser.add_argument(
        "--embed", action="store_true",
        help="Use sentence embedding for query (enables fuzzy semantic recall)",
    )
    parser.add_argument(
        "--reconstruct", action="store_true",
        help="Enable reconstructive recall (Darman architecture)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="Reconstruction weight: 0=pure memory, 1=pure query (default: 0.3)",
    )
    parser.add_argument(
        "--salience", choices=["low", "medium", "high"], default=None,
        help="Attention level: low (fast/loose), medium (default), high (deep/tight)",
    )
    parser.add_argument(
        "--trauma-mode", choices=["auto", "safe", "suppress"], default="auto",
        help="Trauma processing: auto (default), safe (therapy), suppress (skip)",
    )
    args = parser.parse_args()

    sal = salience_from_label(args.salience) if args.salience else None
    results = recall_memory(
        args.query,
        top_k=args.top_k,
        data_dir=args.data_dir,
        chunk=args.chunk,
        temperature_boost=args.temperature_boost,
        use_embedding=args.embed,
        reconstruct=args.reconstruct,
        reconstruct_alpha=args.alpha,
        salience=sal,
        trauma_mode=args.trauma_mode,
    )

    if not results:
        print("No memories stored yet.")
        return

    if args.reconstruct:
        print(f"{'Rank':<5} {'Sim':>6} {'r→stored':>9} {'r→query':>8} {'RState':<12} {'Chunk':<12} Text")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            text_preview = r["text"][:35] + "..." if len(r["text"]) > 35 else r["text"]
            chunk_name = r.get("chunk", "?")
            print(
                f"{i:<5} {r['similarity']:>6.3f} "
                f"{r.get('correlation_with_stored', 0):>9.4f} "
                f"{r.get('correlation_with_query', 0):>8.4f} "
                f"{r.get('reconstruction_state', '?'):<12} "
                f"{chunk_name:<12} {text_preview}"
            )
    else:
        print(f"{'Rank':<5} {'Similarity':>10}  {'Temp':>5} {'Tier':<5} {'Chunk':<12} {'State':<12} {'Ticks':>5}  Text")
        print("-" * 95)
        for i, r in enumerate(results, 1):
            text_preview = r["text"][:40] + "..." if len(r["text"]) > 40 else r["text"]
            if r.get("trauma_avoidance"):
                text_preview = f"[AVOIDANCE] {text_preview}"
            chunk_name = r.get("chunk", "?")
            print(
                f"{i:<5} {r['similarity']:>10.4f}  "
                f"{r['temperature']:>5.3f} {r['temperature_tier']:<5} "
                f"{chunk_name:<12} {r['state']:<12} {r['convergence_ticks']:>5}  {text_preview}"
            )


if __name__ == "__main__":
    main()
