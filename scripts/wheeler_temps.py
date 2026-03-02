"""CLI: Inspect temperature distribution across Wheeler memories.

Usage:
    wheeler-temps                     # all chunks
    wheeler-temps --chunk code        # specific chunk
    wheeler-temps --tier hot          # filter by tier
    wheeler-temps --sort temp         # sort by temperature (default)
"""

import argparse
import sys

from wheeler_memory import list_memories


def main():
    parser = argparse.ArgumentParser(description="Inspect Wheeler memory temperatures")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--chunk", default=None, help="Show only this chunk")
    parser.add_argument("--tier", default=None, choices=["hot", "warm", "cold"], help="Filter by tier")
    parser.add_argument(
        "--sort", default="temp", choices=["temp", "hits", "chunk"],
        help="Sort order (default: temp)",
    )
    args = parser.parse_args()

    try:
        memories = list_memories(data_dir=args.data_dir, chunk=args.chunk)
    except FileNotFoundError:
        print('No memories found. Store something first with: wheeler-store "your text"')
        return
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.tier:
        memories = [m for m in memories if m["temperature_tier"] == args.tier]

    if args.sort == "temp":
        memories.sort(key=lambda m: m["temperature"], reverse=True)
    elif args.sort == "hits":
        memories.sort(key=lambda m: m["metadata"]["hit_count"], reverse=True)
    elif args.sort == "chunk":
        memories.sort(key=lambda m: m["chunk"])

    hot = sum(1 for m in memories if m["temperature_tier"] == "hot")
    warm = sum(1 for m in memories if m["temperature_tier"] == "warm")
    cold = sum(1 for m in memories if m["temperature_tier"] == "cold")

    print(f"Memories: {len(memories)}  |  hot: {hot}  warm: {warm}  cold: {cold}")
    print()

    if not memories:
        return

    print(f"{'Temp':>6} {'Tier':<5} {'Hits':>5} {'Chunk':<15} Text")
    print("-" * 70)
    for m in memories:
        hits = m["metadata"]["hit_count"]
        text_preview = m["text"][:40] + "..." if len(m["text"]) > 40 else m["text"]
        print(f"{m['temperature']:>6.3f} {m['temperature_tier']:<5} {hits:>5} {m['chunk']:<15} {text_preview}")


if __name__ == "__main__":
    main()
