#!/usr/bin/env python3
"""CLI: Evict cold memories or forget specific ones.

Usage:
    wheeler-forget                      # full sweep (fade + evict + capacity)
    wheeler-forget --dry-run            # show what would happen
    wheeler-forget --text "some memory" # forget a specific memory
    wheeler-forget --hex abc123...      # forget by hex key
    wheeler-forget --coldest 10         # diagnostic: show 10 coldest memories
"""

import argparse
import sys

from wheeler_memory.eviction import (
    forget_by_text,
    forget_memory,
    score_memories,
    sweep_and_evict,
)
from wheeler_memory.storage import DEFAULT_DATA_DIR
from wheeler_memory.temperature import temperature_tier_detailed


def main():
    parser = argparse.ArgumentParser(description="Evict cold memories or forget specific ones")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without deleting")
    parser.add_argument("--text", default=None, help="Forget a specific memory by text")
    parser.add_argument("--hex", default=None, help="Forget a specific memory by hex key")
    parser.add_argument("--coldest", type=int, default=None, metavar="N", help="Show N coldest memories")
    args = parser.parse_args()

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    try:
        _run(args, data_dir)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run(args, data_dir):
    # --- Targeted forget by text ---
    if args.text is not None:
        found = forget_by_text(args.text, data_dir)
        if found:
            print(f"Forgotten: {args.text[:40]}")
        else:
            print(f"Not found: {args.text[:40]}")
            sys.exit(1)
        return

    # --- Targeted forget by hex key ---
    if args.hex is not None:
        found = forget_memory(args.hex, data_dir)
        if found:
            print(f"Forgotten: {args.hex[:16]}...")
        else:
            print(f"Not found: {args.hex[:16]}...")
            sys.exit(1)
        return

    # --- Diagnostic: show coldest ---
    if args.coldest is not None:
        scored = score_memories(data_dir)
        n = min(args.coldest, len(scored))
        if n == 0:
            print("No memories found.")
            return
        print(f"{'Temp':>6} {'Tier':<7} {'Hits':>5} {'Age':>6} {'Chunk':<15} Text")
        print("-" * 78)
        for m in scored[:n]:
            tier = temperature_tier_detailed(m["temperature"])
            text_preview = m["text"][:40] + "..." if len(m["text"]) > 40 else m["text"]
            print(f"{m['temperature']:>6.4f} {tier:<7} {m['hit_count']:>5} {m['age_days']:>5.1f}d {m['chunk']:<15} {text_preview}")
        return

    # --- Full sweep ---
    result = sweep_and_evict(data_dir, dry_run=args.dry_run)

    prefix = "[DRY RUN] " if args.dry_run else ""

    if result.bricks_deleted:
        print(f"{prefix}Bricks faded ({len(result.bricks_deleted)}):")
        for m in result.bricks_deleted:
            text_preview = m["text"][:40] + "..." if len(m["text"]) > 40 else m["text"]
            print(f"  {m['temperature']:.4f}  {m['chunk']:<15} {text_preview}")

    if result.memories_evicted:
        print(f"{prefix}Memories evicted ({len(result.memories_evicted)}):")
        for m in result.memories_evicted:
            text_preview = m["text"][:40] + "..." if len(m["text"]) > 40 else m["text"]
            print(f"  {m['temperature']:.4f}  {m['chunk']:<15} {text_preview}")

    if not result.bricks_deleted and not result.memories_evicted:
        print(f"{prefix}Nothing to evict.")

    print(f"\nTotal: {result.total_before} → {result.total_after}")


if __name__ == "__main__":
    main()
