#!/usr/bin/env python3
"""CLI: Sleep consolidation — prune redundant frames within bricks.

Usage:
    wheeler-sleep                      # consolidate all eligible memories
    wheeler-sleep --dry-run            # show what would be consolidated
    wheeler-sleep --chunk code         # consolidate specific chunk
    wheeler-sleep --stats              # show per-memory frame counts + potential savings
"""

import argparse
import sys

from wheeler_memory.consolidation import consolidation_stats, sleep_consolidate
from wheeler_memory.storage import DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Sleep consolidation: prune redundant frames within bricks"
    )
    parser.add_argument(
        "--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be consolidated without modifying bricks",
    )
    parser.add_argument(
        "--chunk", default=None, help="Consolidate only this chunk"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show per-memory frame counts and potential savings",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or DEFAULT_DATA_DIR

    # --- Stats mode ---
    if args.stats:
        stats = consolidation_stats(data_dir, chunk=args.chunk)
        if not stats:
            print("No memories found.")
            return

        print(
            f"{'Frames':>6} {'After':>5} {'Save':>5} "
            f"{'Temp':>6} {'Tier':<5} {'Cons':>4}  {'Chunk':<15} Text"
        )
        print("-" * 85)

        total_frames = 0
        total_potential = 0
        for s in stats:
            text_preview = (
                s["text"][:35] + "..." if len(s["text"]) > 35 else s["text"]
            )
            savings = s["frame_count"] - s["potential_frames"]
            cons = "yes" if s["consolidated"] else "no"
            print(
                f"{s['frame_count']:>6} {s['potential_frames']:>5} {savings:>5} "
                f"{s['temperature']:>6.4f} {s['tier']:<5} {cons:>4}  "
                f"{s['chunk']:<15} {text_preview}"
            )
            total_frames += s["frame_count"]
            total_potential += s["potential_frames"]

        total_savings = total_frames - total_potential
        print(f"\nTotal: {total_frames} frames -> {total_potential} frames "
              f"({total_savings} saveable, "
              f"{total_savings / total_frames * 100:.0f}%)" if total_frames > 0 else "")
        return

    # --- Consolidation mode ---
    result = sleep_consolidate(data_dir, dry_run=args.dry_run, chunk=args.chunk)

    prefix = "[DRY RUN] " if args.dry_run else ""

    if result.memories_consolidated:
        print(f"{prefix}Consolidated ({len(result.memories_consolidated)}):")
        for m in result.memories_consolidated:
            text_preview = (
                m["text"][:35] + "..." if len(m["text"]) > 35 else m["text"]
            )
            print(
                f"  {m['tier']:<5} {m['frames_before']:>3} -> {m['frames_after']:>3} frames  "
                f"{m['chunk']:<15} {text_preview}"
            )

    if result.memories_skipped:
        skip_reasons = {}
        for m in result.memories_skipped:
            reason = m["reason"]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        reasons_str = ", ".join(f"{v} {k}" for k, v in skip_reasons.items())
        print(f"{prefix}Skipped: {reasons_str}")

    if not result.memories_consolidated:
        print(f"{prefix}Nothing to consolidate.")

    if result.total_frames_before > 0:
        saved = result.total_frames_before - result.total_frames_after
        print(
            f"\nFrames: {result.total_frames_before} -> {result.total_frames_after} "
            f"({saved} pruned, {saved / result.total_frames_before * 100:.0f}%)"
        )


if __name__ == "__main__":
    main()
