#!/usr/bin/env python3
"""CLI: Dual-attractor trauma encoding — store, inspect, and treat trauma pairs.

Usage:
    wheeler-trauma store "the car crash" --avoidance "driving on highways"
    wheeler-trauma list
    wheeler-trauma status <pair_id>
    wheeler-trauma expose <pair_id>       # record safe exposure (therapy)
    wheeler-trauma remove <pair_id>       # unlink pair, keeps memories
"""

import argparse
import sys
from pathlib import Path

from wheeler_memory.storage import DEFAULT_DATA_DIR
from wheeler_memory.trauma import (
    list_trauma_pairs,
    record_trauma_activation,
    remove_trauma_pair,
    store_trauma_pair,
    therapy_status,
)


def cmd_store(args):
    result = store_trauma_pair(
        args.experience,
        args.avoidance,
        data_dir=args.data_dir or DEFAULT_DATA_DIR,
        chunk=args.chunk,
    )
    print(f"Trauma pair stored.")
    print(f"  Pair ID:     {result['pair_id']}")
    print(f"  Experience:  {result['experience_hex'][:12]}... (chunk: {result['experience_chunk']})")
    print(f"  Avoidance:   {result['avoidance_hex'][:12]}... (chunk: {result['avoidance_chunk']})")


def cmd_list(args):
    pairs = list_trauma_pairs(data_dir=args.data_dir or DEFAULT_DATA_DIR)
    if not pairs:
        print("No trauma pairs registered.")
        return

    print(f"{'ID':<14} {'Status':<12} {'Suppress':>8} {'Act':>4} {'Safe':>5}  Experience / Avoidance")
    print("-" * 90)
    for p in pairs:
        exp_preview = p["experience_text"][:25] + "..." if len(p["experience_text"]) > 25 else p["experience_text"]
        avoid_preview = p["avoidance_text"][:25] + "..." if len(p["avoidance_text"]) > 25 else p["avoidance_text"]
        print(
            f"{p['pair_id']:<14} {p['status']:<12} {p['suppression_strength']:>8.3f} "
            f"{p['activation_count']:>4} {p['safe_exposure_count']:>5}  "
            f"{exp_preview} / {avoid_preview}"
        )


def cmd_status(args):
    try:
        st = therapy_status(args.pair_id, data_dir=args.data_dir or DEFAULT_DATA_DIR)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Trauma Pair: {st['pair_id']}")
    print(f"  Status:              {st['status']}")
    print(f"  Suppression:         {st['suppression_strength']:.4f}")
    print(f"  Activations:         {st['activation_count']}")
    print(f"  Avoidance co-fires:  {st['avoidance_co_fire_count']}")
    print(f"  Safe exposures:      {st['safe_exposure_count']}")
    print(f"  Created:             {st['created']}")
    print(f"  Experience:          {st['experience']['text']}")
    print(f"    hex: {st['experience']['hex_key'][:16]}...  chunk: {st['experience']['chunk']}")
    print(f"  Avoidance:           {st['avoidance']['text']}")
    print(f"    hex: {st['avoidance']['hex_key'][:16]}...  chunk: {st['avoidance']['chunk']}")

    if st["therapy_history"]:
        print(f"\n  Therapy history ({len(st['therapy_history'])} entries):")
        for entry in st["therapy_history"][-5:]:
            print(f"    [{entry['timestamp'][:19]}] {entry['type']}  "
                  f"suppression={entry['suppression_before']:.4f}  "
                  f"safe_count={entry['safe_exposure_count']}")


def cmd_expose(args):
    try:
        pair = record_trauma_activation(
            args.pair_id,
            Path(args.data_dir or DEFAULT_DATA_DIR),
            safe_context=True,
        )
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    suppression = pair["suppression_strength"]
    safe_count = pair["safe_exposure_count"]
    threshold = 3

    if suppression <= 0.05:
        label = "resolved"
    elif suppression > 0.5:
        label = "active"
    else:
        label = "in_therapy"

    print(f"Safe exposure recorded for pair {args.pair_id}")
    print(f"  Safe exposures:  {safe_count}")
    print(f"  Suppression:     {suppression:.4f}")
    print(f"  Status:          {label}")

    if safe_count <= threshold:
        remaining = threshold - safe_count + 1
        print(f"  (Threshold not reached — {remaining} more safe exposure(s) before decay begins)")


def cmd_remove(args):
    removed = remove_trauma_pair(args.pair_id, data_dir=args.data_dir or DEFAULT_DATA_DIR)
    if removed:
        print(f"Trauma pair {args.pair_id} unlinked. Memories preserved.")
    else:
        print(f"Trauma pair {args.pair_id} not found.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Wheeler Memory: dual-attractor trauma encoding"
    )
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    sub = parser.add_subparsers(dest="command", required=True)

    # store
    p_store = sub.add_parser("store", help="Store a trauma pair (experience + avoidance)")
    p_store.add_argument("experience", help="Experience text (the traumatic event)")
    p_store.add_argument("--avoidance", required=True, help="Avoidance text (the avoidance response)")
    p_store.add_argument("--chunk", default=None, help="Force chunk for both memories")

    # list
    sub.add_parser("list", help="List all trauma pairs")

    # status
    p_status = sub.add_parser("status", help="Show therapy status for a pair")
    p_status.add_argument("pair_id", help="12-char pair ID")

    # expose
    p_expose = sub.add_parser("expose", help="Record a safe exposure (therapy)")
    p_expose.add_argument("pair_id", help="12-char pair ID")

    # remove
    p_remove = sub.add_parser("remove", help="Unlink a pair (keeps memories)")
    p_remove.add_argument("pair_id", help="12-char pair ID")

    args = parser.parse_args()

    commands = {
        "store": cmd_store,
        "list": cmd_list,
        "status": cmd_status,
        "expose": cmd_expose,
        "remove": cmd_remove,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
