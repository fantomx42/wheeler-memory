#!/usr/bin/env python3
"""CLI tool to display system hardware information and optimal device configuration."""

import json
import sys
from wheeler_memory.hardware import get_system_summary

def main():
    try:
        summary = get_system_summary()

        # Print formatted JSON
        print(json.dumps(summary, indent=2))

        # Highlight optimal device
        optimal = summary.get("optimal_device", "cpu")
        print(f"\n[ Wheeler Memory Auto-Config ]")
        print(f"Optimal Device Selected: \033[1;32m{optimal.upper()}\033[0m")

        # Print warnings if any
        warnings = summary.get("warnings", [])
        if warnings:
            print("\n\033[1;33m[ Warnings ]\033[0m")
            for warn in warnings:
                print(f"- {warn}")
    except ImportError as e:
        print(f"Error: Missing dependency — {e}\nRun: pip install -e .", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
