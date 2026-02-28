"""wheeler-agent — interactive Wheeler Memory agent powered by Ollama/qwen3.

Usage
-----
    wheeler-agent "What do you know about the GPU backend?"
    wheeler-agent --interactive
    wheeler-agent --auto-memory --interactive      # full closed loop
    wheeler-agent --model qwen3:8b "Remember: the project uses ROCm for GPU."
    wheeler-agent --ollama http://192.168.1.10:11434 "Recall anything about sleep."
    wheeler-agent --verbose "Store three facts about cellular automata."

Memory loop flags:
    --auto-memory    Enable both auto-recall and auto-store (full closed loop).
    --auto-recall    Silently inject recalled memories as context before each turn.
    --auto-store     Silently store each agent reply as a new memory.
    --recall-k N     Number of memories to inject per turn (default: 3).

In interactive mode, type 'exit' or 'quit' to leave. Type '/reset' to clear
conversation history, '/history' to see the current exchange.
"""

from __future__ import annotations

import argparse
import sys

from wheeler_memory.agent import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, WheelerAgent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wheeler-agent",
        description="LLM agent with access to Wheeler Memory tools (via Ollama).",
    )
    p.add_argument(
        "message",
        nargs="?",
        help="Single message to process. Omit for --interactive mode.",
    )
    p.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive REPL mode.",
    )
    p.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Ollama model name (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--ollama",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Wheeler Memory data directory (default: ~/.wheeler_memory).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print tool call details.",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        metavar="N",
        help="Max tool-call rounds per turn (default: 10).",
    )
    p.add_argument(
        "--auto-memory",
        action="store_true",
        help="Enable both --auto-recall and --auto-store (full closed loop).",
    )
    p.add_argument(
        "--auto-recall",
        action="store_true",
        help="Inject recalled memories as context before each turn.",
    )
    p.add_argument(
        "--auto-store",
        action="store_true",
        help="Store each agent reply as a new memory after responding.",
    )
    p.add_argument(
        "--recall-k",
        type=int,
        default=3,
        metavar="N",
        help="Memories to inject per turn when --auto-recall is active (default: 3).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    do_recall = args.auto_recall or args.auto_memory
    do_store = args.auto_store or args.auto_memory

    agent = WheelerAgent(
        model=args.model,
        ollama_url=args.ollama,
        data_dir=args.data_dir,
        max_tool_rounds=args.max_rounds,
        auto_recall=do_recall,
        auto_store=do_store,
        auto_recall_k=args.recall_k,
        verbose=args.verbose,
    )

    if args.message and not args.interactive:
        # Single-shot mode
        try:
            reply = agent.run(args.message)
            print(reply)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.interactive or not args.message:
        # REPL mode
        print(f"Wheeler Agent  [model={args.model}  ollama={args.ollama}]")
        print("Type your message, or: /reset  /history  exit")
        print()
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input == "/reset":
                agent.reset()
                print("[history cleared]")
                continue
            if user_input == "/history":
                for turn in agent._history:
                    role = turn["role"].capitalize()
                    print(f"{role}: {turn['content'][:200]}")
                continue

            try:
                reply = agent.run(user_input)
                print(f"Agent: {reply}\n")
            except RuntimeError as exc:
                print(f"Error: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
