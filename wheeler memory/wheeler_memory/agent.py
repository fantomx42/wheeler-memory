"""Wheeler Memory agent loop driven by a local Ollama LLM (qwen3 / any tool-capable model).

The agent can store, recall, list, forget, and consolidate memories.
It runs in a tool-call loop: the LLM decides which operations to perform
until it produces a final plain-text response.

Usage
-----
>>> from wheeler_memory.agent import WheelerAgent
>>> agent = WheelerAgent()
>>> reply = agent.run("What do you remember about the project goals?")
>>> print(reply)

Or via the CLI:
    wheeler-agent "What do you remember about the project goals?"
    wheeler-agent --interactive          # REPL mode
    wheeler-agent --model qwen3:8b       # override model
    wheeler-agent --ollama http://host:11434
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from .storage import list_memories, recall_memory, store_memory
from .eviction import forget_by_text
from .consolidation import sleep_consolidate as _sleep_consolidate
from .hashing import text_to_hex
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame

DEFAULT_MODEL = "qwen3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

_SYSTEM_PROMPT = """\
You are a memory-augmented assistant backed by Wheeler Memory — a cellular
automata-based associative memory system. You have access to tools that let
you store, recall, list, forget, and consolidate memories.

Guidelines:
- When the user asks you to remember something, use store_memory.
- When the user asks what you know or remember about a topic, use recall_memory first.
- When the user asks to see all memories, use list_memories.
- When asked to forget something, use forget_memory.
- After a recall, you may store new inferences if they seem worth keeping.
- Keep your final responses concise and grounded in what the memories show.
"""

# ── Tool definitions (OpenAI-compatible format Ollama understands) ────────────

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": (
                "Store a new memory in Wheeler Memory. "
                "Use this to remember facts, observations, or information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to store as a memory.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Recall memories related to a query using associative "
                "similarity. Returns the top matching stored memories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search memories for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of memories to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List all stored memories with metadata (text, state, temperature).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to return (default 20).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Forget (delete) a specific memory by its exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text of the memory to forget.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sleep_consolidate",
            "description": (
                "Run sleep consolidation: compress and strengthen memories by "
                "pruning low-temperature bricks and merging similar attractors. "
                "Use when the memory store has grown large."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────────

def _exec_store_memory(text: str, data_dir: Path | None) -> str:
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    from .brick import MemoryBrick
    brick = MemoryBrick(
        evolution_history=result.get("history", []),
        final_attractor=result["attractor"],
        convergence_ticks=result["convergence_ticks"],
        state=result["state"],
        metadata=result.get("metadata", {}),
    )
    key = store_memory(text, result, brick, data_dir)
    return json.dumps({
        "stored": True,
        "key": key,
        "state": result["state"],
        "ticks": result["convergence_ticks"],
    })


def _exec_recall_memory(query: str, top_k: int, data_dir: Path | None) -> str:
    hits = recall_memory(query, top_k=top_k, data_dir=data_dir)
    if not hits:
        return json.dumps({"results": [], "message": "No memories found."})
    return json.dumps({
        "results": [
            {
                "text": h["text"],
                "similarity": round(h["similarity"], 4),
                "state": h.get("state", "?"),
                "temperature": round(h.get("temperature", 0.0), 3),
            }
            for h in hits
        ]
    })


def _exec_list_memories(limit: int, data_dir: Path | None) -> str:
    mems = list_memories(data_dir=data_dir)
    mems = mems[:limit]
    if not mems:
        return json.dumps({"memories": [], "message": "No memories stored."})
    return json.dumps({
        "count": len(mems),
        "memories": [
            {
                "text": m["text"],
                "state": m.get("state", "?"),
                "temperature": round(m.get("temperature", 0.0), 3),
                "timestamp": m.get("timestamp", ""),
            }
            for m in mems
        ],
    })


def _exec_forget_memory(text: str, data_dir: Path | None) -> str:
    result = forget_by_text(text, data_dir=data_dir)
    if result.forgotten:
        return json.dumps({"forgotten": True, "key": result.key})
    return json.dumps({"forgotten": False, "reason": result.reason})


def _exec_sleep_consolidate(data_dir: Path | None) -> str:
    result = _sleep_consolidate(data_dir=data_dir)
    return json.dumps({
        "pruned": result.pruned,
        "kept": result.kept,
        "merged": getattr(result, "merged", 0),
    })


def _dispatch_tool(name: str, args: dict, data_dir: Path | None) -> str:
    """Execute a tool call and return its JSON string result."""
    try:
        if name == "store_memory":
            return _exec_store_memory(args["text"], data_dir)
        if name == "recall_memory":
            return _exec_recall_memory(args["query"], args.get("top_k", 5), data_dir)
        if name == "list_memories":
            return _exec_list_memories(args.get("limit", 20), data_dir)
        if name == "forget_memory":
            return _exec_forget_memory(args["text"], data_dir)
        if name == "sleep_consolidate":
            return _exec_sleep_consolidate(data_dir)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Ollama HTTP client ────────────────────────────────────────────────────────

def _ollama_chat(
    messages: list[dict],
    model: str,
    tools: list[dict],
    base_url: str,
    stream: bool = False,
) -> dict:
    """POST /api/chat to Ollama and return the parsed response."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": stream,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. "
            "Is it running? Try: ollama serve"
        ) from exc


# ── Agent class ───────────────────────────────────────────────────────────────

class WheelerAgent:
    """LLM agent loop with access to Wheeler Memory tools.

    Parameters
    ----------
    model:
        Ollama model name (default "qwen3"). Any tool-capable model works.
    ollama_url:
        Base URL of the Ollama server (default "http://localhost:11434").
    data_dir:
        Wheeler Memory data directory. Uses ~/.wheeler_memory if None.
    max_tool_rounds:
        Safety limit on consecutive tool calls before forcing a response.
    auto_recall:
        If True, automatically recall related memories before each turn and
        inject them as context into the system prompt. Closes the read side
        of the memory loop without requiring an explicit tool call.
    auto_store:
        If True, automatically store each final reply as a new memory after
        the LLM responds. Closes the write side of the memory loop.
    auto_recall_k:
        Number of memories to inject when auto_recall is True (default 3).
    verbose:
        If True, print tool call and auto-memory info to stdout.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        data_dir: str | Path | None = None,
        max_tool_rounds: int = 10,
        auto_recall: bool = False,
        auto_store: bool = False,
        auto_recall_k: int = 3,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_tool_rounds = max_tool_rounds
        self.auto_recall = auto_recall
        self.auto_store = auto_store
        self.auto_recall_k = auto_recall_k
        self.verbose = verbose
        self._history: list[dict] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    # ── Auto-memory helpers ───────────────────────────────────────────────────

    def _build_recall_context(self, query: str) -> str | None:
        """Return a formatted memory-context block, or None if nothing recalled."""
        try:
            hits = recall_memory(query, top_k=self.auto_recall_k, data_dir=self.data_dir)
        except Exception:
            return None
        if not hits:
            return None
        lines = ["[Recalled memories (injected automatically):]"]
        for i, h in enumerate(hits, 1):
            sim = round(h["similarity"], 3)
            temp = round(h.get("temperature", 0.0), 2)
            lines.append(f"  {i}. \"{h['text']}\"  (similarity={sim}, temperature={temp})")
        return "\n".join(lines)

    def _auto_store_reply(self, text: str) -> None:
        """Store the agent reply as a Wheeler memory (best-effort, silent on error)."""
        try:
            _exec_store_memory(text, self.data_dir)
            if self.verbose:
                print(f"[auto-store] stored reply ({len(text)} chars)")
        except Exception as exc:
            if self.verbose:
                print(f"[auto-store] failed: {exc}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, user_message: str) -> str:
        """Process a user message and return the agent's final text reply.

        Internally runs the tool-call loop until the LLM produces a plain
        text response or max_tool_rounds is exhausted.

        If auto_recall is True, relevant memories are silently injected into
        the system context before the LLM sees the message.
        If auto_store is True, the final reply is silently stored as a memory.
        """
        # ── Auto-recall: inject relevant memories into system context ─────────
        system_content = _SYSTEM_PROMPT
        if self.auto_recall:
            ctx = self._build_recall_context(user_message)
            if ctx:
                system_content = _SYSTEM_PROMPT + "\n\n" + ctx
                if self.verbose:
                    print(ctx)

        self._history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system_content}] + self._history

        for _ in range(self.max_tool_rounds):
            resp = _ollama_chat(
                messages=messages,
                model=self.model,
                tools=_TOOLS,
                base_url=self.ollama_url,
            )
            msg = resp.get("message", {})
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                # Final text response
                content = msg.get("content", "")
                self._history.append({"role": "assistant", "content": content})
                messages.append({"role": "assistant", "content": content})
                # ── Auto-store: persist the reply as a new memory ─────────────
                if self.auto_store and content.strip():
                    self._auto_store_reply(content)
                return content

            # Execute each tool call
            messages.append({
                "role": "assistant",
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)

                if self.verbose:
                    print(f"[tool] {name}({json.dumps(args)})")

                result_str = _dispatch_tool(name, args, self.data_dir)

                if self.verbose:
                    print(f"[tool result] {result_str[:200]}")

                messages.append({
                    "role": "tool",
                    "content": result_str,
                })

        # Fallback: ask for a final answer without tools
        messages.append({
            "role": "user",
            "content": "(Please give your final answer now — no more tool calls needed.)",
        })
        resp = _ollama_chat(
            messages=messages,
            model=self.model,
            tools=[],
            base_url=self.ollama_url,
        )
        content = resp.get("message", {}).get("content", "")
        self._history.append({"role": "assistant", "content": content})
        if self.auto_store and content.strip():
            self._auto_store_reply(content)
        return content
