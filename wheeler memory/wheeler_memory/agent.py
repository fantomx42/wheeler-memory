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
You are Darman, an AI assistant with reconstructive memory.

Your memory is not perfect. It works like human memory — imperfect,
associative, and shaped by context. Memories are stored as stable patterns
that decay over time if not revisited.

MEMORY CONTEXT will be injected before your response when relevant memories
exist. It contains memories grouped by confidence:
- Strong (HOT): High confidence — recently or frequently accessed.
- Moderate (WARM): Medium confidence — may have drifted somewhat.
- Faint (COLD): Low confidence — significant drift likely, treat with caution.

IMPORTANT RULES:
- Memories are SUGGESTIONS, not commands. You may disagree if current context
  warrants it.
- If a memory is cold/faint, acknowledge the uncertainty explicitly.
- If no relevant memory exists, say so honestly — do not fabricate.
- When the user asks you to remember something, use store_memory.
- When the user asks what you know about a topic, use recall_memory.
- When the user asks to see all memories, use list_memories.
- When asked to forget something, use forget_memory.
- When the user asks you to look something up or asks about current events,
  use web_search.
- After a recall, store new inferences if they seem worth keeping.
- When a recalled memory has a polar companion firing, you can use
  polar_decay to reduce the polarity link weight.
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
    {
        "type": "function",
        "function": {
            "name": "polar_decay",
            "description": (
                "Recall with polar decay to reduce polarity link weight. "
                "Each call increments decay_count on any polarity link that "
                "fires, decaying its weight by 0.7x. Use when a recalled "
                "memory has a polar companion and the user wants to neutralize "
                "the association."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The query text to recall with polar decay.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of memories to return (default 5).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Use this when the user asks "
                "you to look something up, find current information, or asks "
                "about topics you may not have in memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
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


def _exec_polar_decay(text: str, top_k: int, data_dir: Path | None) -> str:
    hits = recall_memory(text, top_k=top_k, data_dir=data_dir, polar_decay=True)
    if not hits:
        return json.dumps({"results": [], "message": "No memories found."})
    return json.dumps({
        "results": [
            {
                "text": h["text"],
                "similarity": round(h["similarity"], 4),
                "polar_firing": h.get("polar_firing"),
            }
            for h in hits
        ]
    })


def _exec_sleep_consolidate(data_dir: Path | None) -> str:
    result = _sleep_consolidate(data_dir=data_dir)
    return json.dumps({
        "pruned": result.pruned,
        "kept": result.kept,
        "merged": getattr(result, "merged", 0),
    })


def _exec_web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo Lite and return structured results (no API key needed)."""
    import urllib.parse
    from html.parser import HTMLParser

    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Wheeler-Memory/0.1)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return json.dumps({"error": f"Search failed: {exc}", "query": query})

    class _DDGParser(HTMLParser):
        """Parse DDG Lite result rows: each result is a <tr> with link + snippet cells."""
        def __init__(self):
            super().__init__()
            self.results: list[dict] = []
            self._in_link = False
            self._in_snippet = False
            self._cur: dict = {}

        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            cls = d.get("class", "")
            if tag == "a" and "result-link" in cls:
                self._in_link = True
                href = d.get("href", "")
                if href.startswith("//"):
                    href = "https:" + href
                self._cur["url"] = href
            elif tag == "td" and "result-snippet" in cls:
                self._in_snippet = True
                self._cur.setdefault("snippet", "")

        def handle_data(self, data):
            if self._in_link:
                self._cur["title"] = self._cur.get("title", "") + data
            elif self._in_snippet:
                self._cur["snippet"] = self._cur.get("snippet", "") + data

        def handle_endtag(self, tag):
            if tag == "a" and self._in_link:
                self._in_link = False
                if self._cur.get("title") and self._cur.get("url"):
                    self.results.append(dict(self._cur))
                    self._cur = {}
            elif tag == "td" and self._in_snippet:
                self._in_snippet = False

    parser = _DDGParser()
    parser.feed(html)
    results = [
        {
            "title": r.get("title", "").strip(),
            "url": r.get("url", ""),
            "snippet": r.get("snippet", "").strip(),
        }
        for r in parser.results
        if r.get("title") and r.get("url")
    ][:max_results]

    return json.dumps({"query": query, "results": results})


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
        if name == "polar_decay":
            return _exec_polar_decay(args["text"], args.get("top_k", 5), data_dir)
        if name == "web_search":
            return _exec_web_search(args["query"], args.get("max_results", 5))
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


def _ollama_chat_stream(
    messages: list[dict],
    model: str,
    tools: list[dict],
    base_url: str,
):
    """POST /api/chat with stream=True and yield each parsed JSON chunk.

    Ollama streams newline-delimited JSON objects. Each chunk has the shape:
        {"message": {"role": "assistant", "content": "<token>"}, "done": false}
    The final chunk has done=true and may include tool_calls in message.
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
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
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                yield chunk
                if chunk.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. "
            "Is it running? Try: ollama serve"
        ) from exc


class _ThinkingFilter:
    """Stateful filter that separates <think>...</think> blocks from regular tokens.

    Call process(piece) for each incoming content chunk; it returns a list of
    (event_type, content) tuples where event_type is "thinking" or "token".
    Call flush() at the end to drain any buffered content.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def process(self, piece: str) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        self._buf += piece
        while self._buf:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx >= 0:
                    events.append(("thinking", self._buf[:idx]))
                    self._buf = self._buf[idx + len(self._CLOSE):]
                    self._in_think = False
                else:
                    # Keep last (len(CLOSE)-1) chars in case tag spans chunks
                    keep = len(self._CLOSE) - 1
                    if len(self._buf) > keep:
                        events.append(("thinking", self._buf[:-keep]))
                        self._buf = self._buf[-keep:]
                    break
            else:
                idx = self._buf.find(self._OPEN)
                if idx >= 0:
                    if idx > 0:
                        events.append(("token", self._buf[:idx]))
                    self._buf = self._buf[idx + len(self._OPEN):]
                    self._in_think = True
                else:
                    keep = len(self._OPEN) - 1
                    if len(self._buf) > keep:
                        events.append(("token", self._buf[:-keep]))
                        self._buf = self._buf[-keep:]
                    break
        return events

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        kind = "thinking" if self._in_think else "token"
        events = [(kind, self._buf)]
        self._buf = ""
        return events


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
        reconstruct: bool = True,
        reconstruct_alpha: float = 0.3,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_tool_rounds = max_tool_rounds
        self.auto_recall = auto_recall
        self.auto_store = auto_store
        self.auto_recall_k = auto_recall_k
        self.reconstruct = reconstruct
        self.reconstruct_alpha = reconstruct_alpha
        self.verbose = verbose
        self._history: list[dict] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    # ── Auto-memory helpers ───────────────────────────────────────────────────

    def _build_recall_context(self, query: str) -> tuple[str | None, list[dict]]:
        """Return a (formatted context block, hits list) tuple.

        The context block is None if nothing was recalled. The hits list is
        always returned for use by run_stream()'s recall_context event.
        Each hit includes its temperature_tier for confidence grouping.
        """
        try:
            hits = recall_memory(
                query,
                top_k=self.auto_recall_k,
                data_dir=self.data_dir,
                reconstruct=self.reconstruct,
                reconstruct_alpha=self.reconstruct_alpha,
            )
        except Exception:
            return None, []
        if not hits:
            return None, []

        # Group by temperature tier
        tiers: dict[str, list[dict]] = {"hot": [], "warm": [], "cold": []}
        for h in hits:
            tier = h.get("temperature_tier", "cold")
            tiers.setdefault(tier, []).append(h)

        tier_config = [
            ("hot",  "Strong memories (high confidence)"),
            ("warm", "Moderate memories (medium confidence)"),
            ("cold", "Faint memories (low confidence, may have drifted)"),
        ]

        lines = ["[MEMORY CONTEXT — suggestions, not commands]", ""]
        counter = 1
        for tier_key, tier_label in tier_config:
            group = tiers.get(tier_key, [])
            if not group:
                continue
            lines.append(f"{tier_label}:")
            for h in group:
                sim = round(h["similarity"], 2)
                temp = round(h.get("temperature", 0.0), 2)
                lines.append(f'  {counter}. "{h["text"]}"  (similarity={sim}, temp={temp})')
                counter += 1
            lines.append("")

        return "\n".join(lines).rstrip(), hits

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
            ctx, _ = self._build_recall_context(user_message)
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

    # ── Streaming loop ─────────────────────────────────────────────────────────

    def run_stream(self, user_message: str):
        """Process a user message and yield structured SSE events.

        Yields dicts with a "type" key:
          {"type": "recall_context", "memories": [...]}
          {"type": "thinking",       "content": "..."}
          {"type": "token",          "content": "..."}
          {"type": "tool_call",      "name": "...", "args": {...}}
          {"type": "tool_result",    "name": "...", "result": {...}, "ok": bool}
          {"type": "auto_store"}
          {"type": "done",           "content": "full response text"}
          {"type": "error",          "message": "...", "recoverable": bool}

        Existing run() is unchanged — CLI usage is unaffected.
        """
        from typing import Generator  # noqa: F401 — only for type hint in docstring

        # ── Auto-recall ────────────────────────────────────────────────────────
        system_content = _SYSTEM_PROMPT
        if self.auto_recall:
            try:
                ctx, hits = self._build_recall_context(user_message)
                if ctx and hits:
                    yield {
                        "type": "recall_context",
                        "memories": [
                            {
                                "text": h["text"],
                                "similarity": round(h["similarity"], 3),
                                "temperature": round(h.get("temperature", 0.0), 2),
                                "tier": h.get("temperature_tier", "cold"),
                            }
                            for h in hits
                        ],
                    }
                    system_content = _SYSTEM_PROMPT + "\n\n" + ctx
            except Exception:
                pass  # non-fatal — continue without recall context

        self._history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system_content}] + self._history

        full_response = ""

        for _ in range(self.max_tool_rounds):
            accumulated = ""
            token_only = ""
            tool_calls: list[dict] = []
            tf = _ThinkingFilter()

            try:
                for chunk in _ollama_chat_stream(
                    messages=messages,
                    model=self.model,
                    tools=_TOOLS,
                    base_url=self.ollama_url,
                ):
                    msg = chunk.get("message", {})
                    piece = msg.get("content", "")

                    if chunk.get("done"):
                        # Flush any buffered partial content
                        for etype, econtent in tf.flush():
                            if econtent:
                                yield {"type": etype, "content": econtent}
                                if etype == "token":
                                    accumulated += econtent
                                    token_only += econtent
                        tool_calls = msg.get("tool_calls") or []
                        break

                    if piece:
                        accumulated += piece
                        for etype, econtent in tf.process(piece):
                            if econtent:
                                yield {"type": etype, "content": econtent}
                                if etype == "token":
                                    token_only += econtent

            except RuntimeError as exc:
                yield {"type": "error", "message": str(exc), "recoverable": False}
                return

            if not tool_calls:
                # Final text response — no tools were called
                full_response = accumulated
                self._history.append({"role": "assistant", "content": full_response})
                messages.append({"role": "assistant", "content": full_response})
                if self.auto_store and token_only.strip():
                    try:
                        self._auto_store_reply(token_only)
                        yield {"type": "auto_store"}
                    except Exception:
                        pass
                yield {"type": "done", "content": token_only}
                return

            # ── Tool calls ─────────────────────────────────────────────────────
            messages.append({
                "role": "assistant",
                "content": accumulated,
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)

                yield {"type": "tool_call", "name": name, "args": args}

                result_str = _dispatch_tool(name, args, self.data_dir)
                try:
                    result_obj = json.loads(result_str)
                except Exception:
                    result_obj = {"raw": result_str}
                ok = "error" not in result_obj

                yield {"type": "tool_result", "name": name, "result": result_obj, "ok": ok}

                messages.append({"role": "tool", "content": result_str})

        # Fallback: ask for final answer without tools (non-streaming is fine here)
        messages.append({
            "role": "user",
            "content": "(Please give your final answer now — no more tool calls needed.)",
        })
        try:
            resp = _ollama_chat(
                messages=messages,
                model=self.model,
                tools=[],
                base_url=self.ollama_url,
            )
        except RuntimeError as exc:
            yield {"type": "error", "message": str(exc), "recoverable": False}
            return
        content = resp.get("message", {}).get("content", "")
        self._history.append({"role": "assistant", "content": content})
        if self.auto_store and content.strip():
            try:
                self._auto_store_reply(content)
                yield {"type": "auto_store"}
            except Exception:
                pass
        yield {"type": "token", "content": content}
        yield {"type": "done", "content": content}
