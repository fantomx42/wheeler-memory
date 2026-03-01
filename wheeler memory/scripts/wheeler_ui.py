"""Wheeler Memory Web UI — single-command local server.

Serves two interfaces:
  /         — memory dashboard (store, recall, inspect)
  /chat     — LLM chat interface with streaming, memory tools, and web search
"""

import json
import sys
import threading
import time
import uuid
import urllib.request
import urllib.error
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from wheeler_memory.rotation import store_with_rotation_retry
from wheeler_memory.storage import recall_memory, list_memories, DEFAULT_DATA_DIR
from wheeler_memory.eviction import forget_by_text
from wheeler_memory.consolidation import sleep_consolidate
from wheeler_memory.chunking import select_chunk
from wheeler_memory.hashing import text_to_hex
from wheeler_memory.agent import WheelerAgent
import numpy as np

PORT = 7437
UI_FILE = Path(__file__).parent.parent / "ui" / "dashboard.html"
CHAT_FILE = Path(__file__).parent.parent / "ui" / "chat.html"

# ── Chat session management ────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}  # session_id -> {"agent": WheelerAgent, "last_active": float, "lock": Lock}
_sessions_lock = threading.Lock()
_SESSION_TTL = 3600  # 1 hour

DEFAULT_MODEL = "qwen3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    """Return (session_id, session_dict), creating a new session if needed."""
    with _sessions_lock:
        if session_id and session_id in _sessions:
            sess = _sessions[session_id]
            sess["last_active"] = time.time()
            return session_id, sess
        sid = session_id or str(uuid.uuid4())
        agent = WheelerAgent(
            model=DEFAULT_MODEL,
            ollama_url=DEFAULT_OLLAMA_URL,
            auto_recall=True,
            auto_store=True,
        )
        sess = {
            "agent": agent,
            "last_active": time.time(),
            "lock": threading.Lock(),
        }
        _sessions[sid] = sess
        return sid, sess


def _reap_stale_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        stale = [k for k, v in _sessions.items() if now - v["last_active"] > _SESSION_TTL]
        for k in stale:
            del _sessions[k]

SEED_MEMORIES = [
    "cellular automata evolve through local neighbor rules toward stable attractors",
    "the Wheeler Memory system uses Pearson correlation for attractor recall",
    "fix the python debug error in the async network stack",
    "GPU kernel ping-pong buffers enable parallel CA evolution without race conditions",
    "check GPU temperatures before long training runs to avoid thermal throttling",
    "meaning is what survives symbolic pressure — stable attractors encode durable concepts",
    "sleep consolidation prunes redundant frames from cold memory bricks",
    "von Neumann neighborhood: up, down, left, right — no diagonal neighbors",
]


def _seed_if_empty(data_dir):
    if list_memories(data_dir):
        return
    print("Seeding example memories...")
    for text in SEED_MEMORIES:
        store_with_rotation_retry(text, data_dir=data_dir)
    print(f"Seeded {len(SEED_MEMORIES)} example memories.")


class WheelerHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress per-request logging

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"error": message}, status)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            content = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/chat":
            if not CHAT_FILE.exists():
                self.send_error_json("chat.html not found", 404)
                return
            content = CHAT_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/health":
            try:
                req = urllib.request.Request(
                    f"{DEFAULT_OLLAMA_URL}/api/tags",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                models = [m.get("name", "") for m in data.get("models", [])]
                self.send_json({"ollama": True, "models": models, "default_model": DEFAULT_MODEL})
            except Exception:
                self.send_json({"ollama": False, "models": [], "default_model": DEFAULT_MODEL})
            return

        if path == "/api/memories":
            try:
                memories = list_memories(DEFAULT_DATA_DIR)
                result = [
                    {
                        "key": m["hex_key"],
                        "chunk": m["chunk"],
                        "temperature": m["temperature"],
                        "tier": m["temperature_tier"],
                        "text": m["text"],
                        "state": m["state"],
                        "ticks": m["convergence_ticks"],
                        "timestamp": m["timestamp"],
                    }
                    for m in memories
                ]
                self.send_json(result)
            except Exception as e:
                self.send_error_json(str(e), 500)
            return

        if path.startswith("/api/attractor/"):
            key = path[len("/api/attractor/"):]
            params = parse_qs(parsed.query)
            chunk = params.get("chunk", ["general"])[0]
            npy_path = DEFAULT_DATA_DIR / "chunks" / chunk / "attractors" / f"{key}.npy"
            if npy_path.exists():
                try:
                    data = np.load(npy_path).flatten().tolist()
                    self.send_json(data)
                except Exception as e:
                    self.send_error_json(str(e), 500)
            else:
                self.send_error_json("not found", 404)
            return

        self.send_error_json("not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self.read_body()
        except Exception:
            self.send_error_json("invalid JSON")
            return

        if path == "/api/store":
            text = body.get("text", "").strip()
            if not text:
                self.send_error_json("text required")
                return
            try:
                r = store_with_rotation_retry(text, data_dir=DEFAULT_DATA_DIR)
                chunk = select_chunk(text)
                key = text_to_hex(text)
                self.send_json({
                    "stored": r["state"] == "CONVERGED",
                    "key": key,
                    "state": r["state"],
                    "ticks": r["convergence_ticks"],
                    "chunk": chunk,
                })
            except Exception as e:
                self.send_error_json(str(e), 500)
            return

        if path == "/api/recall":
            text = body.get("text", "").strip()
            if not text:
                self.send_error_json("text required")
                return
            top_k = int(body.get("top_k", 10))
            try:
                results = recall_memory(text, top_k=top_k, data_dir=DEFAULT_DATA_DIR)
                self.send_json([
                    {
                        "key": m["hex_key"],
                        "chunk": m["chunk"],
                        "temperature": m["temperature"],
                        "tier": m["temperature_tier"],
                        "text": m["text"],
                        "state": m["state"],
                        "ticks": m["convergence_ticks"],
                        "similarity": m["similarity"],
                        "timestamp": m["timestamp"],
                    }
                    for m in results
                ])
            except Exception as e:
                self.send_error_json(str(e), 500)
            return

        if path == "/api/forget":
            text = body.get("text", "").strip()
            if not text:
                self.send_error_json("text required")
                return
            try:
                forgotten = forget_by_text(text, DEFAULT_DATA_DIR)
                self.send_json({"forgotten": bool(forgotten)})
            except Exception as e:
                self.send_error_json(str(e), 500)
            return

        if path == "/api/sleep":
            try:
                result = sleep_consolidate(DEFAULT_DATA_DIR)
                self.send_json({
                    "consolidated": len(result.memories_consolidated),
                    "skipped": len(result.memories_skipped),
                })
            except Exception as e:
                self.send_error_json(str(e), 500)
            return

        if path == "/api/chat":
            message = body.get("message", "").strip()
            if not message:
                self.send_error_json("message required")
                return

            session_id = body.get("session_id") or None
            _reap_stale_sessions()
            sid, sess = _get_or_create_session(session_id)

            # SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def send_event(event_type: str, data: dict) -> bool:
                """Write one SSE frame. Returns False if client disconnected."""
                try:
                    frame = (
                        f"event: {event_type}\n"
                        f"data: {json.dumps(data)}\n\n"
                    ).encode()
                    self.wfile.write(frame)
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return False

            # First event: confirm session ID to client
            if not send_event("session", {"session_id": sid}):
                return

            agent: WheelerAgent = sess["agent"]
            with sess["lock"]:
                try:
                    for event in agent.run_stream(message):
                        if not send_event(event["type"], event):
                            break  # client disconnected
                except Exception as exc:
                    send_event("error", {"message": str(exc), "recoverable": False})
            return

        self.send_error_json("not found", 404)


def main():
    if not UI_FILE.exists():
        print(f"ERROR: ui/dashboard.html not found at {UI_FILE}", file=sys.stderr)
        sys.exit(1)

    _seed_if_empty(DEFAULT_DATA_DIR)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), WheelerHandler)
    except OSError:
        print(f"ERROR: port {PORT} already in use.", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{PORT}"
    print(f"Wheeler Memory  →  {url}\nCtrl+C to stop.")
    threading.Thread(
        target=lambda: (__import__("time").sleep(0.5), webbrowser.open(url)),
        daemon=True,
    ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
