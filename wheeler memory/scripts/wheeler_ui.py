"""Wheeler Memory Web UI — single-command local server."""

import json
import sys
import threading
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
import numpy as np

PORT = 7437
UI_FILE = Path(__file__).parent.parent / "ui" / "dashboard.html"

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
