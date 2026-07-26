#!/usr/bin/env python3
"""Development server with caching disabled.

`python3 -m http.server` sends Last-Modified but no Cache-Control, so the
browser caches ES modules heuristically and you can spend a while debugging a
stale file. This forces no-store on everything.

Serves `web/`, exactly what Netlify publishes.

Usage: python3 serve.py [port]
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        return


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    root = Path(__file__).resolve().parent / "web"
    handler = partial(NoCacheHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"http://localhost:{port}  (no cache)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped")


if __name__ == "__main__":
    main()
