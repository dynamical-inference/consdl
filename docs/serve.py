#!/usr/bin/env python3
"""Local preview server (stdlib only).

    python3 serve.py            # http://localhost:8000
    python3 serve.py 8123       # custom port
    python3 serve.py --public   # listen on all interfaces → http://<hostname>:8000 from other machines

Caching is disabled, so a browser refresh always shows the latest files.
On a remote machine, forward the port first:  ssh -L 8000:localhost:8000 <host>
"""
import functools
import http.server
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    # Never serve dotfiles or local source material.
    BLOCKED_SUFFIXES = (".pdf", ".tex", "consdl-project-content.md")

    def send_head(self):
        parts = self.path.split("?", 1)[0].split("#", 1)[0].strip("/").split("/")
        if any(p.startswith(".") for p in parts) or parts[-1].endswith(self.BLOCKED_SUFFIXES):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Keep the console quiet apart from errors and the live-reload polling noise.
        if args and str(args[1])[0] in "45":
            super().log_message(fmt, *args)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    public = "--public" in sys.argv
    port = int(args[0]) if args else 8000
    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("0.0.0.0" if public else "127.0.0.1", port), handler) as httpd:
        host = socket.getfqdn()
        if public:
            print(f"Serving {ROOT} at http://{host}:{port}")
            print("Anyone who can reach this host on that port can see the page.")
        else:
            print(f"Serving {ROOT} at http://localhost:{port}")
            print(f"Remote machine? On your laptop run:  ssh -L {port}:localhost:{port} {host}")
            print("Or run with --public to get a URL reachable from other machines.")
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
