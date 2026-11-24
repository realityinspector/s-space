#!/usr/bin/env python3
"""Local server for the S-space decision explorer.

Serves the static page and persists answers to answers.json / answers.md
next to this file. Stdlib only.

    python3 explorer/server.py [port]
"""
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANS_JSON = ROOT / "answers.json"
ANS_MD = ROOT / "answers.md"


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send(self, code: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/answers":
            body = ANS_JSON.read_text(encoding="utf-8") if ANS_JSON.exists() else "{}"
            self._send(200, body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/answers":
            self._send(404, '{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > 2_000_000:
            self._send(413, '{"error":"too large"}')
            return
        try:
            obj = json.loads(self.rfile.read(length))
            assert isinstance(obj, dict)
        except Exception:
            self._send(400, '{"error":"bad json"}')
            return
        markdown = obj.pop("markdown", None)
        atomic_write(ANS_JSON, json.dumps(obj, indent=2, ensure_ascii=False))
        if isinstance(markdown, str):
            atomic_write(ANS_MD, markdown)
        self._send(200, '{"ok":true}')

    def log_message(self, format, *args):  # noqa: A002 (matches base signature)
        if args and "/api/" in str(args[0]):
            return
        super().log_message(format, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8742
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"S-space explorer → http://127.0.0.1:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
