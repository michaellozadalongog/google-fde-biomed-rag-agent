"""Minimal JSON HTTP API (stdlib). Same contract a Cloud Run service would expose.
POST /query {"question": "..."} -> {answer, citations, critique, latency_ms, trace_id}"""
from __future__ import annotations
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
from .pipeline import build

ORCH, _ = build()

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, {"status": "ok"}) if self.path == "/healthz" else self._send(404, {})
    def do_POST(self):
        if self.path != "/query": return self._send(404, {})
        q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("question", "")
        if not q: return self._send(400, {"error": "question required"})
        s, t = ORCH.answer(q)
        self._send(200, {"answer": s.answer, "citations": s.citations, "critique": s.critique,
                         "steps": s.steps, "latency_ms": t.total_ms(), "trace_id": t.trace_id})

def main():
    port = int(os.getenv("PORT", "8080"))  # Cloud Run convention
    print(f"serving on :{port}"); HTTPServer(("0.0.0.0", port), H).serve_forever()

if __name__ == "__main__":
    main()
