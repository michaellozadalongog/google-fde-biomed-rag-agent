"""Lightweight tracing: span timings + structured events (maps to Cloud Trace/Logging)."""
from __future__ import annotations
import json, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attrs):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000, 3), **attrs})

    def event(self, agent: str, msg: str, **data):
        self.events.append({"agent": agent, "msg": msg, **data})

    def total_ms(self) -> float:
        return round(sum(s["ms"] for s in self.spans if s["name"].startswith("agent.")), 3)

    def to_json(self) -> str:
        return json.dumps({"trace_id": self.trace_id, "spans": self.spans, "events": self.events}, indent=2)
