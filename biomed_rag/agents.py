"""Hierarchical multi-agent orchestration with a ReAct-style loop.

Orchestrator (supervisor) -> Planner -> {StructuredTool, Retriever} -> Synthesizer -> Critic
The Critic can send the loop back to the Retriever with a reformulated query (Thought/Action/Observation).
Synthesizer is extractive by default (offline, hallucination-free by construction); swap in an LLM
(e.g. Gemini on Vertex AI) via the `LLM` protocol without changing the rest of the graph.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from .ingest import Chunk
from .store import HybridStore
from .telemetry import Trace

STOP = set("the a an of for to is are what which how and or in on at be should does do with by from when than per".split())
EXPANSIONS = {  # domain query expansion used by the Retriever on Critic retry
    "resolution": "axial lateral filament spacing", "cleaning": "reprocessing disinfection HLD",
    "disinfect": "high-level disinfection HLD Spaulding", "bubbles": "degas vacuum artifacts",
    "safety": "mechanical index thermal index ALARA", "contrast": "CNR gCNR ROI",
    "penetration": "depth of penetration DOP attenuation", "fail": "fails removed service",
}

def terms(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9.]+", s.lower()) if w not in STOP and len(w) > 1]

class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...

@dataclass
class State:
    question: str
    query: str = ""
    contexts: list = field(default_factory=list)
    structured: list = field(default_factory=list)
    answer: str = ""
    citations: list = field(default_factory=list)
    critique: dict = field(default_factory=dict)
    steps: int = 0

class Planner:
    PROBE = re.compile(r"\bP-\d{3}\b", re.I)
    def plan(self, s: State, t: Trace) -> list[str]:
        acts = []
        if self.PROBE.search(s.question) or re.search(r"\bprobes?\b.*\b(fail|status|review|pass)", s.question, re.I):
            acts.append("structured_lookup")
        acts.append("retrieve")
        t.event("planner", "Thought: choose tools", actions=acts)
        return acts

class StructuredTool:
    def __init__(self, rows): self.rows = rows
    def run(self, s: State, t: Trace):
        ids = {m.upper() for m in Planner.PROBE.findall(s.question)}
        st = re.search(r"\b(fail|review|pass)", s.question, re.I)
        hits = [r for r in self.rows if r["probe_id"] in ids or (st and not ids and r["qa_status"].startswith(st.group(1).lower()))]
        s.structured = hits
        t.event("structured_tool", "Observation", rows=len(hits))

class Retriever:
    def __init__(self, store: HybridStore, k: int = 4): self.store, self.k = store, k
    def run(self, s: State, t: Trace, reformulate: bool = False):
        q = s.question
        if reformulate:
            q += " " + " ".join(v for k, v in EXPANSIONS.items() if k in q.lower())
        s.query = q
        s.contexts = self.store.search(q, self.k)
        t.event("retriever", "Action: search", query=q, top=[c.chunk_id for c, _ in s.contexts])

class Synthesizer:
    def __init__(self, store: HybridStore, llm: LLM | None = None, n_sent: int = 3):
        self.store, self.llm, self.n = store, llm, n_sent
    def run(self, s: State, t: Trace):
        parts, cites = [], []
        for r in s.structured:
            note = ""
            if r.get("dop_cm") and float(r["baseline_dop_cm"]) - float(r["dop_cm"]) > 1.0:
                note = " DOP dropped >1 cm from baseline, which triggers service review per QA protocol."
            parts.append(f"{r['probe_id']} ({r['model_type']}, {r['center_freq_mhz']} MHz): QA status '{r['qa_status']}' on {r['last_qa_date']}; "
                         f"dead zone {r['dead_zone_mm']} mm, axial res {r['axial_res_mm']} mm, DOP {r['dop_cm']} cm (baseline {r['baseline_dop_cm']} cm).{note}")
            cites.append(f"probes#{r['probe_id']}")
        cand = []
        for c, _ in s.contexts:
            if c.doc == "probes": continue
            body = c.text.split(": ", 1)[-1]
            for sent in re.split(r"(?<=[.!?])\s+", body):
                if len(sent.split()) >= 5: cand.append((sent, c))
        if cand:
            qv = self.store.emb.sparse([s.query or s.question])
            M = self.store.emb.sparse([x for x, _ in cand])
            sc = (M @ qv.T).toarray().ravel()
            for i in np.argsort(-sc)[: self.n]:
                if sc[i] > 0:
                    parts.append(cand[i][0]); cites.append(cand[i][1].chunk_id)
        if self.llm and parts:
            s.answer = self.llm.generate(f"Answer using only:\n{parts}\nQ: {s.question}")
        else:
            s.answer = " ".join(parts) if parts else "Insufficient evidence in the indexed corpus."
        s.citations = list(dict.fromkeys(cites))
        t.event("synthesizer", "Draft answer", n_cites=len(s.citations))

class Critic:
    def __init__(self, min_coverage: float = 0.5): self.min_cov = min_coverage
    def run(self, s: State, t: Trace) -> bool:
        q, a = set(terms(s.question)), set(terms(s.answer))
        coverage = len(q & a) / max(1, len(q))
        ctx = " ".join(c.text for c, _ in s.contexts) + " " + " ".join(map(str, s.structured))
        sents = [x for x in re.split(r"(?<=[.!?])\s+", s.answer) if x]
        grounded = np.mean([len(set(terms(x)) & set(terms(ctx))) / max(1, len(set(terms(x)))) for x in sents]) if sents else 0.0
        ok = coverage >= self.min_cov and grounded >= 0.8
        s.critique = {"coverage": round(coverage, 3), "groundedness": round(float(grounded), 3), "accept": bool(ok)}
        t.event("critic", "Observation: critique", **s.critique)
        return ok

class Orchestrator:
    def __init__(self, store: HybridStore, rows, llm: LLM | None = None, max_steps: int = 2):
        self.planner, self.tool = Planner(), StructuredTool(rows)
        self.retriever, self.synth = Retriever(store), Synthesizer(store, llm)
        self.critic, self.max_steps = Critic(), max_steps

    def answer(self, question: str) -> tuple[State, Trace]:
        s, t = State(question), Trace()
        with t.span("agent.planner"):
            acts = self.planner.plan(s, t)
        if "structured_lookup" in acts:
            with t.span("agent.structured_tool"):
                self.tool.run(s, t)
        for step in range(self.max_steps):
            s.steps = step + 1
            with t.span("agent.retriever", step=step):
                self.retriever.run(s, t, reformulate=step > 0)
            with t.span("agent.synthesizer", step=step):
                self.synth.run(s, t)
            with t.span("agent.critic", step=step):
                if self.critic.run(s, t):
                    break
        return s, t
