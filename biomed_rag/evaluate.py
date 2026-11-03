"""Offline evaluation: retrieval (Recall@k, MRR, nDCG@k) per mode, answer fact-hit rate,
groundedness, critic retry rate, and latency percentiles. Writes results/ + figs/."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from .pipeline import build, ROOT

K = 4

def rel_hits(results, rel):
    rs = {tuple(r) for r in rel}
    return [1 if (c.doc, c.section) in rs else 0 for c, _ in results]

def retrieval_metrics(store, items, mode):
    rec, mrr, ndcg, lat = [], [], [], []
    for it in items:
        t0 = time.perf_counter(); res = store.search(it["q"], K, mode); lat.append((time.perf_counter() - t0) * 1000)
        h = rel_hits(res, it["rel"]); n_rel = len(it["rel"])
        rec.append(min(1.0, sum(h) / n_rel))
        mrr.append(next((1 / (i + 1) for i, x in enumerate(h) if x), 0.0))
        dcg = sum(x / math.log2(i + 2) for i, x in enumerate(h))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, K)))
        ndcg.append(dcg / idcg)
    return {"recall@4": float(np.mean(rec)), "mrr": float(np.mean(mrr)), "ndcg@4": float(np.mean(ndcg)),
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95))}

def run(write: bool = True, verbose: bool = True):
    orch, store = build()
    items = json.loads((ROOT / "data" / "eval_set.json").read_text())
    retr = {m: retrieval_metrics(store, items, m) for m in ("lexical", "dense", "hybrid")}
    rows, lat = [], []
    for it in items:
        s, t = orch.answer(it["q"])
        lat.append(t.total_ms())
        hit = all(f.lower() in s.answer.lower() for f in it["facts"])
        rows.append({"q": it["q"], "fact_hit": hit, "steps": s.steps, **s.critique, "latency_ms": t.total_ms()})
    e2e = {"fact_hit_rate": float(np.mean([r["fact_hit"] for r in rows])),
           "groundedness": float(np.mean([r["groundedness"] for r in rows])),
           "critic_retry_rate": float(np.mean([r["steps"] > 1 for r in rows])),
           "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
           "n": len(rows), "embedder": store.emb.name, "chunks": len(store.chunks)}
    out = {"retrieval": retr, "end_to_end": e2e, "per_question": rows}
    if verbose:
        print("\n== Retrieval (k=4) ==")
        for m, v in retr.items():
            print(f"{m:8s} recall@4={v['recall@4']:.3f} mrr={v['mrr']:.3f} ndcg@4={v['ndcg@4']:.3f} p95={v['p95_ms']:.2f}ms")
        print("\n== End-to-end agent ==")
        for k, v in e2e.items():
            print(f"{k:18s} {v:.3f}" if isinstance(v, float) else f"{k:18s} {v}")
    if write:
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "eval_results.json").write_text(json.dumps(out, indent=2))
        plot(out, lat)
    return out

def plot(out, lat):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figs = ROOT / "figs"; figs.mkdir(exist_ok=True)
    modes = list(out["retrieval"]); mets = ["recall@4", "mrr", "ndcg@4"]
    x = np.arange(len(mets)); w = 0.25
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, m in enumerate(modes):
        ax.bar(x + (i - 1) * w, [out["retrieval"][m][k] for k in mets], w, label=m)
    ax.set_xticks(x, mets); ax.set_ylim(0, 1.05); ax.set_title("Retrieval quality by mode (n=20, k=4)"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "retrieval_by_mode.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lat, bins=10); ax.set_xlabel("end-to-end agent latency (ms)"); ax.set_ylabel("queries")
    ax.set_title(f"Agent latency p50={np.percentile(lat,50):.1f} ms, p95={np.percentile(lat,95):.1f} ms")
    fig.tight_layout(); fig.savefig(figs / "agent_latency.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    run()
