from __future__ import annotations
from pathlib import Path
from .agents import Orchestrator
from .embed import make_embedder
from .ingest import load_corpus
from .store import HybridStore

ROOT = Path(__file__).resolve().parent.parent

def build(data_dir: Path = ROOT / "data"):
    chunks, rows = load_corpus(data_dir)
    store = HybridStore(make_embedder(), chunks)
    return Orchestrator(store, rows), store
