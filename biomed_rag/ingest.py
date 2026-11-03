"""Load unstructured markdown + structured CSV; section-aware chunking with overlap."""
from __future__ import annotations
import csv, re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n(?=\d+\.)", text) if s.strip()]

def chunk_markdown(path: Path, max_words: int = 80, overlap_sents: int = 1) -> list[Chunk]:
    raw = path.read_text()
    title = raw.splitlines()[0].lstrip("# ").strip()
    out: list[Chunk] = []
    for sec in re.split(r"\n(?=## )", raw):
        lines = sec.strip().splitlines()
        if not lines or lines[0].startswith("# "):
            # title block: keep any body under the H1
            body = "\n".join(l for l in lines[1:] if not l.startswith("## "))
            name = "Overview"
        else:
            name, body = lines[0].lstrip("# ").strip(), "\n".join(lines[1:])
        sents, buf = _sentences(body), []
        def flush():
            if buf:
                out.append(Chunk(f"{path.stem}#{len(out)}", path.stem, name, f"{title} — {name}: " + " ".join(buf)))
        for s in sents:
            if buf and len(" ".join(buf + [s]).split()) > max_words:
                flush(); buf[:] = buf[-overlap_sents:] if overlap_sents else []
            buf.append(s)
        flush()
    return out

def load_structured(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))

def structured_to_chunks(rows: list[dict], name: str) -> list[Chunk]:
    """Also index each row as text so hybrid questions can hit it via vector search."""
    return [Chunk(f"{name}#{r['probe_id']}", name, r["probe_id"],
                  "Probe record: " + "; ".join(f"{k}={v}" for k, v in r.items())) for r in rows]

def load_corpus(data_dir: Path):
    chunks = []
    for p in sorted((data_dir / "docs").glob("*.md")):
        chunks += chunk_markdown(p)
    rows = load_structured(data_dir / "structured" / "probes.csv")
    chunks += structured_to_chunks(rows, "probes")
    return chunks, rows
