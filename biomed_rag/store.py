"""FAISS inner-product index + sparse lexical scores fused with Reciprocal Rank Fusion."""
from __future__ import annotations
import faiss, numpy as np
from .ingest import Chunk

class HybridStore:
    def __init__(self, embedder, chunks: list[Chunk]):
        self.emb, self.chunks = embedder, chunks
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        X = embedder.encode(texts)
        self.index = faiss.IndexFlatIP(X.shape[1]); self.index.add(X)
        self.S = embedder.sparse(texts)

    def dense(self, q: str, k: int):
        D, I = self.index.search(self.emb.encode([q]), k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

    def lexical(self, q: str, k: int):
        s = (self.S @ self.emb.sparse([q]).T).toarray().ravel()
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]

    def search(self, q: str, k: int = 5, mode: str = "hybrid", rrf_k: int = 60):
        if mode == "dense":
            return [(self.chunks[i], s) for i, s in self.dense(q, k)]
        if mode == "lexical":
            return [(self.chunks[i], s) for i, s in self.lexical(q, k)]
        fused: dict[int, float] = {}
        for ranked in (self.dense(q, 20), self.lexical(q, 20)):
            for r, (i, _) in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self.chunks[i], s) for i, s in top]
