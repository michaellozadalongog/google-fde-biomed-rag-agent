"""Embedders. Default: offline TF-IDF -> LSA (dense, L2-normalized).
Optional: sentence-transformers if installed (BIOMED_RAG_EMBEDDER=st)."""
from __future__ import annotations
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class LSAEmbedder:
    name = "tfidf-lsa"
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    def fit(self, texts):
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self
    def encode(self, texts) -> np.ndarray:
        Z = self.svd.transform(self.vec.transform(texts)).astype("float32")
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    def sparse(self, texts):
        return self.vec.transform(texts)

class STEmbedder:  # pragma: no cover - optional
    name = "sentence-transformers"
    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    def fit(self, texts):
        self.vec.fit(texts); return self
    def encode(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True), dtype="float32")
    def sparse(self, texts):
        return self.vec.transform(texts)

def make_embedder():
    return STEmbedder() if os.getenv("BIOMED_RAG_EMBEDDER") == "st" else LSAEmbedder()
