"""
Embedding backend. Prefers sentence-transformers; falls back to TF-IDF so the
prototype runs without heavy deps.
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


class Embedder:
    def __init__(self, backend: str = None):
        self.backend = backend or os.getenv("EMBED_BACKEND", "tfidf")
        self._st_model = None
        self._tfidf = None
        self._fitted = False

    def fit(self, corpus: List[str]):
        if self.backend == "tfidf":
            self._tfidf = TfidfVectorizer().fit(corpus)
            self._fitted = True
        elif self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._fitted = True
        return self

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "tfidf":
            assert self._fitted, "Call fit() first for TF-IDF backend"
            return self._tfidf.transform(texts).toarray()
        elif self.backend == "sentence-transformers":
            return self._st_model.encode(texts, normalize_embeddings=True)
        raise ValueError(self.backend)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))