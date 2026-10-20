"""BM25 retrieval over the structured corpus."""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, docs: List[Dict[str, Any]]):
        if not docs:
            raise ValueError(
                "Retriever received an empty corpus. "
                "Check data/corpus.jsonl — it may be empty or missing."
            )
        self.docs = docs
        self.tokenized = [d["text"].lower().split() for d in docs]
        # Filter out zero-length tokenized docs (would break avgdl)
        self.tokenized = [toks if toks else ["<empty>"] for toks in self.tokenized]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(query.lower().split())
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.docs[i] for i in idx]