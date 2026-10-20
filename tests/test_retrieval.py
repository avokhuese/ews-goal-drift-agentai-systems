from src.retrieval.corpus import load_corpus
from src.retrieval.retriever import Retriever


def test_retrieval_topical():
    docs = load_corpus("data/corpus.jsonl")
    r = Retriever(docs)
    hits = r.search("thermal expansion ocean", top_k=3)
    assert any("thermal" in d["text"].lower() for d in hits)