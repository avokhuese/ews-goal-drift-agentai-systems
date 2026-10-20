"""
Turn observables → scalar time series. These are the signals on which EWS are computed.
"""

import numpy as np
from collections import Counter
from typing import Dict
from src.signals.embeddings import cosine
from src.agent.loop import Episode


def goal_drift_signal(ep: Episode) -> np.ndarray:
    """1 - cosine(anchor_goal, current_understanding). Rises when the goal has drifted."""
    return np.array([
        1.0 - cosine(ep.goal_anchor_emb, t.understanding_emb) for t in ep.turns
    ])


def query_novelty_signal(ep: Episode) -> np.ndarray:
    """1 - max cosine similarity to any previous query."""
    sims = []
    queries = [t.query_emb for t in ep.turns]
    for i, q in enumerate(queries):
        if i == 0:
            sims.append(0.0)
            continue
        prev = queries[:i]
        sims.append(1.0 - max(cosine(q, p) for p in prev))
    return np.array(sims)


def evidence_entropy_signal(ep: Episode) -> np.ndarray:
    """Shannon entropy of the topic distribution of retrieved evidence per turn."""
    out = []
    for t in ep.turns:
        if not t.evidence_topics:
            out.append(0.0)
            continue
        counts = Counter(t.evidence_topics)
        p = np.array(list(counts.values()), dtype=float)
        p /= p.sum()
        out.append(float(-np.sum(p * np.log(p + 1e-12))))
    return np.array(out)


def query_step_distance_signal(ep: Episode) -> np.ndarray:
    """Embedding distance between consecutive queries."""
    dists = [0.0]
    for i in range(1, len(ep.turns)):
        dists.append(1.0 - cosine(ep.turns[i].query_emb, ep.turns[i-1].query_emb))
    return np.array(dists)


def retrieval_offtopic_signal(ep: Episode, off_topic_label: str = "economics") -> np.ndarray:
    """Fraction of retrieved docs labeled as off-topic (a proxy for silent drift)."""
    return np.array([
        sum(1 for tp in t.evidence_topics if tp == off_topic_label) / max(1, len(t.evidence_topics))
        for t in ep.turns
    ])


def extract_all_signals(ep: Episode) -> Dict[str, np.ndarray]:
    return {
        "goal_drift": goal_drift_signal(ep),
        "query_novelty": query_novelty_signal(ep),
        "evidence_entropy": evidence_entropy_signal(ep),
        "query_step_distance": query_step_distance_signal(ep),
        "retrieval_offtopic": retrieval_offtopic_signal(ep),
    }