"""
LLM backend abstraction.

- "mock": corpus-aware, deterministic. Derives its vocabulary from the loaded
  corpus so changing the corpus changes the queries. Simulates a goal-driven
  agent whose latent "drift pressure" ramps up over turns.
- "openai": real LLM via OpenAI Chat Completions.
- "ollama": local LLM.

The mock backend is what makes this prototype reproducible offline; when you
plug in a real LLM, the agent loop and signal extraction are unchanged.
"""

import os
import json
import math
import random
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class LLMResponse:
    query: str
    understanding: str


class BaseLLM:
    def generate(self, original_goal, understanding, evidence, turn, max_turns=60):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock LLM: corpus-aware, drift-pressured
# ---------------------------------------------------------------------------
class MockLLM(BaseLLM):
    """
    Corpus-aware mock agent.

    Vocabulary is extracted from the corpus. The "primary" topic is inferred
    from the original goal (via keyword overlap); everything else is treated
    as an off-topic attractor. As `drift_pressure` ramps up over turns, more
    off-topic terms leak into queries and the understanding drifts.

    Because the vocabulary comes from the corpus, changing the JSONL changes
    the queries on the next run.
    """

    STOPWORDS = {
        "the","a","an","of","in","on","to","and","or","is","are","was","were",
        "be","been","has","have","had","it","its","this","that","these","those",
        "for","by","with","as","at","from","which","who","what","how","when",
        "where","why","can","could","should","would","may","might","will","do",
        "does","did","not","no","than","then","also","such","into","over",
        "under","about","between","after","before","during","through","very",
        "more","most","some","any","all","each","other","only","own","same",
        "so","too","but","if","because","while","using","used","one","two",
    }

    def __init__(self, drift_pressure: float = 0.9, seed: int = 42,
                 corpus: Optional[List[Any]] = None, goal: str = ""):
        self.drift_pressure = drift_pressure
        self.rng = random.Random(seed)

        if corpus:
            self.topic_keywords, self.primary_topic, self.off_topics = \
                self._build_vocab(corpus, goal)
        else:
            # Fallback vocab so the mock still runs without a corpus
            self.topic_keywords = {
                "generic": ["evidence", "data", "analysis", "signal", "trend"]
            }
            self.primary_topic = "generic"
            self.off_topics = ["generic"]

        self.corpus_fingerprint = self._fingerprint(corpus or [])

    # ---------------------------------------------------------- vocab
    @staticmethod
    def _fingerprint(corpus: list) -> str:
        if not corpus:
            return "empty"
        h = hashlib.md5(json.dumps(corpus, sort_keys=True).encode()).hexdigest()
        return h[:8]

    @classmethod
    def _extract_keywords(cls, text: str, top_n: int = 12) -> List[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower())
        words = [w for w in words if w not in cls.STOPWORDS]
        return [w for w, _ in Counter(words).most_common(top_n)]

    def _build_vocab(self, corpus: list, goal: str):
        topic_keywords: dict[str, list[str]] = {}
        for doc in corpus:
            topic = doc.get("topic", "unknown")
            kws = self._extract_keywords(doc.get("text", ""), top_n=10)
            topic_keywords.setdefault(topic, []).extend(kws)

        # Dedupe, preserve order
        for t in topic_keywords:
            topic_keywords[t] = list(dict.fromkeys(topic_keywords[t]))

        # Pick the primary topic = the one whose keywords best match the goal
        goal_kws = set(self._extract_keywords(goal, top_n=15))
        if goal_kws and topic_keywords:
            scores = {
                t: len(goal_kws.intersection(kws))
                for t, kws in topic_keywords.items()
            }
            best = max(scores, key=scores.get)
            if scores[best] == 0:
                # No overlap — fall back to the topic with the most docs
                best = Counter(d.get("topic", "unknown") for d in corpus).most_common(1)[0][0]
            primary_topic = best
        else:
            primary_topic = next(iter(topic_keywords), "generic")

        off_topics = [t for t in topic_keywords if t != primary_topic]
        return topic_keywords, primary_topic, off_topics

    # ---------------------------------------------------------- dynamics
    def _mu(self, turn: int, max_turns: int) -> float:
        """Effective bifurcation parameter: negative (stable) → positive (drifted)."""
        return -1.0 + 2.0 * (turn / max_turns) * self.drift_pressure

    # ---------------------------------------------------------- generate
    def generate(self, original_goal, understanding, evidence, turn, max_turns=60):
        mu = self._mu(turn, max_turns)
        p_drift = 1.0 / (1.0 + math.exp(-4.0 * mu))
        oscillation = 0.5 * math.sin(2.0 * turn) * max(0.0, mu)

        primary_kws = self.topic_keywords.get(self.primary_topic) or ["evidence"]

        off_kws: list[str] = []
        for t in self.off_topics:
            off_kws.extend(self.topic_keywords.get(t, []))
        if not off_kws:
            off_kws = primary_kws

        # Occasionally boost the vocabulary with words pulled from the current
        # retrieved evidence so the agent's phrasing tracks the corpus too.
        evidence_kws: list[str] = []
        for e in (evidence or [])[-3:]:
            evidence_kws.extend(self._extract_keywords(e, top_n=4))

        if self.rng.random() < p_drift + oscillation * 0.2:
            a = self.rng.choice(off_kws)
            b = self.rng.choice(primary_kws)
            query = f"how does {a} relate to {b}"
            understanding = (
                f"The goal may actually concern {a}, with {b} as a secondary factor."
            )
        else:
            pool = list(dict.fromkeys(primary_kws + evidence_kws)) or primary_kws
            a, b = self.rng.sample(pool, k=min(2, len(pool)))
            query = f"what is the contribution of {a} versus {b}"
            understanding = f"{a} and {b} are the main drivers."

        return LLMResponse(query=query, understanding=understanding)


# ---------------------------------------------------------------------------
# Real LLM backends (unchanged)
# ---------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, original_goal, understanding, evidence, turn, max_turns=60):
        from src.agent.prompts import SYSTEM_PROMPT, QUERY_TEMPLATE
        prompt = QUERY_TEMPLATE.format(
            original_goal=original_goal,
            understanding=understanding,
            k=min(5, len(evidence)),
            evidence="\n".join(f"- {e}" for e in evidence[-5:]) or "(none yet)",
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(resp.choices[0].message.content)
        return LLMResponse(query=data["query"], understanding=data["understanding"])


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = None):
        import requests
        self._requests = requests
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")

    def generate(self, original_goal, understanding, evidence, turn, max_turns=60):
        from src.agent.prompts import SYSTEM_PROMPT, QUERY_TEMPLATE
        prompt = QUERY_TEMPLATE.format(
            original_goal=original_goal,
            understanding=understanding,
            k=min(5, len(evidence)),
            evidence="\n".join(f"- {e}" for e in evidence[-5:]) or "(none yet)",
        )
        r = self._requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "format": "json",
                "stream": False,
            },
        )
        data = json.loads(r.json()["message"]["content"])
        return LLMResponse(query=data["query"], understanding=data["understanding"])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_llm(backend: str = None, **kwargs) -> BaseLLM:
    backend = backend or os.getenv("LLM_BACKEND", "mock")
    if backend == "mock":
        allowed = {"drift_pressure", "seed", "corpus", "goal"}
        return MockLLM(**{k: v for k, v in kwargs.items() if k in allowed})
    if backend == "openai":
        return OpenAILLM()
    if backend == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM backend: {backend}")