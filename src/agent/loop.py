"""
The agentic loop:
   goal → generate query → retrieve evidence → revise understanding → repeat.

Every turn we record observables:
   - the query string
   - the understanding string
   - the retrieved docs (with their topic labels)
   - embeddings of goal, query, understanding, evidence
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from src.agent.llm import get_llm
from src.retrieval.retriever import Retriever
from src.signals.embeddings import Embedder, cosine


@dataclass
class Turn:
    turn: int
    query: str
    understanding: str
    evidence_texts: list
    evidence_topics: list
    goal_emb: Optional[np.ndarray] = None
    query_emb: Optional[np.ndarray] = None
    understanding_emb: Optional[np.ndarray] = None
    evidence_emb: Optional[np.ndarray] = None


@dataclass
class Episode:
    original_goal: str
    turns: list = field(default_factory=list)
    goal_anchor_emb: Optional[np.ndarray] = None


class AgentLoop:
    """Multi-turn query → retrieve → revise loop."""

    def __init__(self, retriever: Retriever, embedder: Embedder, llm, config: dict):
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm
        self.config = config or {}

    # ------------------------------------------------------------------ run
    def run(
        self,
        original_goal: str,
        max_turns: int = 60,
        on_turn: Optional[Callable[[Turn, Episode], None]] = None,
    ) -> Episode:
        """
        Execute the loop for `max_turns` turns.

        Parameters
        ----------
        original_goal : str
        max_turns     : int
        on_turn       : optional callable(turn_obj, episode) invoked after each turn.
                        Used by the Streamlit app to stream UI updates.
        """
        ep = Episode(original_goal=original_goal)
        ep.goal_anchor_emb = self.embedder.encode(original_goal)[0]

        understanding = "No prior understanding."
        evidence_texts = []
        top_k = int(self.config.get("top_k", 3))

        for t in range(max_turns):
            # 1) LLM generates next query + revised understanding
            resp = self.llm.generate(
                original_goal=original_goal,
                understanding=understanding,
                evidence=evidence_texts,
                turn=t,
                max_turns=max_turns,
            )

            # 2) Retrieval
            retrieved = self.retriever.search(resp.query, top_k=top_k)
            retrieved_texts = [d["text"] for d in retrieved]
            retrieved_topics = [d.get("topic", "unknown") for d in retrieved]

            # 3) Embed this turn's artifacts
            q_emb = self.embedder.encode(resp.query)[0]
            u_emb = self.embedder.encode(resp.understanding)[0]
            if retrieved_texts:
                ev_emb = np.mean(self.embedder.encode(retrieved_texts), axis=0)
            else:
                ev_emb = np.zeros_like(q_emb)

            turn_obj = Turn(
                turn=t,
                query=resp.query,
                understanding=resp.understanding,
                evidence_texts=retrieved_texts,
                evidence_topics=retrieved_topics,
                goal_emb=ep.goal_anchor_emb,
                query_emb=q_emb,
                understanding_emb=u_emb,
                evidence_emb=ev_emb,
            )
            ep.turns.append(turn_obj)

            # 4) Callback for live UI
            if on_turn is not None:
                try:
                    on_turn(turn_obj, ep)
                except Exception as e:
                    # never let a UI callback kill the run
                    print(f"[AgentLoop] on_turn callback raised: {e}")

            # 5) Advance state
            understanding = resp.understanding
            evidence_texts.extend(retrieved_texts)

        return ep