#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

# Make `src` importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.retrieval.corpus import load_corpus
from src.retrieval.retriever import Retriever
from src.signals.embeddings import Embedder
from src.agent.llm import get_llm
from src.agent.loop import AgentLoop


def parse_args():
    p = argparse.ArgumentParser(description="Run one agentic episode.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    p.add_argument("--goal", default=None, help="Override the original goal string.")
    p.add_argument("--max-turns", type=int, default=None, help="Override max turns.")
    p.add_argument("--backend", default=None, help="LLM backend: mock | openai | ollama.")
    p.add_argument("--drift-pressure", type=float, default=None,
                   help="Mock-LLM drift pressure (0..1+). Ignored by real backends.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (mock LLM).")
    p.add_argument("--top-k", type=int, default=None, help="Docs retrieved per query.")
    p.add_argument("--corpus", default="data/corpus.jsonl", help="Path to corpus JSONL.")
    p.add_argument("--embed-backend", default=None,
                   help="tfidf (default) | sentence-transformers")
    p.add_argument("--output", default="results/episodes/episode.json",
                   help="Where to save the episode JSON.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --- config ---
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.setdefault("agent", {})
    cfg.setdefault("signals", {})
    cfg.setdefault("ews", {})
    cfg.setdefault("output", {"dir": "results"})

    original_goal = args.goal or cfg["agent"].get(
        "original_goal",
        "Identify the primary drivers of sea-level rise over the last 50 years.",
    )
    max_turns = args.max_turns or cfg["agent"].get("max_turns", 60)
    top_k = args.top_k or cfg["agent"].get("top_k", 3)
    drift_pressure = (
        args.drift_pressure
        if args.drift_pressure is not None
        else cfg["agent"].get("drift_pressure", 0.9)
    )
    seed = args.seed if args.seed is not None else cfg["agent"].get("seed", 42)
    backend = args.backend or cfg["agent"].get("backend") or os.getenv("LLM_BACKEND", "mock")

    # Embedder backend override (env var takes priority unless CLI given)
    if args.embed_backend:
        os.environ["EMBED_BACKEND"] = args.embed_backend

    # --- corpus + retriever + embedder ---
    docs = load_corpus(args.corpus)
    retriever = Retriever(docs)
    embedder = Embedder().fit(
        [d["text"] for d in docs] + [original_goal]
    )

    # --- LLM ---
    llm = get_llm(backend, drift_pressure=drift_pressure, seed=seed)

    # --- loop ---
    agent_cfg = {
        "top_k": top_k,
        "drift_pressure": drift_pressure,
        "seed": seed,
        "original_goal": original_goal,
        "max_turns": max_turns,
        "backend": backend,
    }
    loop = AgentLoop(retriever, embedder, llm, agent_cfg)

    if not args.quiet:
        print(f"[run_agent] backend={llm.__class__.__name__} goal='{original_goal[:60]}...'")
        print(f"[run_agent] max_turns={max_turns} top_k={top_k} seed={seed}")

    ep = loop.run(original_goal, max_turns=max_turns)

    # --- save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "original_goal": ep.original_goal,
        "config": {
            "backend": backend,
            "max_turns": max_turns,
            "top_k": top_k,
            "drift_pressure": drift_pressure,
            "seed": seed,
            "embed_backend": embedder.backend,
            "corpus": args.corpus,
        },
        "turns": [
            {
                "turn": t.turn,
                "query": t.query,
                "understanding": t.understanding,
                "evidence_texts": t.evidence_texts,
                "evidence_topics": t.evidence_topics,
            }
            for t in ep.turns
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    # --- console summary ---
    if not args.quiet:
        n_offtopic_turns = sum(
            1 for t in ep.turns
            if any(tp != "sea_level" for tp in t.evidence_topics)
        )
        print(f"[run_agent] ran {len(ep.turns)} turns")
        print(f"[run_agent] turns with any off-topic retrieval: {n_offtopic_turns}")
        print(f"[run_agent] saved -> {out_path}")


if __name__ == "__main__":
    main()