"""
End-to-end prototype:
  1. Build corpus + retriever + embedder
  2. Run the agent loop (mock LLM by default, real LLM if configured)
  3. Extract per-turn signals
  4. Compute EWS on each signal
  5. Project trajectory to 2D, estimate Stuart–Landau mu(t)
  6. Save episode, metrics, and figures
"""

# --- Bootstrap: make `src` importable no matter where you run from ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------

import os
import json
import yaml
import numpy as np
import pandas as pd

from src.retrieval.corpus import load_corpus
from src.retrieval.retriever import Retriever
from src.signals.embeddings import Embedder
from src.agent.llm import get_llm
from src.agent.loop import AgentLoop
from src.signals.extractors import extract_all_signals
from src.signals.ews import compute_ews
from src.dynamical.projection import project_trajectory, radial_amplitude
from src.dynamical.stuart_landau_fit import estimate_mu
from src.viz.plots import plot_signals_with_ews, plot_phase_portrait, plot_mu_estimate

def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # 1. Corpus, retriever, embedder
    docs = load_corpus("data/corpus.jsonl")
    retriever = Retriever(docs)
    embedder = Embedder().fit([d["text"] for d in docs] + [cfg["agent"]["original_goal"]])

    # 2. Agent loop
    llm = get_llm(cfg["agent"].get("backend"), drift_pressure=cfg["agent"]["drift_pressure"],
                  seed=cfg["agent"]["seed"])
    loop = AgentLoop(retriever, embedder, llm, cfg["agent"])
    print(f"Running agent loop with backend={llm.__class__.__name__} ...")
    ep = loop.run(cfg["agent"]["original_goal"], max_turns=cfg["agent"]["max_turns"])
    print(f"Ran {len(ep.turns)} turns.")

    # 3. Signals
    signals = extract_all_signals(ep)

    # --- Robust drift-onset detection (smoothed + thresholded) ----------
    offt = signals["retrieval_offtopic"]
    w = int(cfg["signals"].get("drift_onset_window", 5))
    thr = float(cfg["signals"].get("drift_onset_threshold", 0.30))

    # causal rolling mean
    offt_smooth = np.array([
        offt[max(0, i - w + 1):i + 1].mean() for i in range(len(offt))
    ])
    exceeds = offt_smooth > thr
    if exceeds.any():
        drift_onset = int(np.argmax(exceeds))
    else:
        # Fallback: peak of the smoothed signal (still informative)
        drift_onset = int(np.argmax(offt_smooth))

    print(f"Detected drift onset at turn: {drift_onset} "
          f"(offtopic peak = {offt.max():.2f}, "
          f"smoothed peak = {offt_smooth.max():.2f})")
    # ---------------------------------------------------------------------

    # 4. EWS
    ews_by_signal = {
        name: compute_ews(sig,
                          window=cfg["signals"]["window"],
                          detrend_method=cfg["ews"]["detrend"],
                          bandwidth=cfg["ews"]["bandwidth"])
        for name, sig in signals.items()
    }

    # 5. Dynamical projection + mu(t)
    Z = project_trajectory(ep)
    r = radial_amplitude(Z)
    mu = estimate_mu(Z)

    # 6. Save everything
    pd.DataFrame({**{k: v for k, v in signals.items()},
        "offtopic_smooth": offt_smooth,
    }).to_csv(out_dir / "metrics" / "signals.csv", index=False)
    
    pd.DataFrame({"pc1": Z[:, 0], "pc2": Z[:, 1], "mu_hat": mu, "r": r}).to_csv(
        out_dir / "metrics" / "phase_space.csv", index=False)
    summary = {
        "drift_onset_turn": drift_onset,
        **{f"tau_{k}": v for k, v in {
            "variance_goal_drift": ews_by_signal["goal_drift"]["tau_variance"],
            "autocorr_goal_drift": ews_by_signal["goal_drift"]["tau_autocorr"],
            "variance_offtopic": ews_by_signal["retrieval_offtopic"]["tau_variance"],
        }.items()},
    }
    pd.DataFrame([summary]).to_csv(out_dir / "metrics" / "summary.csv", index=False)

    with open(out_dir / "episodes" / "episode.json", "w") as f:
        json.dump({
            "original_goal": ep.original_goal,
            "turns": [
                {"turn": t.turn, "query": t.query, "understanding": t.understanding,
                 "evidence_topics": t.evidence_topics, "evidence_texts": t.evidence_texts}
                for t in ep.turns
            ],
        }, f, indent=2)

    # 7. Figures
    plot_signals_with_ews(signals, ews_by_signal,
                          save_path=str(out_dir / "figures" / "signals_ews.png"),
                          drift_onset=drift_onset)
    plot_phase_portrait(Z, save_path=str(out_dir / "figures" / "phase_portrait.png"),
                        drift_onset=drift_onset)
    plot_mu_estimate(mu, save_path=str(out_dir / "figures" / "mu_estimate.png"),
                     drift_onset=drift_onset)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFigures in {out_dir/'figures'}, metrics in {out_dir/'metrics'}, "
          f"episode in {out_dir/'episodes'}.")


if __name__ == "__main__":
    main()