#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.agent.loop import Turn, Episode
from src.signals.embeddings import Embedder
from src.signals.extractors import extract_all_signals
from src.signals.ews import compute_ews, kendall_tau
from src.dynamical.projection import project_trajectory, radial_amplitude
from src.dynamical.stuart_landau_fit import estimate_mu
from src.viz.plots import (
    plot_signals_with_ews,
    plot_phase_portrait,
    plot_mu_estimate,
)


# ---------------------------------------------------------------------------
# Episode reconstruction
# ---------------------------------------------------------------------------
def load_episode(path: str, embedder: Embedder, off_topic_labels=("economics",)) -> Episode:
    """
    Rebuild an Episode object from a saved JSON. Queries, understandings and
    evidence texts are re-embedded here so the EWS step is independent of the
    embedder used during the agent run.
    """
    data = json.loads(Path(path).read_text())
    original_goal = data["original_goal"]

    # Fit the embedder on the goal + all texts we'll see
    all_texts = [original_goal]
    for t in data["turns"]:
        all_texts.append(t["query"])
        all_texts.append(t["understanding"])
        all_texts.extend(t.get("evidence_texts", []))
    embedder.fit(all_texts)

    goal_emb = embedder.encode(original_goal)[0]

    turns = []
    for t in data["turns"]:
        q_emb = embedder.encode(t["query"])[0]
        u_emb = embedder.encode(t["understanding"])[0]
        ev_texts = t.get("evidence_texts", [])
        if ev_texts:
            ev_emb = np.mean(embedder.encode(ev_texts), axis=0)
        else:
            ev_emb = np.zeros_like(q_emb)

        turns.append(Turn(
            turn=t["turn"],
            query=t["query"],
            understanding=t["understanding"],
            evidence_texts=ev_texts,
            evidence_topics=t.get("evidence_topics", []),
            goal_emb=goal_emb,
            query_emb=q_emb,
            understanding_emb=u_emb,
            evidence_emb=ev_emb,
        ))

    ep = Episode(original_goal=original_goal, turns=turns, goal_anchor_emb=goal_emb)
    return ep


# ---------------------------------------------------------------------------
# Drift onset heuristic (ground-truth-like label for evaluation)
# ---------------------------------------------------------------------------
def detect_drift_onset(signals: dict, threshold: float = 0.5) -> int | None:
    """
    Drift onset = first turn where the off-topic retrieval fraction exceeds
    `threshold`. For the mock LLM this coincides with the effective Hopf
    bifurcation; for a real LLM it is a heuristic ground-truth label.
    """
    offt = signals.get("retrieval_offtopic")
    if offt is None:
        return None
    idx = np.where(offt > threshold)[0]
    return int(idx[0]) if len(idx) else None


def detection_lag(drift_onset: int | None, ews_curve: np.ndarray,
                  ews_threshold_quantile: float = 0.9) -> int | None:
    """
    First turn where the EWS curve exceeds its own high quantile — as a proxy
    for 'the EWS fired'. Returns the signed lead time (drift_onset - ews_fire).
    Positive => EWS fired before drift onset (good).
    """
    if drift_onset is None or len(ews_curve) < 4:
        return None
    thresh = np.quantile(ews_curve, ews_threshold_quantile)
    idx = np.where(ews_curve > thresh)[0]
    if len(idx) == 0:
        return None
    return int(drift_onset - idx[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute EWS on a saved agent episode.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--episode", default="results/episodes/episode.json",
                   help="Path to an episode JSON produced by run_agent.py.")
    p.add_argument("--output-dir", default="results",
                   help="Base output directory (metrics/, figures/ written under it).")
    p.add_argument("--window", type=int, default=None,
                   help="Rolling window for EWS (overrides config).")
    p.add_argument("--detrend", default=None, choices=[None, "gaussian", "linear"],
                   help="Detrending method (overrides config).")
    p.add_argument("--bandwidth", type=float, default=None,
                   help="Gaussian bandwidth (overrides config).")
    p.add_argument("--offtopic-label", default="economics",
                   help="Topic label treated as off-topic for the retrieval signal.")
    p.add_argument("--offtopic-threshold", type=float, default=0.5,
                   help="Fraction of off-topic retrieved docs that marks drift onset.")
    p.add_argument("--embed-backend", default=None,
                   help="Override EMBED_BACKEND (tfidf | sentence-transformers).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.embed_backend:
        os.environ["EMBED_BACKEND"] = args.embed_backend

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.setdefault("signals", {"window": 15})
    cfg.setdefault("ews", {"detrend": "gaussian", "bandwidth": 4.0})

    window = args.window or cfg["signals"].get("window", 15)
    detrend_method = args.detrend or cfg["ews"].get("detrend", "gaussian")
    bandwidth = args.bandwidth or cfg["ews"].get("bandwidth", 4.0)

    out_dir = Path(args.output_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # --- load + re-embed ---
    if not Path(args.episode).exists():
        raise SystemExit(f"[run_ews] episode not found: {args.episode}. "
                         f"Run scripts/run_agent.py first.")
    embedder = Embedder()
    ep = load_episode(args.episode, embedder)

    if not args.quiet:
        print(f"[run_ews] loaded episode: {args.episode} ({len(ep.turns)} turns)")
        print(f"[run_ews] embedder={embedder.backend}  window={window}  "
              f"detrend={detrend_method}  bandwidth={bandwidth}")

    # --- signals ---
    signals = extract_all_signals(ep)

    # Recompute the off-topic signal with the user-specified label
    if args.offtopic_label != "economics":
        signals["retrieval_offtopic"] = np.array([
            sum(1 for tp in t.evidence_topics if tp == args.offtopic_label)
            / max(1, len(t.evidence_topics))
            for t in ep.turns
        ])

    drift_onset = detect_drift_onset(signals, threshold=args.offtopic_threshold)
    if not args.quiet:
        print(f"[run_ews] drift onset turn (heuristic): {drift_onset}")

    # --- EWS on every signal ---
    ews_by_signal = {
        name: compute_ews(sig, window=window,
                          detrend_method=detrend_method, bandwidth=bandwidth)
        for name, sig in signals.items()
    }

    # --- dynamical projection + Stuart–Landau mu(t) ---
    Z = project_trajectory(ep)
    r = radial_amplitude(Z)
    mu = estimate_mu(Z)

    # --- write metrics ---
    pd.DataFrame(signals).to_csv(out_dir / "metrics" / "signals.csv", index=False)
    pd.DataFrame({
        "pc1": Z[:, 0], "pc2": Z[:, 1],
        "r": r, "mu_hat": mu,
    }).to_csv(out_dir / "metrics" / "phase_space.csv", index=False)

    ews_rows = {}
    for name, ews in ews_by_signal.items():
        ews_rows[f"variance__{name}"] = ews["variance"]
        ews_rows[f"autocorr__{name}"] = ews["autocorr"]
        ews_rows[f"skewness__{name}"] = ews["skewness"]
    pd.DataFrame(ews_rows).to_csv(out_dir / "metrics" / "ews_curves.csv", index=False)

    # --- summary + lead time ---
    summary = {
        "episode": str(args.episode),
        "n_turns": len(ep.turns),
        "drift_onset_turn": drift_onset,
        "window": window,
        "detrend": detrend_method,
        "bandwidth": bandwidth,
        "embed_backend": embedder.backend,
    }
    for name, ews in ews_by_signal.items():
        summary[f"tau_variance__{name}"] = float(kendall_tau(ews["variance"]))
        summary[f"tau_autocorr__{name}"] = float(kendall_tau(ews["autocorr"]))
        summary[f"lead_turns_variance__{name}"] = detection_lag(
            drift_onset, ews["variance"], 0.9)
        summary[f"lead_turns_autocorr__{name}"] = detection_lag(
            drift_onset, ews["autocorr"], 0.9)

    pd.DataFrame([summary]).to_csv(out_dir / "metrics" / "ews_summary.csv", index=False)
    (out_dir / "metrics" / "ews_summary.json").write_text(json.dumps(summary, indent=2))

    # --- figures ---
    plot_signals_with_ews(
        signals, ews_by_signal,
        save_path=str(out_dir / "figures" / "signals_ews.png"),
        drift_onset=drift_onset,
    )
    plot_phase_portrait(
        Z,
        save_path=str(out_dir / "figures" / "phase_portrait.png"),
        drift_onset=drift_onset,
    )
    plot_mu_estimate(
        mu,
        save_path=str(out_dir / "figures" / "mu_estimate.png"),
        drift_onset=drift_onset,
    )

    # --- console summary ---
    if not args.quiet:
        print("\n=== EWS summary ===")
        for k in [
            "drift_onset_turn",
            "tau_variance__goal_drift",
            "tau_autocorr__goal_drift",
            "lead_turns_variance__goal_drift",
            "lead_turns_autocorr__goal_drift",
            "tau_variance__retrieval_offtopic",
        ]:
            if k in summary:
                print(f"  {k}: {summary[k]}")
        print(f"\n  figures -> {out_dir/'figures'}")
        print(f"  metrics -> {out_dir/'metrics'}")


if __name__ == "__main__":
    main()