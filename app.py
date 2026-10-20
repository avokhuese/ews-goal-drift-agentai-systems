"""
Streamlit dashboard.

Run:
    streamlit run app.py

Panels:
  • Live Run          — configure + run the agent, watch turns stream in
  • Signals & EWS     — per-signal time series with rolling variance / AR(1)
  • Phase Space       — PCA trajectory + estimated Stuart–Landau mu(t)
  • Episode Log       — full turn-by-turn table (query, understanding, evidence)
  • Sweep             — parameter sweep + heatmap of EWS metrics
  • History           — session history, side-by-side comparison
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.retrieval.corpus import load_corpus
from src.retrieval.retriever import Retriever
from src.signals.embeddings import Embedder
from src.agent.llm import get_llm
from src.agent.loop import AgentLoop, Episode
from src.signals.extractors import extract_all_signals
from src.signals.ews import compute_ews, kendall_tau
from src.dynamical.projection import project_trajectory, radial_amplitude
from src.dynamical.stuart_landau_fit import estimate_mu


st.set_page_config(
    page_title="Goal-Drift EWS",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 44px; padding: 0 18px; }
    .drift-badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        background:#fee; color:#a00; font-size:12px; font-weight:600;
    }
    .ok-badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        background:#efe; color:#070; font-size:12px; font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


_DEFAULTS = {
    "episode": None,
    "signals": None,
    "ews_by_signal": None,
    "Z": None,
    "mu": None,
    "drift_onset": None,
    "summary": None,
    "run_cfg": None,
    "history": [],
    "sweep_df": None,
    "custom_corpus": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def build_corpus_retriever_embedder(corpus_path: str, goal: str, embed_backend: str):
    if st.session_state.custom_corpus is not None and corpus_path == "__custom__":
        docs = st.session_state.custom_corpus
    else:
        docs = load_corpus(corpus_path)
    retriever = Retriever(docs)
    embedder = Embedder(backend=embed_backend).fit(
        [d["text"] for d in docs] + [goal]
    )
    return docs, retriever, embedder


def run_agent_episode(cfg: dict, on_turn=None) -> Episode:
    docs, retriever, embedder = build_corpus_retriever_embedder(
        cfg["corpus_path"], cfg["goal"], cfg["embed_backend"]
    )
    llm = get_llm(
        cfg["backend"],
        drift_pressure=cfg["drift_pressure"],
        seed=cfg["seed"],
        corpus=docs,          # <-- mock LLM now sees the corpus
        goal=cfg["goal"],
    )
    loop = AgentLoop(retriever, embedder, llm, {"top_k": cfg["top_k"]})
    return loop.run(cfg["goal"], max_turns=cfg["max_turns"], on_turn=on_turn)


def analyze_episode(ep: Episode, cfg: dict):
    signals = extract_all_signals(ep)

    lbl = cfg["offtopic_label"]
    if lbl != "economics":
        signals["retrieval_offtopic"] = np.array([
            sum(1 for tp in t.evidence_topics if tp == lbl) / max(1, len(t.evidence_topics))
            for t in ep.turns
        ])

    offt = signals["retrieval_offtopic"]
    onset_idx = np.where(offt > cfg["offtopic_threshold"])[0]
    drift_onset = int(onset_idx[0]) if len(onset_idx) else None

    ews_by_signal = {
        name: compute_ews(sig, window=cfg["window"],
                          detrend_method=cfg["detrend"], bandwidth=cfg["bandwidth"])
        for name, sig in signals.items()
    }

    Z = project_trajectory(ep)
    mu = estimate_mu(Z)

    def lead(curve):
        if drift_onset is None or len(curve) < 4:
            return None
        thresh = np.quantile(curve, 0.9)
        idx = np.where(curve > thresh)[0]
        return int(drift_onset - idx[0]) if len(idx) else None

    summary = {
        "n_turns": len(ep.turns),
        "drift_onset_turn": drift_onset,
        "goal": ep.original_goal,
        "backend": cfg["backend"],
        "seed": cfg["seed"],
        "drift_pressure": cfg["drift_pressure"],
        "window": cfg["window"],
        "bandwidth": cfg["bandwidth"],
        "detrend": cfg["detrend"],
        "tau_variance__goal_drift":   float(kendall_tau(ews_by_signal["goal_drift"]["variance"])),
        "tau_autocorr__goal_drift":   float(kendall_tau(ews_by_signal["goal_drift"]["autocorr"])),
        "tau_variance__retrieval_offtopic": float(kendall_tau(ews_by_signal["retrieval_offtopic"]["variance"])),
        "lead_variance__goal_drift":   lead(ews_by_signal["goal_drift"]["variance"]),
        "lead_autocorr__goal_drift":   lead(ews_by_signal["goal_drift"]["autocorr"]),
        "lead_variance__retrieval_offtopic": lead(ews_by_signal["retrieval_offtopic"]["variance"]),
    }

    return signals, ews_by_signal, Z, mu, drift_onset, summary


def _base_layout(fig, title=""):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        height=350,
    )
    return fig


def chart_live(signals):
    fig = go.Figure()
    colors = {
        "goal_drift": "#d62728",
        "retrieval_offtopic": "#ff7f0e",
        "query_novelty": "#1f77b4",
        "evidence_entropy": "#2ca02c",
        "query_step_distance": "#9467bd",
    }
    for name, sig in signals.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(sig))), y=sig,
            mode="lines", name=name,
            line=dict(color=colors.get(name, "gray"), width=2),
        ))
    return _base_layout(fig, "Live signals")


def chart_signals_ews(signals, ews_by_signal, drift_onset):
    """Grid: for each signal — raw signal, rolling variance, rolling autocorr."""
    names = list(signals.keys())
    n = len(names)
    n_cols = 3

    # Flat list of titles, row-major
    titles = []
    for nm in names:
        titles.extend([f"{nm} — signal", f"{nm} — variance", f"{nm} — AR(1)"])

    fig = make_subplots(
        rows=n, cols=n_cols,
        subplot_titles=tuple(titles),
        horizontal_spacing=0.06,
        vertical_spacing=0.05,
        shared_xaxes=True,
    )

    for i, name in enumerate(names, start=1):
        sig = signals[name]
        ews = ews_by_signal[name]
        x = list(range(len(sig)))

        fig.add_trace(
            go.Scatter(x=x, y=sig, mode="lines",
                       line=dict(color="#1f77b4"), showlegend=False),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=ews["variance"], mode="lines",
                       line=dict(color="#d62728"), showlegend=False),
            row=i, col=2,
        )
        fig.add_trace(
            go.Scatter(x=x, y=ews["autocorr"], mode="lines",
                       line=dict(color="#2ca02c"), showlegend=False),
            row=i, col=3,
        )

        if drift_onset is not None:
            for c in (1, 2, 3):
                fig.add_vline(
                    x=drift_onset,
                    line=dict(color="red", dash="dot", width=1),
                    row=i, col=c,
                )

    fig.update_layout(
        template="plotly_white",
        height=260 * n,
        margin=dict(l=40, r=20, t=80, b=40),
        showlegend=False,
    )
    fig.update_annotations(font_size=11)
    return fig


def chart_phase(Z, mu, drift_onset):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Agent trajectory in PCA space", "Estimated μ(t) (Stuart–Landau)"),
        column_widths=[0.5, 0.5],
    )
    fig.add_trace(
        go.Scatter(
            x=Z[:, 0], y=Z[:, 1], mode="markers+lines",
            marker=dict(color=np.arange(len(Z)), colorscale="Viridis",
                        size=8, colorbar=dict(title="turn", x=1.02)),
            line=dict(color="rgba(120,120,120,0.5)", width=1),
            name="trajectory",
        ),
        row=1, col=1,
    )
    if drift_onset is not None:
        fig.add_trace(
            go.Scatter(x=[Z[drift_onset, 0]], y=[Z[drift_onset, 1]],
                       mode="markers",
                       marker=dict(size=18, color="red", symbol="star"),
                       name="drift onset"),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=list(range(len(mu))), y=mu, mode="lines",
                   line=dict(color="#d62728", width=2), name="μ̂(t)"),
        row=1, col=2,
    )
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"), row=1, col=2)
    if drift_onset is not None:
        fig.add_vline(x=drift_onset, line=dict(color="red", dash="dot"),
                      row=1, col=2)

    fig.update_layout(template="plotly_white", height=480,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def chart_sweep_heatmap(sweep_df, param, metric="goal_drift"):
    piv = sweep_df.pivot_table(index=param, columns="turn",
                               values=f"signal__{metric}", aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index,
        colorscale="Reds", colorbar=dict(title=metric),
    ))
    fig.update_layout(
        title=f"{metric} across sweep of {param}",
        xaxis_title="turn", yaxis_title=param,
        template="plotly_white", height=420,
    )
    return fig


def chart_sweep_metric(sweep_df, param, metric):
    grouped = sweep_df.groupby(param)[metric].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped[param], y=grouped["mean"],
                             mode="markers+lines", error_y=dict(
                                 type="data", array=grouped["std"].fillna(0)),
                             line=dict(color="#1f77b4"), name=metric))
    fig.update_layout(
        title=f"{metric} vs {param}",
        xaxis_title=param, yaxis_title=metric,
        template="plotly_white", height=380,
    )
    return fig


with st.sidebar:
    st.title("🧭 Goal-Drift EWS")
    st.caption("Early-warning signals for critical transitions in agentic AI.")

    with st.expander("🎯 Goal & corpus", expanded=True):
        goal = st.text_area(
            "Original goal",
            value="Identify the primary drivers of sea-level rise over the last 50 years.",
            height=80,
        )
        corpus_path = st.text_input("Corpus path (JSONL)", value="data/corpus.jsonl")

        uploaded = st.file_uploader("Upload custom corpus (JSONL)", type="jsonl")
        if uploaded is not None:
            try:
                lines = uploaded.read().decode("utf-8").splitlines()
                st.session_state.custom_corpus = [json.loads(l) for l in lines if l.strip()]
                st.success(f"Loaded {len(st.session_state.custom_corpus)} custom docs.")
            except Exception as e:
                st.error(f"Failed to parse JSONL: {e}")
        import hashlib
        from pathlib import Path as _Path

        st.divider()
        if st.session_state.custom_corpus is not None:
            fp = hashlib.md5(
                json.dumps(st.session_state.custom_corpus, sort_keys=True).encode()
            ).hexdigest()[:8]
            st.caption(
                f"📎 **Uploaded corpus** · {len(st.session_state.custom_corpus)} docs · fp=`{fp}`"
            )
            if st.button("Clear uploaded corpus", use_container_width=True):
                st.session_state.custom_corpus = None
                st.rerun()
        else:
            try:
                content = _Path(corpus_path).read_bytes()
                fp = hashlib.md5(content).hexdigest()[:8]
                st.caption(f"📎 **Disk corpus** · `{corpus_path}` · fp=`{fp}`")
            except FileNotFoundError:
                st.caption(f"⚠️ Corpus not found: `{corpus_path}`")
        offtopic_label = st.selectbox(
            "Off-topic label", ["economics", "noise", "custom"], index=0
        )
        offtopic_threshold = st.slider("Drift onset threshold (off-topic fraction)",
                                       0.1, 1.0, 0.5, 0.05)

    with st.expander("🤖 Agent", expanded=True):
        backend = st.selectbox("LLM backend", ["mock", "openai", "ollama"], index=0)
        max_turns = st.slider("Max turns", 10, 200, 60, 5)
        top_k = st.slider("Top-k retrieved docs", 1, 8, 3)
        drift_pressure = st.slider("Drift pressure (mock only)", 0.0, 1.5, 0.9, 0.05)
        seed = st.number_input("Seed", value=42, step=1)
        embed_backend = st.selectbox(
            "Embedding backend", ["tfidf", "sentence-transformers"], index=0
        )

    with st.expander("📈 EWS", expanded=False):
        window = st.slider("Rolling window", 5, 60, 15, 1)
        detrend = st.selectbox("Detrending", ["gaussian", "linear"], index=0)
        bandwidth = st.slider("Gaussian bandwidth", 1.0, 20.0, 4.0, 0.5)

    st.divider()
    # NEW — works on Streamlit < 1.49
    run_button = st.button("🚀 Run agent", type="primary", use_container_width=True)

    st.caption("Session history: {} runs".format(len(st.session_state.history)))


def _cfg():
    return {
        "goal": goal,
        "corpus_path": "__custom__" if st.session_state.custom_corpus is not None else corpus_path,
        "backend": backend,
        "max_turns": max_turns,
        "top_k": top_k,
        "drift_pressure": drift_pressure,
        "seed": int(seed),
        "embed_backend": embed_backend,
        "window": window,
        "detrend": detrend,
        "bandwidth": bandwidth,
        "offtopic_label": offtopic_label,
        "offtopic_threshold": offtopic_threshold,
    }


tab_live, tab_signals, tab_phase, tab_log, tab_sweep, tab_history = st.tabs([
    "▶️ Live Run", "📈 Signals & EWS", "🌀 Phase Space",
    "📝 Episode Log", "🔬 Sweep", "🗂️ History",
])


with tab_live:
    st.header("Live agent run")
    st.caption("Configure the sidebar, hit **Run agent**, and watch turns stream in.")

    live_chart = st.empty()
    live_metrics = st.empty()
    live_log = st.empty()
    status_slot = st.empty()

    if run_button:
        cfg = _cfg()
        st.session_state.run_cfg = cfg

        turn_buffer = []
        signal_series = {
            "goal_drift": [],
            "retrieval_offtopic": [],
            "query_novelty": [],
            "evidence_entropy": [],
            "query_step_distance": [],
        }

        def on_turn(turn, ep):
            turn_buffer.append(turn)
            if turn.turn == 0:
                signal_series["goal_drift"].append(0.0)
                signal_series["query_novelty"].append(0.0)
                signal_series["query_step_distance"].append(0.0)
            else:
                def cos(a, b):
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    return float(np.dot(a, b) / (na * nb + 1e-12))
                signal_series["goal_drift"].append(
                    1.0 - cos(ep.goal_anchor_emb, turn.understanding_emb))
                prev_q = ep.turns[-2].query_emb
                signal_series["query_novelty"].append(1.0 - cos(turn.query_emb, prev_q))
                signal_series["query_step_distance"].append(1.0 - cos(turn.query_emb, prev_q))

            offt = (sum(1 for tp in turn.evidence_topics
                        if tp == cfg["offtopic_label"])
                    / max(1, len(turn.evidence_topics)))
            signal_series["retrieval_offtopic"].append(offt)

            from collections import Counter
            c = Counter(turn.evidence_topics)
            p = np.array(list(c.values()), dtype=float)
            p /= p.sum() if p.sum() else 1.0
            signal_series["evidence_entropy"].append(float(-np.sum(p * np.log(p + 1e-12))))

            status_slot.markdown(
                f"**Turn {turn.turn + 1}/{cfg['max_turns']}** — "
                f"query: `{turn.query[:90]}`"
            )
            live_chart.plotly_chart(
                chart_live({k: np.array(v) for k, v in signal_series.items()}),
                use_container_width=True,
            )
            live_log.dataframe(
                pd.DataFrame([{
                    "turn": t.turn,
                    "query": t.query[:80],
                    "understanding": t.understanding[:80],
                    "topics": ", ".join(t.evidence_topics),
                } for t in turn_buffer]),
                use_container_width=True, height=220,
            )
            time.sleep(0.02)

        status_slot.info("Running agent…")
        try:
            ep = run_agent_episode(cfg, on_turn=on_turn)
        except Exception as e:
            st.error(f"Agent run failed: {e}")
            st.stop()

        status_slot.success(f"Agent finished — {len(ep.turns)} turns.")

        with st.spinner("Computing EWS, PCA projection, μ(t)…"):
            signals, ews_by_signal, Z, mu, drift_onset, summary = analyze_episode(ep, cfg)

        st.session_state.episode = ep
        st.session_state.signals = signals
        st.session_state.ews_by_signal = ews_by_signal
        st.session_state.Z = Z
        st.session_state.mu = mu
        st.session_state.drift_onset = drift_onset
        st.session_state.summary = summary

        st.session_state.history.append({
            "label": f"{cfg['backend']} · seed={cfg['seed']} · "
                     f"dp={cfg['drift_pressure']} · w={cfg['window']}",
            "summary": summary,
            "signals": signals,
            "ews_by_signal": ews_by_signal,
            "Z": Z, "mu": mu, "drift_onset": drift_onset,
        })

        live_metrics.success(
            f"✅ Drift onset at turn **{drift_onset}** · "
            f"τ(var | goal_drift) = {summary['tau_variance__goal_drift']:.3f} · "
            f"lead(var) = {summary['lead_variance__goal_drift']} turns"
        )

    elif st.session_state.episode is not None:
        s = st.session_state.summary
        st.info(
            f"Last run: {s['n_turns']} turns · drift onset @ turn {s['drift_onset_turn']} · "
            f"τ(var|goal_drift)={s['tau_variance__goal_drift']:.3f}"
        )
    else:
        st.info("No episode yet. Press **Run agent** in the sidebar.")


with tab_signals:
    st.header("Signals & Early-Warning Indicators")
    if st.session_state.signals is None:
        st.info("Run an agent first.")
    else:
        st.plotly_chart(
            chart_signals_ews(
                st.session_state.signals,
                st.session_state.ews_by_signal,
                st.session_state.drift_onset,
            ),
            use_container_width=True,
        )

        rows = []
        for name, ews in st.session_state.ews_by_signal.items():
            rows.append({
                "signal": name,
                "τ(variance)": kendall_tau(ews["variance"]),
                "τ(autocorr)": kendall_tau(ews["autocorr"]),
                "τ(skewness)": kendall_tau(ews["skewness"]),
            })
        st.subheader("Kendall-τ trends (rising ⇒ critical slowing down)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download signals (CSV)",
            pd.DataFrame(st.session_state.signals).to_csv(index=False).encode(),
            "signals.csv", "text/csv",
        )


with tab_phase:
    st.header("Phase-Space Trajectory & Stuart–Landau Fit")
    if st.session_state.Z is None:
        st.info("Run an agent first.")
    else:
        st.plotly_chart(
            chart_phase(st.session_state.Z, st.session_state.mu,
                        st.session_state.drift_onset),
            use_container_width=True,
        )
        st.caption(
            "Left: PCA of per-turn understanding/query/evidence embeddings. "
            "Right: estimated Stuart–Landau μ(t) = r'/r + r². A sign change "
            "(μ crossing 0) anticipates the drift."
        )
        c1, c2 = st.columns(2)
        r = radial_amplitude(st.session_state.Z)
        c1.metric("Final radius r(T)", f"{r[-1]:.3f}")
        c2.metric("μ(T) estimate", f"{st.session_state.mu[-1]:.3f}")


with tab_log:
    st.header("Episode Log")
    if st.session_state.episode is None:
        st.info("Run an agent first.")
    else:
        ep = st.session_state.episode
        rows = []
        for t in ep.turns:
            offt = (sum(1 for tp in t.evidence_topics
                        if tp == st.session_state.run_cfg["offtopic_label"])
                    / max(1, len(t.evidence_topics)))
            rows.append({
                "turn": t.turn,
                "query": t.query,
                "understanding": t.understanding,
                "off_topic_frac": round(offt, 3),
                "topics": ", ".join(t.evidence_topics),
                "evidence": " | ".join(t.evidence_texts)[:180],
            })
        df = pd.DataFrame(rows)

        def _style(row):
            if row["off_topic_frac"] > 0.5:
                return ["background-color:#fee"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(_style, axis=1),
            use_container_width=True, height=560,
        )

        st.download_button(
            "⬇️ Download episode (JSON)",
            json.dumps({
                "original_goal": ep.original_goal,
                "config": st.session_state.run_cfg,
                "turns": [
                    {"turn": t.turn, "query": t.query, "understanding": t.understanding,
                     "evidence_texts": t.evidence_texts, "evidence_topics": t.evidence_topics}
                    for t in ep.turns
                ],
            }, indent=2).encode(),
            "episode.json", "application/json",
        )


with tab_sweep:
    st.header("Parameter Sweep")
    st.caption(
        "Sweep an agent or EWS parameter and see how the EWS metrics respond. "
        "Agent params re-run the agent; EWS params re-analyze the existing episode."
    )

    if st.session_state.episode is None:
        st.info("Run an agent first to establish a baseline.")
    else:
        with st.form("sweep_form"):
            c1, c2, c3 = st.columns(3)
            sweep_param = c1.selectbox(
                "Parameter",
                ["drift_pressure", "seed", "max_turns", "top_k", "window", "bandwidth"],
            )
            n_values = c2.slider("Number of values", 3, 15, 6)
            metric = c3.selectbox(
                "EWS metric",
                ["tau_variance__goal_drift", "tau_autocorr__goal_drift",
                 "lead_variance__goal_drift", "drift_onset_turn",
                 "tau_variance__retrieval_offtopic"],
            )
            submitted = st.form_submit_button("Run sweep", type="primary")

        if submitted:
            base_cfg = dict(st.session_state.run_cfg)

            if sweep_param == "drift_pressure":
                values = np.linspace(0.1, 1.4, n_values)
            elif sweep_param == "seed":
                values = np.arange(base_cfg["seed"], base_cfg["seed"] + n_values)
            elif sweep_param == "max_turns":
                values = np.linspace(20, 120, n_values, dtype=int)
            elif sweep_param == "top_k":
                values = np.arange(1, 1 + n_values)
            elif sweep_param == "window":
                values = np.linspace(5, 50, n_values, dtype=int)
            else:
                values = np.linspace(1.0, 15.0, n_values)

            rows = []
            signal_rows = []
            prog = st.progress(0.0)
            status = st.empty()

            agent_params = {"drift_pressure", "seed", "max_turns", "top_k"}

            for i, v in enumerate(values):
                cfg = dict(base_cfg)
                cfg[sweep_param] = int(v) if sweep_param in {"seed", "max_turns", "top_k"} else float(v)

                status.info(f"[{i+1}/{len(values)}] {sweep_param} = {v}")
                try:
                    if sweep_param in agent_params:
                        ep_i = run_agent_episode(cfg)
                    else:
                        ep_i = st.session_state.episode
                    _, _, _, _, onset_i, summ_i = analyze_episode(ep_i, cfg)
                except Exception as e:
                    st.warning(f"Failed at {sweep_param}={v}: {e}")
                    continue

                rows.append({sweep_param: v, **{k: summ_i[k] for k in [
                    "drift_onset_turn",
                    "tau_variance__goal_drift",
                    "tau_autocorr__goal_drift",
                    "tau_variance__retrieval_offtopic",
                    "lead_variance__goal_drift",
                    "lead_autocorr__goal_drift",
                ]}})

                sig_i = extract_all_signals(ep_i)
                for name in ["goal_drift", "retrieval_offtopic"]:
                    for t_idx, val in enumerate(sig_i[name]):
                        signal_rows.append({
                            sweep_param: v, "turn": t_idx,
                            f"signal__{name}": float(val),
                        })

                prog.progress((i + 1) / len(values))

            status.success(f"Sweep complete: {len(rows)} runs.")
            sweep_df = pd.DataFrame(rows)
            signal_df = pd.DataFrame(signal_rows)
            st.session_state.sweep_df = sweep_df

            st.subheader("EWS metric vs parameter")
            st.plotly_chart(chart_sweep_metric(sweep_df, sweep_param, metric),
                            use_container_width=True)

            st.subheader("Signal heatmap (goal drift)")
            st.plotly_chart(
                chart_sweep_heatmap(signal_df, sweep_param, metric="goal_drift"),
                use_container_width=True,
            )

            st.subheader("Raw sweep table")
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)


with tab_history:
    st.header("Session History")
    if not st.session_state.history:
        st.info("No runs yet this session.")
    else:
        hist_rows = []
        for i, h in enumerate(st.session_state.history):
            s = h["summary"]
            hist_rows.append({
                "#": i,
                "label": h["label"],
                "turns": s["n_turns"],
                "drift_onset": s["drift_onset_turn"],
                "τ(var|gd)": round(s["tau_variance__goal_drift"], 3),
                "τ(ar1|gd)": round(s["tau_autocorr__goal_drift"], 3),
                "lead(var)": s["lead_variance__goal_drift"],
                "lead(ar1)": s["lead_autocorr__goal_drift"],
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

        fig = go.Figure()
        for i, h in enumerate(st.session_state.history):
            sig = h["signals"]["goal_drift"]
            fig.add_trace(go.Scatter(
                x=list(range(len(sig))), y=sig,
                mode="lines", name=h["label"],
            ))
        fig.update_layout(
            title="goal_drift signal across runs",
            xaxis_title="turn", yaxis_title="goal_drift",
            template="plotly_white", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()