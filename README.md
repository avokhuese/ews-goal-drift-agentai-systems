# Early-Warning Signals for Critical Transitions in Agentic AI Systems
# Project Summary
Multi-turn large language model (LLM) agents that iteratively query structured
sources, retrieve evidence, and revise their understanding can silently drift
from their original goal. We ask whether such drift can be anticipated as a
loss of stability in a low-dimensional dynamical system, and we build a
reproducible prototype to test this. The agent loop is formalised as a
discrete-time map on semantic space. We extract five per-turn observables
without architectural modification; goal drift, query novelty, query step
distance, evidence entropy, and retrieval off-topicness, and apply classical
early-warning indicators (rolling variance, lag-1 autocorrelation, skewness,
and Kendall-rank trends) to each. A Stuart-Landau normal form is fitted to
the PCA-projected semantic trajectory as a diagnostic for a Hopf-type critical
transition. The prototype ships a corpus-aware mock LLM backends behind one loop. On a representative 60-turn episode we observe a sharp regime
change in the semantic channel and strongly heterogeneous early-warning
behaviour: goal drift and retrieval off-topicness show rising variance
(Kendall-rank = +0.67$) while query novelty and evidence entropy
move in the opposite direction, and lag-1 autocorrelation is flat or negative
on every signal. No single observable is a reliable detector, which motivates
multi-signal monitoring. 

## How to Run
pip install -r requirements.txt

# Default: mock LLM, TF-IDF embeddings, no API key needed
python scripts/run_full_prototype.py


# Better embeddings
export EMBED_BACKEND=sentence-transformers
python scripts/run_full_prototype.py

# Produce one episode
python scripts/run_agent.py \
    --goal "Identify the primary drivers of sea-level rise over the last 50 years." \
    --max-turns 60 --backend mock --seed 42 \
    --output results/episodes/episode_seed42.json

# Analyze it
python scripts/run_ews.py \
    --episode results/episodes/episode_seed42.json \
    --output-dir results \
    --window 15 --detrend gaussian --bandwidth 4.0

# Swap embedder without re-running the agent (episode JSON is embedder-agnostic)
EMBED_BACKEND=sentence-transformers python scripts/run_ews.py \
    --episode results/episodes/episode_seed42.json \
    --output-dir results_st

# Real LLM
LLM_BACKEND=openai OPENAI_API_KEY=sk-... python scripts/run_agent.py \
    --max-turns 40 --output results/episodes/episode_openai.json
python scripts/run_ews.py --episode results/episodes/episode_openai.json

# Streamlit dashboard.

# Run:
    streamlit run app.py
