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