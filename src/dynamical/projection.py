"""Project the agent's semantic trajectory to 2D via PCA for phase-space analysis."""

import numpy as np
from sklearn.decomposition import PCA


def project_trajectory(ep) -> np.ndarray:
    """Stack goal/query/understanding/evidence embeddings and reduce to 2D."""
    X = np.stack([
        np.concatenate([t.understanding_emb, t.query_emb, t.evidence_emb])
        for t in ep.turns
    ])
    X = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2)
    return pca.fit_transform(X)


def radial_amplitude(Z: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Z, axis=1)