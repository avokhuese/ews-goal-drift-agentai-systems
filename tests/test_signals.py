import numpy as np
from src.signals.ews import compute_ews, kendall_tau


def test_kendall_tau_monotone():
    y = np.arange(50, dtype=float)
    assert abs(kendall_tau(y) - 1.0) < 1e-6


def test_ews_shapes():
    np.random.seed(0)
    x = np.random.randn(100)
    ews = compute_ews(x, window=10)
    assert len(ews["variance"]) == 100
    assert len(ews["autocorr"]) == 100
    assert "tau_variance" in ews