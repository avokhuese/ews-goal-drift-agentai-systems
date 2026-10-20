"""Early-warning indicators on scalar time series."""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict


def detrend(x: np.ndarray, method: str = "gaussian", bandwidth: float = 4.0) -> np.ndarray:
    if method == "gaussian":
        trend = gaussian_filter1d(x, sigma=bandwidth)
    elif method == "linear":
        t = np.arange(len(x))
        slope, intercept = np.polyfit(t, x, 1)
        trend = slope * t + intercept
    else:
        trend = np.zeros_like(x)
    return x - trend


def rolling_variance(x, window=15):
    return np.array([np.var(x[max(0, i-window):i+1]) for i in range(len(x))])


def rolling_autocorr(x, lag=1, window=15):
    out = []
    for i in range(len(x)):
        seg = x[max(0, i - window):i + 1]
        if len(seg) > lag + 1:
            a = seg[:-lag]
            b = seg[lag:]
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.corrcoef(a, b)[0, 1]
                out.append(0.0 if np.isnan(c) else float(c))
            else:
                out.append(0.0)
        else:
            out.append(0.0)
    return np.array(out)


def rolling_skew(x, window=15):
    return np.array([stats.skew(x[max(0, i-window):i+1]) for i in range(len(x))])


def kendall_tau(y):
    if len(y) < 3:
        return 0.0
    tau, _ = stats.kendalltau(np.arange(len(y)), y)
    return 0.0 if np.isnan(tau) else float(tau)


def compute_ews(signal: np.ndarray, window: int = 15, detrend_method: str = "gaussian",
                bandwidth: float = 4.0) -> Dict[str, np.ndarray]:
    x = detrend(signal, detrend_method, bandwidth)
    var = rolling_variance(x, window)
    ar1 = rolling_autocorr(x, window=window)
    skw = rolling_skew(x, window)
    return {
        "signal": signal,
        "residual": x,
        "variance": var,
        "autocorr": ar1,
        "skewness": skw,
        "tau_variance": kendall_tau(var),
        "tau_autocorr": kendall_tau(ar1),
        "tau_skewness": kendall_tau(skw),
    }