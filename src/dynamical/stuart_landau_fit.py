"""
Estimate the effective Stuart–Landau parameter mu(t) from a 2D trajectory.

For a Stuart–Landau oscillator, r' = mu r - r^3, so
   mu_hat(t) ≈ r'(t)/r(t) + r(t)^2.
A sign change of mu_hat indicates the onset of a limit cycle (drift).
"""

import numpy as np


def estimate_mu(Z: np.ndarray, dt: float = 1.0, smooth: int = 5) -> np.ndarray:
    r = np.linalg.norm(Z, axis=1)
    r = np.convolve(r, np.ones(smooth)/smooth, mode="same")
    dr = np.gradient(r, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = dr / np.maximum(r, 1e-6) + r**2
    return np.nan_to_num(mu)