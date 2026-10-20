import matplotlib.pyplot as plt
import numpy as np


def plot_signals_with_ews(signals: dict, ews_by_signal: dict, save_path: str,
                          drift_onset: int = None):
    n = len(signals)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n), sharex=True)

    for i, (name, sig) in enumerate(signals.items()):
        axes[i, 0].plot(sig, color="C0")
        axes[i, 0].set_ylabel(name)
        if drift_onset is not None:
            axes[i, 0].axvline(drift_onset, color="red", ls="--", alpha=0.6)

        ews = ews_by_signal[name]
        axes[i, 1].plot(ews["variance"], color="C1")
        axes[i, 1].set_ylabel("variance")

        axes[i, 2].plot(ews["autocorr"], color="C2")
        axes[i, 2].set_ylabel("AR(1)")

        for j in range(3):
            axes[i, j].grid(alpha=0.3)

    axes[0, 0].set_title("Signal")
    axes[0, 1].set_title("EWS: rolling variance")
    axes[0, 2].set_title("EWS: rolling autocorrelation (lag-1)")
    axes[-1, 0].set_xlabel("turn")
    axes[-1, 1].set_xlabel("turn")
    axes[-1, 2].set_xlabel("turn")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_phase_portrait(Z: np.ndarray, save_path: str, drift_onset: int = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Z[:, 0], Z[:, 1], "-", alpha=0.6)
    ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(len(Z)), cmap="viridis", s=20)
    if drift_onset is not None:
        ax.scatter(Z[drift_onset, 0], Z[drift_onset, 1], c="red", s=120, marker="*", label="drift onset")
        ax.legend()
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Agent goal trajectory (PCA of understanding/query/evidence)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mu_estimate(mu: np.ndarray, save_path: str, drift_onset: int = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mu, color="C3")
    ax.axhline(0, color="black", lw=0.8)
    if drift_onset is not None:
        ax.axvline(drift_onset, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("turn")
    ax.set_ylabel(r"$\hat{\mu}(t)$")
    ax.set_title("Estimated Stuart–Landau parameter (sign change ⇒ critical transition)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()