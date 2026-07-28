"""Figure generation (Phase 6).

All figures are written to disk from results so ``make figures`` reproduces
them deterministically. Matplotlib only (no seaborn), Agg-safe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from wakeguard.features import kinematic as kin  # noqa: E402

# A small colourblind-safe categorical set (Okabe–Ito subset).
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _ensure(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_pr_curves(
    labels: np.ndarray,
    score_dict: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Precision–Recall by detector",
) -> Path:
    """Overlay PR curves for several detectors on one axis."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    base_rate = float(np.mean(labels))
    for i, (name, scores) in enumerate(score_dict.items()):
        p, r, _ = precision_recall_curve(labels, scores)
        ax.plot(r, p, color=_PALETTE[i % len(_PALETTE)], lw=2, label=name)
    ax.axhline(base_rate, ls="--", lw=1, color="0.5", label=f"chance ({base_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = _ensure(out_path)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_score_hist(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str | Path,
    title: str = "Anomaly-score separation",
) -> Path:
    """Histogram of anomaly scores split by clean / attacked."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    clean = scores[labels == 0]
    atk = scores[labels == 1]
    bins = np.linspace(np.min(scores), np.max(scores), 40)
    ax.hist(clean, bins=bins, color=_PALETTE[0], alpha=0.6, label="clean", density=True)
    ax.hist(atk, bins=bins, color=_PALETTE[1], alpha=0.6, label="attacked", density=True)
    ax.set_xlabel("anomaly score (larger = more anomalous)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = _ensure(out_path)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_window_example(
    clean_window: pd.DataFrame,
    attacked_window: pd.DataFrame,
    out_path: str | Path,
    title: str = "Attack example",
) -> Path:
    """Side-by-side: track geometry (clean vs attacked) + speed consistency."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140)

    ax = axes[0]
    ax.plot(
        clean_window["lon"], clean_window["lat"],
        "-o", ms=3, color=_PALETTE[0], label="clean",
    )
    atk = attacked_window.sort_values("point_idx")
    is_atk = atk["is_attack"].to_numpy().astype(bool) if "is_attack" in atk else np.zeros(len(atk), bool)
    ax.plot(atk["lon"], atk["lat"], "-", color=_PALETTE[1], lw=1, label="attacked")
    ax.scatter(
        atk["lon"][is_atk], atk["lat"][is_atk],
        color=_PALETTE[1], s=40, zorder=5, label="spoofed points",
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("Track geometry")
    ax.legend(frameon=False, fontsize=8)
    ax.ticklabel_format(useOffset=False, style="plain")

    ax = axes[1]
    pf = kin.point_features(atk)
    ax.plot(pf["point_idx"], pf["implied_speed_kn"], "-o", ms=3,
            color=_PALETTE[2], label="implied speed (Δpos)")
    ax.plot(pf["point_idx"], pf["reported_sog_kn"], "-s", ms=3,
            color=_PALETTE[3], label="reported SOG")
    ax.set_xlabel("point index")
    ax.set_ylabel("speed (kn)")
    ax.set_title("Speed consistency")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    p = _ensure(out_path)
    fig.savefig(p)
    plt.close(fig)
    return p
