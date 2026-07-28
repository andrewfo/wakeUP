"""Evaluation metrics (Phase 5).

Window-level detection quality: PR-AUC, ROC-AUC, and FPR at a fixed recall
operating point — plus a per-attack-type breakdown, computed by scoring each
attack family against the clean windows only (so one attack's difficulty is
not diluted by the others).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def fpr_at_recall(y_true: np.ndarray, scores: np.ndarray, recall: float = 0.90) -> float:
    """False-positive rate at the threshold achieving >= ``recall`` (TPR)."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= recall
    if not mask.any():
        return 1.0
    return float(np.min(fpr[mask]))


def evaluate_scores(
    y_true: np.ndarray, scores: np.ndarray, recall_target: float = 0.90
) -> dict[str, float]:
    """Core detection metrics for one score vector."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        f"fpr_at_recall_{int(recall_target * 100)}": float("nan"),
    }
    if 0 < y_true.sum() < len(y_true):
        out["pr_auc"] = float(average_precision_score(y_true, scores))
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out[f"fpr_at_recall_{int(recall_target * 100)}"] = fpr_at_recall(
            y_true, scores, recall_target
        )
    return out


def per_attack_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_type_per_window: np.ndarray,
    recall_target: float = 0.90,
) -> pd.DataFrame:
    """Per-attack-type metrics, each scored against clean windows only.

    ``attack_type_per_window`` gives the dominant attack label for each window
    ("none" for clean windows). For each non-clean type we build a binary task
    of {that type vs clean} and report the standard metrics.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    attack_type_per_window = np.asarray(attack_type_per_window)

    clean_mask = labels == 0
    rows = []
    for atype in sorted(set(attack_type_per_window[labels == 1])):
        sel = clean_mask | (attack_type_per_window == atype)
        m = evaluate_scores(labels[sel], scores[sel], recall_target)
        m = {"attack_type": atype, **m}
        rows.append(m)
    # overall row
    overall = {"attack_type": "ALL", **evaluate_scores(labels, scores, recall_target)}
    rows.append(overall)
    return pd.DataFrame(rows)


def dominant_attack_type(windows: pd.DataFrame) -> pd.Series:
    """Per-window dominant attack type (the non-'none' type if any)."""
    def _dom(g):
        vals = [a for a in g["attack_type"].unique() if a != "none"]
        return vals[0] if vals else "none"

    return windows.groupby("window_id", sort=True).apply(
        _dom, include_groups=False
    )
