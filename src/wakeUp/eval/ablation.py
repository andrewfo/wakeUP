"""Features-vs-learned-vs-supervision ablation (Phase 5).

The headline result — "the supervised Transformer buys about a decade of attack
subtlety over IsolationForest" — confounds two things the Transformer changes at
once: it swaps hand-built aggregate features for a **learned sequence
representation**, *and* it trains on **labels** the unsupervised baselines never
see. This module decomposes that gain with a 2×2 over the two axes:

                    | unsupervised            | supervised
    ----------------+-------------------------+---------------------------
    hand features   | IsolationForest         | LogisticFeatureDetector
    learned sequence| ReconTransformer        | TransformerDetector

Reading the grid:

* **left → right** (holding the row fixed) isolates the value of *supervision*:
  IForest → Logistic on the features row, ReconTransformer → Transformer on the
  learned row.
* **top → bottom** (holding the column fixed) isolates the value of the *learned
  sequence representation*: IForest → ReconTransformer unsupervised, Logistic →
  Transformer supervised.

The two supervised cells share a **linear** head (sklearn ``LogisticRegression``
vs the Transformer's single linear ``cls_head`` on the pooled encoding), so the
supervised comparison is representation-vs-representation with the classifier
held fixed — the cleanest available control.

Every cell runs under one held-out-by-vessel protocol so the columns are
comparable; the supervised cells train on labelled train vessels and are scored
on unseen ones. ``run_ablation_sweeps`` reuses the robustness harness to trace
the same grid across attack subtlety, which is where the saturated headline
table finally separates the four.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from wakeUp.attacks import build_attacked_dataset
from wakeUp.config import AttackConfig, ModelConfig
from wakeUp.eval.metrics import dominant_attack_type, per_attack_metrics
from wakeUp.eval.robustness import run_robustness_sweeps
from wakeUp.eval.splits import train_test_split_by_vessel
from wakeUp.features import build_feature_matrix

# Each cell: display name -> (representation, supervision, factory). The factory
# is zero-arg so every use gets a fresh, unfitted instance.
AblationCell = tuple[str, str, Callable[[], object]]


def ablation_cells(
    model_cfg: ModelConfig | None = None, include_lstm: bool = False
) -> dict[str, AblationCell]:
    """The 2×2 ablation grid (plus the LSTM-AE as an optional learned/unsup cell).

    The learned cells need the ``learned`` extra, so they are imported lazily and
    the grid still constructs without torch for the two hand-feature cells.
    """
    cfg = model_cfg or ModelConfig()
    from wakeUp.models import (
        IsolationForestDetector,
        LogisticFeatureDetector,
        ReconTransformerDetector,
        TransformerDetector,
    )

    cells: dict[str, AblationCell] = {
        "IForest": ("features", "unsupervised", lambda: IsolationForestDetector(cfg)),
        "Logistic": ("features", "supervised", lambda: LogisticFeatureDetector(cfg)),
        "ReconTransformer": (
            "learned",
            "unsupervised",
            lambda: ReconTransformerDetector(cfg),
        ),
        "Transformer": ("learned", "supervised", lambda: TransformerDetector(cfg)),
    }
    if include_lstm:
        from wakeUp.models import LSTMAutoencoderDetector

        cells["LSTM-AE"] = (
            "learned",
            "unsupervised",
            lambda: LSTMAutoencoderDetector(cfg),
        )
    return cells


def _fit_score_holdout(det, fit_frame, fit_feat, score_frame, score_feat):
    """Fit/score one detector under the held-out protocol.

    Mirrors the branching in :func:`wakeUp.eval.robustness.sweep_attack_severity`
    so the ablation table and the ablation sweeps treat each cell identically:
    sequence models get the per-point frame, supervised models get labels.
    """
    seq = getattr(det, "consumes_windows", False)
    fit_payload = fit_frame if seq else fit_feat
    score_payload = score_frame if seq else score_feat
    if getattr(det, "supports_supervision", False):
        det.fit(fit_payload, supervised=True)
    else:
        det.fit(fit_payload)
    return det.score(score_payload)


def run_ablation_table(
    windows: pd.DataFrame,
    attack_cfg: AttackConfig | None = None,
    model_cfg: ModelConfig | None = None,
    cells: dict[str, AblationCell] | None = None,
    test_frac: float = 0.3,
    split_seed: int = 0,
    recall_target: float = 0.90,
) -> pd.DataFrame:
    """Per-attack held-out metrics for every ablation cell, one split.

    All five attack families are injected together; each cell is fit on the
    train vessels and scored on the held-out vessels, and per-attack PR-AUC /
    ROC-AUC are reported (each attack vs clean windows). Returns a tidy frame
    with ``representation`` / ``supervision`` columns so the grid reads directly.
    """
    grid = cells or ablation_cells(model_cfg)
    attacked = build_attacked_dataset(windows, attack_cfg or AttackConfig())
    train_w, test_w = train_test_split_by_vessel(
        attacked, test_frac=test_frac, seed=split_seed
    )
    fit_feat, _ = build_feature_matrix(train_w)
    score_feat, test_labels = build_feature_matrix(test_w)
    test_dom = dominant_attack_type(test_w).reindex(score_feat.index).to_numpy()

    rows = []
    for name, (rep, sup, make) in grid.items():
        scores = _fit_score_holdout(make(), train_w, fit_feat, test_w, score_feat)
        per_atk = per_attack_metrics(scores, test_labels, test_dom, recall_target)
        for rec in per_atk.to_dict(orient="records"):
            rows.append(
                {
                    "detector": name,
                    "representation": rep,
                    "supervision": sup,
                    **rec,
                }
            )
    return pd.DataFrame(rows)


def run_ablation_sweeps(
    windows: pd.DataFrame,
    attack_cfg: AttackConfig | None = None,
    model_cfg: ModelConfig | None = None,
    cells: dict[str, AblationCell] | None = None,
    test_frac: float = 0.3,
    split_seed: int = 0,
    recall_target: float = 0.90,
) -> pd.DataFrame:
    """Trace the ablation grid across attack subtlety (held-out sweeps).

    Thin wrapper over :func:`wakeUp.eval.robustness.run_robustness_sweeps` with
    ``holdout=True`` and the ablation factories, then annotated with the
    ``representation`` / ``supervision`` axes so the degradation curves can be
    grouped by either.
    """
    grid = cells or ablation_cells(model_cfg)
    factories = {name: make for name, (_, _, make) in grid.items()}
    sweep = run_robustness_sweeps(
        windows,
        attack_cfg=attack_cfg,
        detectors=factories,
        recall_target=recall_target,
        holdout=True,
        test_frac=test_frac,
        split_seed=split_seed,
    )
    axes = {name: (rep, sup) for name, (rep, sup, _) in grid.items()}
    sweep["representation"] = sweep["detector"].map(lambda d: axes[d][0])
    sweep["supervision"] = sweep["detector"].map(lambda d: axes[d][1])
    return sweep
