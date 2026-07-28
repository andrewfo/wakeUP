"""Baseline detectors (Phase 4).

* :class:`KinematicRuleDetector` — an unsupervised physics rule: a window is
  anomalous if it contains any point that violates a plausibility limit
  (speed / accel / turn-rate) or shows large SOG↔position disagreement. Its
  score is a normalised worst-violation ratio, so it also yields a ranking
  for PR/ROC curves rather than a bare 0/1.

* :class:`IsolationForestDetector` — scales the window feature matrix and
  fits sklearn's IsolationForest; higher score == more anomalous.

Both expose ``fit`` / ``score`` returning a per-window anomaly score where
larger == more likely spoofed, so the eval code treats them uniformly.
The LSTM-autoencoder and Transformer land in later phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wakeUp.config import ModelConfig
from wakeUp.features import build_feature_matrix
from wakeUp.features import kinematic as kin


class KinematicRuleDetector:
    """Unsupervised worst-violation rule detector."""

    def __init__(
        self,
        max_speed_kn: float = kin.MAX_PLAUSIBLE_SPEED_KN,
        max_accel_ms2: float = kin.MAX_PLAUSIBLE_ACCEL_MS2,
        max_turn_dps: float = kin.MAX_PLAUSIBLE_TURN_RATE_DPS,
        max_speed_resid_kn: float = 5.0,
    ):
        self.max_speed_kn = max_speed_kn
        self.max_accel_ms2 = max_accel_ms2
        self.max_turn_dps = max_turn_dps
        self.max_speed_resid_kn = max_speed_resid_kn

    def fit(self, feat_df: pd.DataFrame, y=None) -> "KinematicRuleDetector":
        return self  # stateless

    def score(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Max normalised violation ratio across the four rules per window."""
        ratios = np.vstack(
            [
                feat_df["implied_speed_max"].to_numpy() / self.max_speed_kn,
                feat_df["accel_max"].to_numpy() / self.max_accel_ms2,
                feat_df["turn_rate_max"].to_numpy() / self.max_turn_dps,
                feat_df["speed_resid_max"].to_numpy() / self.max_speed_resid_kn,
            ]
        )
        return np.nanmax(ratios, axis=0)

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Hard 0/1 label: any rule exceeded (ratio > 1)."""
        return (self.score(feat_df) > 1.0).astype(int)


class IsolationForestDetector:
    """IsolationForest over the window feature matrix."""

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.cfg.iforest_estimators,
            contamination=self.cfg.iforest_contamination,
            random_state=self.cfg.seed,
            n_jobs=-1,
        )
        self.columns_: list[str] | None = None

    def fit(self, feat_df: pd.DataFrame, y=None) -> "IsolationForestDetector":
        self.columns_ = list(feat_df.columns)
        X = self.scaler.fit_transform(feat_df.to_numpy())
        self.model.fit(X)
        return self

    def score(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Anomaly score, larger == more anomalous.

        sklearn's ``score_samples`` returns *higher for inliers*, so we negate
        to keep the wakeUp convention (larger == more likely spoofed).
        """
        X = self.scaler.transform(feat_df[self.columns_].to_numpy())
        return -self.model.score_samples(X)

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(feat_df[self.columns_].to_numpy())
        return (self.model.predict(X) == -1).astype(int)


class LogisticFeatureDetector:
    """Supervised linear classifier over the window feature matrix.

    The matched features-only control for the Transformer's supervised arm.
    The Transformer's classification head is a single linear layer on the
    mean-pooled encoder output, so the like-for-like hand-features comparison
    is a linear classifier on the aggregate feature vector. Both see the same
    labels, so any gap between them is attributable to the **learned sequence
    representation**, not to supervision — which is exactly what the ablation
    needs to isolate.

    Like the sequence models it declares ``consumes_windows`` and takes the
    per-point windows frame rather than a prebuilt matrix, so the held-out
    harness can hand it labelled windows and it resolves features **and** labels
    internally through the same ``build_feature_matrix`` ordering the harness
    uses for the label vector. ``class_weight="balanced"`` mirrors the
    Transformer's positive-class rebalancing (attacks are a ~15% minority).
    """

    #: Hand this detector the per-point windows frame, not the feature matrix.
    consumes_windows = True
    #: Supervised: the held-out harness passes ``supervised=True`` with labels.
    supports_supervision = True

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=self.cfg.seed
        )
        self.columns_: list[str] | None = None

    def fit(self, windows: pd.DataFrame, y=None, supervised: bool = True):
        feat_df, labels = build_feature_matrix(windows)
        if y is not None:
            labels = np.asarray(y)
        self.columns_ = list(feat_df.columns)
        X = self.scaler.fit_transform(feat_df.to_numpy())
        self.model.fit(X, labels.astype(int))
        return self

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        """Predicted attack probability per window (larger == more anomalous)."""
        if self.columns_ is None:
            raise RuntimeError("LogisticFeatureDetector must be fit before score")
        feat_df, _ = build_feature_matrix(windows)
        X = self.scaler.transform(feat_df[self.columns_].to_numpy())
        return self.model.predict_proba(X)[:, 1]

    def predict(self, windows: pd.DataFrame) -> np.ndarray:
        return (self.score(windows) > 0.5).astype(int)
