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
from sklearn.preprocessing import StandardScaler

from wakeguard.config import ModelConfig
from wakeguard.features import kinematic as kin


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
        to keep the wakeguard convention (larger == more likely spoofed).
        """
        X = self.scaler.transform(feat_df[self.columns_].to_numpy())
        return -self.model.score_samples(X)

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(feat_df[self.columns_].to_numpy())
        return (self.model.predict(X) == -1).astype(int)
