"""Sequence tensors for the learned models (Phase 3).

The classical baselines consume the fixed-length *window feature vector*
(:func:`~wakeUp.features.kinematic.window_feature_vector`). The learned
sequence models (LSTM-autoencoder, Transformer) instead want the raw per-point
kinematic channels laid out as a dense tensor ``(window, point, channel)``,
normalised per channel so every channel enters the network on a comparable
scale.

:class:`SequenceTensorizer` mirrors the detector ``fit`` / ``transform``
contract: normalisation statistics are estimated on the training windows only
(``fit``) and reused unchanged on held-out windows (``transform``), so no test
information leaks through the per-channel scaling. It is pure NumPy — no torch
dependency — so the tensors can be built (and unit-tested) as part of the
offline benchmark regardless of whether the learned extras are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wakeUp.features.kinematic import point_features

# Per-point channels fed to the sequence models, in a fixed order. These are the
# kinematic-consistency signals from :func:`point_features`; ``point_idx`` is the
# ordering key, not a channel.
SEQUENCE_CHANNELS = [
    "implied_speed_kn",
    "reported_sog_kn",
    "accel_ms2",
    "turn_rate_dps",
    "speed_resid_kn",
    "cog_resid_deg",
    "gap_s",
]


def _window_channel_array(window: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """``(point, channel)`` array of per-point features for one window."""
    pf = point_features(window).sort_values("point_idx").reset_index(drop=True)
    return np.column_stack([pf[c].to_numpy(dtype=float) for c in channels])


def _stack_windows(
    windows: pd.DataFrame, channels: list[str]
) -> tuple[np.ndarray, np.ndarray, list]:
    """Stack every window into ``(n_windows, point, channel)`` with labels/ids.

    Windows are ordered by ``window_id`` and points by ``point_idx`` so the
    layout is deterministic and aligned with :func:`build_feature_matrix`.
    """
    mats = []
    wids = []
    labels = []
    has_label = "window_label" in windows.columns
    for wid, g in windows.groupby("window_id", sort=True):
        mats.append(_window_channel_array(g, channels))
        wids.append(wid)
        labels.append(int(g["window_label"].iloc[0]) if has_label else 0)
    X = np.stack(mats, axis=0)  # (n_windows, L, C)
    return X, np.asarray(labels, dtype=int), wids


@dataclass
class SequenceTensorizer:
    """Build per-channel-normalised ``(window, point, channel)`` tensors.

    ``fit`` estimates each channel's mean/std from the finite values across the
    training windows; ``transform`` applies those stats, replacing non-finite
    entries (the first point of each window has no predecessor, so its Δ-based
    channels are NaN) with ``0.0`` — i.e. the channel mean after standardising.
    """

    channels: list[str] = field(default_factory=lambda: list(SEQUENCE_CHANNELS))
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, windows: pd.DataFrame) -> "SequenceTensorizer":
        X, _, _ = _stack_windows(windows, self.channels)
        flat = X.reshape(-1, X.shape[-1])
        mean = np.zeros(len(self.channels))
        std = np.ones(len(self.channels))
        for c in range(flat.shape[1]):
            col = flat[:, c]
            col = col[np.isfinite(col)]
            if col.size:
                mean[c] = float(col.mean())
                s = float(col.std())
                std[c] = s if s > 1e-8 else 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(
        self, windows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Return ``(X, labels, window_ids)`` with ``X`` shape ``(N, L, C)``."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceTensorizer must be fit before transform")
        X, labels, wids = _stack_windows(windows, self.channels)
        X = (X - self.mean_) / self.std_
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(np.float32), labels, wids

    def fit_transform(
        self, windows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list]:
        return self.fit(windows).transform(windows)


def build_sequence_tensors(
    windows: pd.DataFrame, channels: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list]:
    """Convenience one-shot: fit + transform on the same windows.

    Returns ``(X, labels, window_ids)`` where ``X`` has shape
    ``(n_windows, window_len, n_channels)`` and is per-channel standardised.
    Use :class:`SequenceTensorizer` directly when train/test must share stats.
    """
    tz = SequenceTensorizer() if channels is None else SequenceTensorizer(list(channels))
    return tz.fit_transform(windows)
