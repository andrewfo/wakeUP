"""Transformer encoder over track windows — the main model (Phase 4).

Self-attention over the per-point channel tensor, with **two heads**:

* a **reconstruction** head (per-position linear projection back to channel
  space), giving the same unsupervised signal as the LSTM autoencoder;
* a **classification** head (mean-pooled encoding -> single logit), trained on
  window labels when they are supplied.

The two-head design is a direct response to the LSTM-AE result: reconstruction
error catches discontinuities but is structurally blind to smooth global
offsets like ``gradual_drift``, because a uniformly shifted window is exactly
as easy to reconstruct as a clean one. A discriminative head has no such blind
spot — it can learn that a constant SOG-vs-Δposition residual is itself the
tell — while the reconstruction term keeps the encoder grounded when labels are
scarce.

Protocol, and the reason this model needs care the others do not: ``fit(X)``
with no labels trains reconstruction only and stays unsupervised, directly
comparable to the LSTM-AE. ``fit(X, y)`` adds the classification loss and makes
the model **supervised**, so it must be scored on held-out vessels — see
:mod:`wakeUp.eval.splits`. Scoring a supervised model on its training windows
measures memorisation, not detection.

Score orientation follows the project convention (larger == more anomalous):
the classifier probability when supervised, per-window reconstruction error
otherwise.

Requires the ``learned`` extra: ``pip install -e ".[learned]"``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # torch is an optional extra; fail with an actionable message
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "TransformerDetector requires PyTorch. Install the learned extra:\n"
        '    pip install -e ".[learned]"'
    ) from exc

from wakeUp.config import ModelConfig
from wakeUp.features.sequences import SequenceTensorizer


def _sinusoidal_encoding(length: int, d_model: int) -> "torch.Tensor":
    """Fixed sinusoidal positional encoding, ``(1, length, d_model)``.

    Fixed rather than learned so the model carries no position parameters to
    overfit on a 32-point window, and so it transfers unchanged if the window
    length is reconfigured.
    """
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe.unsqueeze(0)


class _TrackTransformer(nn.Module):
    """Encoder + reconstruction and classification heads."""

    def __init__(self, n_channels: int, cfg: ModelConfig):
        super().__init__()
        d = cfg.tf_d_model
        self.input_proj = nn.Linear(n_channels, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.tf_nhead,
            dim_feedforward=cfg.tf_ff,
            dropout=cfg.tf_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.tf_layers, enable_nested_tensor=False
        )
        self.recon_head = nn.Linear(d, n_channels)
        self.cls_head = nn.Linear(d, 1)
        self.register_buffer("pe", _sinusoidal_encoding(512, d), persistent=False)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        h = self.input_proj(x) + self.pe[:, : x.shape[1]]
        h = self.encoder(h)
        recon = self.recon_head(h)              # (B, L, C)
        logit = self.cls_head(h.mean(dim=1)).squeeze(-1)  # (B,)
        return recon, logit


class TransformerDetector:
    """Transformer encoder with reconstruction + classification heads."""

    #: Hand this detector the per-point windows frame, not the feature matrix.
    consumes_windows = True
    #: This detector can use labels — harnesses running a held-out protocol
    #: pass ``supervised=True``. Without it the model stays unsupervised.
    supports_supervision = True

    def __init__(self, cfg: ModelConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or ModelConfig()
        self.device = device
        self.tensorizer_: SequenceTensorizer | None = None
        self.model_: _TrackTransformer | None = None
        self.supervised_: bool = False
        self.threshold_: float | None = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def _to_tensor(self, data, fit_stats: bool) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if fit_stats:
                self.tensorizer_ = SequenceTensorizer()
                X, _, _ = self.tensorizer_.fit_transform(data)
            else:
                if self.tensorizer_ is None:
                    raise RuntimeError(
                        "detector was fit on arrays; pass an array to score too"
                    )
                X, _, _ = self.tensorizer_.transform(data)
            return X
        X = np.asarray(data, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"expected (n_windows, n_points, n_channels), got {X.shape}")
        return X

    @staticmethod
    def _labels_from(data, y) -> np.ndarray | None:
        """Resolve labels from ``y``, or from the frame's ``window_label``."""
        if y is not None:
            return np.asarray(y, dtype=np.float32)
        if isinstance(data, pd.DataFrame) and "window_label" in data.columns:
            return (
                data.groupby("window_id", sort=True)["window_label"]
                .first()
                .to_numpy()
                .astype(np.float32)
            )
        return None

    # ------------------------------------------------------------------ #
    # fit / score
    # ------------------------------------------------------------------ #
    def fit(self, data, y=None, supervised: bool = False) -> "TransformerDetector":
        """Train the encoder.

        ``supervised=False`` (default) trains reconstruction only, so the
        detector stays unsupervised and drop-in comparable with the other
        baselines — including inside the robustness harness, which calls
        ``fit(payload)`` with no labels. ``supervised=True`` adds the
        classification loss; score it on held-out vessels.
        """
        X = self._to_tensor(data, fit_stats=True)
        labels = self._labels_from(data, y) if supervised else None
        if supervised and labels is None:
            raise ValueError("supervised=True needs labels via y or window_label")
        self.supervised_ = supervised

        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        n, _, n_channels = X.shape
        self.model_ = _TrackTransformer(n_channels, self.cfg).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.cfg.tf_lr)
        mse = nn.MSELoss()
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = (
            torch.as_tensor(labels, dtype=torch.float32, device=self.device)
            if labels is not None
            else None
        )
        # Rebalance the positive class: attacks are a ~15% minority, so an
        # unweighted BCE happily predicts "clean" everywhere.
        bce = None
        if yt is not None:
            pos = float(yt.sum())
            pos_weight = torch.tensor(
                [(len(yt) - pos) / pos if pos > 0 else 1.0], device=self.device
            )
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        batch = min(self.cfg.tf_batch_size, n)
        self.model_.train()
        self.history_ = []
        for _ in range(self.cfg.tf_epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch):
                idx = torch.as_tensor(order[start : start + batch], device=self.device)
                xb = xt[idx]
                opt.zero_grad()
                recon, logit = self.model_(xb)
                loss = mse(recon, xb)
                if bce is not None:
                    loss = loss + self.cfg.tf_cls_weight * bce(logit, yt[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            self.history_.append(total / n)

        if not self.supervised_:
            train_err = self._forward_scores(X)
            self.threshold_ = float(
                np.quantile(train_err, 1.0 - self.cfg.tf_contamination)
            )
        else:
            self.threshold_ = 0.5  # classifier probability
        return self

    def _forward_scores(self, X: np.ndarray) -> np.ndarray:
        """Classifier probability if supervised, else reconstruction MSE."""
        assert self.model_ is not None
        self.model_.eval()
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon, logit = self.model_(xt)
            if self.supervised_:
                out = torch.sigmoid(logit)
            else:
                out = ((recon - xt) ** 2).mean(dim=(1, 2))
        return out.cpu().numpy().astype(float)

    def score(self, data) -> np.ndarray:
        """Anomaly score per window, larger == more anomalous."""
        if self.model_ is None:
            raise RuntimeError("TransformerDetector must be fit before score")
        return self._forward_scores(self._to_tensor(data, fit_stats=False))

    def predict(self, data) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("TransformerDetector must be fit before predict")
        return (self.score(data) > self.threshold_).astype(int)


class ReconTransformerDetector(TransformerDetector):
    """The Transformer's **unsupervised** arm — reconstruction only.

    Identical encoder to :class:`TransformerDetector`, but it never touches the
    classification head or the labels: the anomaly score is always per-window
    reconstruction error. This is the matched control the ablation needs. The
    supervised Transformer's decade-of-subtlety gain is confounded with its
    access to labels; running the *same* architecture with no labels isolates
    the "learned sequence representation" contribution from the "supervision"
    contribution.

    It declares ``supports_supervision = False`` so the held-out harness fits it
    with a plain ``fit()`` (reconstruction) even while it hands the supervised
    detectors their labels — the whole point of the arm is to see how far the
    encoder gets without them.
    """

    #: Stay unsupervised even under a held-out protocol that offers labels.
    supports_supervision = False

    def fit(self, data, y=None, supervised: bool = False) -> "ReconTransformerDetector":
        if supervised:
            raise ValueError(
                "ReconTransformerDetector is the unsupervised arm; use "
                "TransformerDetector for the supervised head"
            )
        super().fit(data, supervised=False)
        return self
