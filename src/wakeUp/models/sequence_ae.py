"""LSTM autoencoder over track windows (Phase 4).

The classical baselines summarise a window into 27 hand-built aggregates and
look for outliers in that space. The autoencoder instead consumes the raw
per-point channel tensor ``(window, point, channel)`` from
:class:`~wakeUp.features.sequences.SequenceTensorizer`, learns to reconstruct
*ordinary* vessel motion, and reports per-window reconstruction error as the
anomaly score — larger error == motion the model has not learned to explain ==
more anomalous, matching the ``fit`` / ``score`` convention of every other
detector so the eval and robustness harnesses treat it uniformly.

Training is **unsupervised on the contaminated set**, exactly like
:class:`~wakeUp.models.baselines.IsolationForestDetector`: labels are never
shown to the model, and the attacked minority is simply what the autoencoder
fails to fit. That keeps the comparison honest and the protocol identical
across detectors.

Input flexibility: ``fit`` / ``score`` accept either the per-point *windows
DataFrame* (in which case an internal ``SequenceTensorizer`` is fit on the
training frame and reused unchanged at score time, so no normalisation
statistics leak) or an already-built ``(N, L, C)`` float array.

Determinism: the model runs on CPU by default and batches are drawn from an
explicit ``numpy`` generator rather than a DataLoader, so repeated runs with
the same seed produce bitwise-identical scores. cuDNN's LSTM kernels are not
guaranteed deterministic, so ``device="cuda"`` trades reproducibility for
speed — the benchmark is small enough that CPU is the better default.

Requires the ``learned`` extra: ``pip install -e ".[learned]"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # torch is an optional extra; fail with an actionable message
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "LSTMAutoencoderDetector requires PyTorch. Install the learned extra:\n"
        '    pip install -e ".[learned]"'
    ) from exc

from wakeUp.config import ModelConfig
from wakeUp.features.sequences import SequenceTensorizer


class _LSTMAutoencoder(nn.Module):
    """Encode a window to a fixed latent, decode it back point by point.

    The encoder's final hidden state is the whole window's summary; it is
    repeated across the time axis as the decoder input, so the decoder must
    regenerate the trajectory from that summary alone. A window whose motion
    is unlike the training distribution survives that bottleneck poorly, which
    is what makes the reconstruction error a usable anomaly score.
    """

    def __init__(self, n_channels: int, hidden: int, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        _, (h, _) = self.encoder(x)
        latent = h[-1]                                   # (B, H)
        seq = latent.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, L, H)
        dec, _ = self.decoder(seq)
        return self.head(dec)                            # (B, L, C)


class LSTMAutoencoderDetector:
    """Reconstruction-error anomaly detector over track-window sequences."""

    #: Tells the eval/robustness harnesses to hand this detector the per-point
    #: windows frame rather than the aggregated window feature matrix.
    consumes_windows = True

    def __init__(self, cfg: ModelConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or ModelConfig()
        self.device = device
        self.tensorizer_: SequenceTensorizer | None = None
        self.model_: _LSTMAutoencoder | None = None
        self.threshold_: float | None = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def _to_tensor(self, data, fit_stats: bool) -> np.ndarray:
        """Accept a windows DataFrame or a prebuilt ``(N, L, C)`` array."""
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

    # ------------------------------------------------------------------ #
    # fit / score
    # ------------------------------------------------------------------ #
    def fit(self, data, y=None) -> "LSTMAutoencoderDetector":
        """Train the autoencoder. ``y`` is ignored — training is unsupervised."""
        X = self._to_tensor(data, fit_stats=True)
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        n, _, n_channels = X.shape
        self.model_ = _LSTMAutoencoder(n_channels, self.cfg.lstm_hidden).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.cfg.lstm_lr)
        loss_fn = nn.MSELoss()
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        batch = min(self.cfg.lstm_batch_size, n)
        self.model_.train()
        self.history_ = []
        for _ in range(self.cfg.lstm_epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch):
                idx = torch.as_tensor(order[start : start + batch], device=self.device)
                xb = xt[idx]
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), xb)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            self.history_.append(total / n)

        # Operating point for `predict`: the top `contamination` fraction of
        # training reconstruction errors.
        train_err = self._reconstruction_error(X)
        self.threshold_ = float(
            np.quantile(train_err, 1.0 - self.cfg.lstm_contamination)
        )
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Mean squared error per window, shape ``(n_windows,)``."""
        assert self.model_ is not None
        self.model_.eval()
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self.model_(xt)
            err = ((recon - xt) ** 2).mean(dim=(1, 2))
        return err.cpu().numpy().astype(float)

    def score(self, data) -> np.ndarray:
        """Anomaly score per window, larger == more anomalous."""
        if self.model_ is None:
            raise RuntimeError("LSTMAutoencoderDetector must be fit before score")
        return self._reconstruction_error(self._to_tensor(data, fit_stats=False))

    def predict(self, data) -> np.ndarray:
        """Hard 0/1 label at the fitted contamination operating point."""
        if self.threshold_ is None:
            raise RuntimeError("LSTMAutoencoderDetector must be fit before predict")
        return (self.score(data) > self.threshold_).astype(int)
