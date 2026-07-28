"""Sequence-tensor construction tests (Phase 3 → learned models).

The learned models depend on these tensors being correctly shaped, ordered,
finite, and normalised with train-only statistics, so they carry their own
coverage the way the attack injectors do.
"""

import numpy as np
import pandas as pd

from wakeUp.config import AttackConfig
from wakeUp.attacks import build_attacked_dataset
from wakeUp.features import (
    SequenceTensorizer,
    build_sequence_tensors,
    SEQUENCE_CHANNELS,
)


def _attacked(windows):
    return build_attacked_dataset(windows, AttackConfig())


def test_tensor_shape_and_alignment(windows, small_cfg):
    attacked = _attacked(windows)
    X, labels, wids = build_sequence_tensors(attacked)
    n_windows = attacked["window_id"].nunique()
    assert X.shape == (n_windows, small_cfg.window_len, len(SEQUENCE_CHANNELS))
    assert len(labels) == n_windows
    assert len(wids) == n_windows
    # window ids are returned in the same sorted order used for the tensor
    assert wids == sorted(attacked["window_id"].unique())


def test_tensor_is_finite(windows):
    X, _, _ = build_sequence_tensors(_attacked(windows))
    assert np.isfinite(X).all()  # NaN first-point deltas imputed, no inf leak
    assert X.dtype == np.float32


def test_per_channel_normalisation(windows):
    X, _, _ = build_sequence_tensors(_attacked(windows))
    # Standardised channels: pooled mean ~0, and every channel on a comparable
    # scale (std ~1). A channel that is constant on clean synthetic tracks
    # (``gap_s`` — regular cadence, no sensor gaps) correctly maps to exactly 0
    # after mean-subtraction with a guarded unit std, rather than blowing up.
    flat = X.reshape(-1, X.shape[-1])
    std = flat.std(axis=0)
    assert np.all(np.abs(flat.mean(axis=0)) < 0.2)
    assert np.all(std <= 1.5)  # no channel dominates the scale
    assert np.all((std > 0.3) | (std == 0.0))  # varying channels ~unit, else degenerate
    assert np.any(std > 0.3)  # at least the kinematic channels carry signal


def test_determinism(windows):
    attacked = _attacked(windows)
    X1, l1, w1 = build_sequence_tensors(attacked)
    X2, l2, w2 = build_sequence_tensors(attacked)
    assert np.array_equal(X1, X2)
    assert np.array_equal(l1, l2)
    assert w1 == w2


def test_train_only_stats_no_leak(windows):
    """transform on held-out windows must reuse the fitted train stats."""
    attacked = _attacked(windows)
    wids = sorted(attacked["window_id"].unique())
    half = len(wids) // 2
    train = attacked[attacked["window_id"].isin(wids[:half])]
    test = attacked[attacked["window_id"].isin(wids[half:])]

    tz = SequenceTensorizer().fit(train)
    mean_before = tz.mean_.copy()
    std_before = tz.std_.copy()
    Xtest, _, _ = tz.transform(test)
    # transforming the test split does not mutate the fitted statistics
    assert np.array_equal(tz.mean_, mean_before)
    assert np.array_equal(tz.std_, std_before)
    assert np.isfinite(Xtest).all()
    assert Xtest.shape[1:] == (
        train.groupby("window_id").size().iloc[0],
        len(SEQUENCE_CHANNELS),
    )


def test_transform_before_fit_raises(windows):
    tz = SequenceTensorizer()
    try:
        tz.transform(_attacked(windows))
    except RuntimeError:
        return
    raise AssertionError("transform before fit should raise RuntimeError")
