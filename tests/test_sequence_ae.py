"""LSTM-autoencoder tests (Phase 4).

Skipped entirely without the ``learned`` extra. What matters here is the
contract the eval harness relies on — score orientation (larger == more
anomalous), determinism, and that train-only normalisation statistics are
reused at score time — rather than any particular accuracy number.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the [learned] extra")

from sklearn.metrics import roc_auc_score  # noqa: E402

from wakeUp.attacks import build_attacked_dataset  # noqa: E402
from wakeUp.config import AttackConfig, ModelConfig  # noqa: E402
from wakeUp.features import SequenceTensorizer  # noqa: E402
from wakeUp.models import LSTMAutoencoderDetector  # noqa: E402

# Small + few epochs: these tests assert the contract, not convergence.
FAST = ModelConfig(seed=0, lstm_hidden=16, lstm_epochs=6, lstm_batch_size=32)


@pytest.fixture(scope="module")
def attacked(windows):
    # Built here rather than from the function-scoped `attack_cfg` fixture so
    # the (slow) training runs share one module-scoped dataset.
    return build_attacked_dataset(windows, AttackConfig())


def test_score_shape_and_orientation(attacked):
    """Larger score == more anomalous, so attacked windows must rank higher."""
    det = LSTMAutoencoderDetector(FAST).fit(attacked)
    scores = det.score(attacked)
    labels = attacked.groupby("window_id")["window_label"].first().to_numpy()

    assert scores.shape == labels.shape
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0  # squared error
    # Orientation, not accuracy: a wrong sign would land well below 0.5.
    assert roc_auc_score(labels, scores) > 0.6


def test_determinism(attacked):
    a = LSTMAutoencoderDetector(FAST).fit(attacked).score(attacked)
    b = LSTMAutoencoderDetector(FAST).fit(attacked).score(attacked)
    assert np.array_equal(a, b)


def test_dataframe_and_tensor_paths_agree(attacked):
    """Passing prebuilt tensors must match the internal tensorizer path."""
    X, _, _ = SequenceTensorizer().fit_transform(attacked)
    from_df = LSTMAutoencoderDetector(FAST).fit(attacked).score(attacked)
    from_arr = LSTMAutoencoderDetector(FAST).fit(X).score(X)
    assert np.allclose(from_df, from_arr, atol=1e-6)


def test_train_only_stats_reused_on_heldout(attacked):
    """Scoring a held-out frame reuses the fitted normalisation, not new stats."""
    wids = sorted(attacked["window_id"].unique())
    half = len(wids) // 2
    train = attacked[attacked["window_id"].isin(wids[:half])]
    test = attacked[attacked["window_id"].isin(wids[half:])]

    det = LSTMAutoencoderDetector(FAST).fit(train)
    mean_before = det.tensorizer_.mean_.copy()
    scores = det.score(test)

    assert np.array_equal(det.tensorizer_.mean_, mean_before)
    assert len(scores) == len(wids) - half
    assert np.isfinite(scores).all()


def test_predict_respects_contamination(attacked):
    det = LSTMAutoencoderDetector(FAST).fit(attacked)
    flags = det.predict(attacked)
    assert set(np.unique(flags)) <= {0, 1}
    # Threshold is the training-error quantile, so the flagged fraction sits
    # near the configured contamination.
    assert abs(flags.mean() - FAST.lstm_contamination) < 0.05


def test_training_loss_decreases(attacked):
    det = LSTMAutoencoderDetector(FAST).fit(attacked)
    assert len(det.history_) == FAST.lstm_epochs
    assert det.history_[-1] < det.history_[0]


def test_score_before_fit_raises(attacked):
    with pytest.raises(RuntimeError):
        LSTMAutoencoderDetector(FAST).score(attacked)


def test_bad_tensor_shape_raises():
    with pytest.raises(ValueError):
        LSTMAutoencoderDetector(FAST).fit(np.zeros((10, 7)))
