"""Transformer detector tests (Phase 4, main model).

Two modes to pin down: unsupervised (reconstruction only, drop-in comparable
with the other detectors) and supervised (classification head, must be scored
on held-out vessels). The tests assert the contract and the mode switch, not
any particular accuracy.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the [learned] extra")

from sklearn.metrics import roc_auc_score  # noqa: E402

from wakeUp.attacks import build_attacked_dataset  # noqa: E402
from wakeUp.config import AttackConfig, ModelConfig  # noqa: E402
from wakeUp.eval.splits import train_test_split_by_vessel, window_labels  # noqa: E402
from wakeUp.features import SequenceTensorizer  # noqa: E402
from wakeUp.models import TransformerDetector  # noqa: E402

# Tiny + few epochs: these assert the contract, not convergence.
FAST = ModelConfig(
    seed=0, tf_d_model=32, tf_nhead=2, tf_layers=1, tf_ff=64,
    tf_epochs=8, tf_batch_size=32, tf_dropout=0.0,
)


@pytest.fixture(scope="module")
def attacked(windows):
    return build_attacked_dataset(windows, AttackConfig())


def test_unsupervised_score_contract(attacked):
    det = TransformerDetector(FAST).fit(attacked)
    scores = det.score(attacked)
    assert det.supervised_ is False
    assert scores.shape == window_labels(attacked).shape
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0  # reconstruction MSE


def test_supervised_scores_are_probabilities(attacked):
    det = TransformerDetector(FAST).fit(attacked, supervised=True)
    scores = det.score(attacked)
    assert det.supervised_ is True
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_supervised_head_learns_and_generalises(attacked):
    """The classification head must actually fit, and transfer above chance.

    The session fixture is 6 vessels — ~40 training windows with ~6 held-out
    positives — which is enough to verify the head learns and that the split
    plumbing works, but far too small to measure detection quality: training
    loss reaches ~0.006 while held-out ROC plateaus at ~0.67 for any epoch
    budget. The real number comes from the full benchmark (see PLANNING.md),
    where the same code reaches 1.00 PR-AUC on held-out vessels.
    """
    train, test = train_test_split_by_vessel(attacked, test_frac=0.4, seed=0)
    cfg = ModelConfig(**{**FAST.__dict__, "tf_epochs": 80})
    det = TransformerDetector(cfg).fit(train, supervised=True)
    y_train, y_test = window_labels(train), window_labels(test)

    assert 0 < y_test.sum() < len(y_test)     # split kept both classes
    assert len(det.score(test)) == len(y_test)
    # The head fits its training data — this is what rules out a broken loss
    # or a mis-aligned label vector.
    assert roc_auc_score(y_train, det.score(train)) > 0.95
    # Generalisation: above chance, but not asserted beyond what 6 positives
    # can support.
    assert roc_auc_score(y_test, det.score(test)) > 0.6


def test_supervised_without_labels_raises():
    X = np.zeros((4, 8, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        TransformerDetector(FAST).fit(X, supervised=True)


def test_determinism(attacked):
    a = TransformerDetector(FAST).fit(attacked).score(attacked)
    b = TransformerDetector(FAST).fit(attacked).score(attacked)
    assert np.array_equal(a, b)


def test_dataframe_and_tensor_paths_agree(attacked):
    X, _, _ = SequenceTensorizer().fit_transform(attacked)
    from_df = TransformerDetector(FAST).fit(attacked).score(attacked)
    from_arr = TransformerDetector(FAST).fit(X).score(X)
    assert np.allclose(from_df, from_arr, atol=1e-6)


def test_training_loss_decreases(attacked):
    det = TransformerDetector(FAST).fit(attacked)
    assert len(det.history_) == FAST.tf_epochs
    assert det.history_[-1] < det.history_[0]


def test_score_before_fit_raises(attacked):
    with pytest.raises(RuntimeError):
        TransformerDetector(FAST).score(attacked)


def test_works_in_robustness_harness(windows):
    """Unsupervised by default, so the sweep harness drives it unchanged."""
    from wakeUp.attacks import AttackType
    from wakeUp.eval.robustness import sweep_attack_severity

    df = sweep_attack_severity(
        windows, AttackType.POSITION_JUMP, values=[8.0],
        attack_cfg=AttackConfig(),
        detectors={"Transformer": lambda: TransformerDetector(FAST)},
    )
    assert len(df) == 1
    assert df["pr_auc"].notna().all()
