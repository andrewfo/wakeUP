"""Ablation tests (Phase 5).

The ablation exists to *decompose* the supervised Transformer's gain into a
learned-representation part and a supervision part, so what has to be trustworthy
is the plumbing: the two new cells honour the detector contract, each cell is fit
under the right (supervised / unsupervised, windows / features) branch, and the
resulting tables carry the representation × supervision axes so the grid reads
back correctly. Accuracy is not asserted here — the 6-vessel fixture is far too
small — only that every cell runs uniformly through the held-out harness.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the [learned] extra")

from wakeUp.attacks import AttackType, build_attacked_dataset  # noqa: E402
from wakeUp.config import AttackConfig, ModelConfig  # noqa: E402
from wakeUp.eval.ablation import (  # noqa: E402
    ablation_cells,
    run_ablation_sweeps,
    run_ablation_table,
)
from wakeUp.eval.splits import window_labels  # noqa: E402
from wakeUp.models import (  # noqa: E402
    HybridDetector,
    LogisticFeatureDetector,
    ReconTransformerDetector,
    TransformerDetector,
)

# Tiny + few epochs: these assert the contract and the plumbing, not convergence.
FAST = ModelConfig(
    seed=0, iforest_estimators=50, tf_d_model=32, tf_nhead=2, tf_layers=1,
    tf_ff=64, tf_epochs=8, tf_batch_size=32, tf_dropout=0.0,
)


@pytest.fixture(scope="module")
def attacked(windows):
    return build_attacked_dataset(windows, AttackConfig())


# --------------------------------------------------------------------------- #
# LogisticFeatureDetector — the supervised hand-features cell
# --------------------------------------------------------------------------- #
def test_logistic_contract_flags():
    assert LogisticFeatureDetector.consumes_windows is True
    assert LogisticFeatureDetector.supports_supervision is True


def test_logistic_scores_are_probabilities(attacked):
    det = LogisticFeatureDetector(FAST).fit(attacked, supervised=True)
    scores = det.score(attacked)
    assert scores.shape == window_labels(attacked).shape
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_logistic_learns_training_data(attacked):
    from sklearn.metrics import roc_auc_score

    det = LogisticFeatureDetector(FAST).fit(attacked, supervised=True)
    # A linear model on the hand features must at least fit its own training
    # windows — that is what rules out a mis-aligned label vector.
    assert roc_auc_score(window_labels(attacked), det.score(attacked)) > 0.9


def test_logistic_determinism(attacked):
    a = LogisticFeatureDetector(FAST).fit(attacked, supervised=True).score(attacked)
    b = LogisticFeatureDetector(FAST).fit(attacked, supervised=True).score(attacked)
    assert np.array_equal(a, b)


def test_logistic_score_before_fit_raises(attacked):
    with pytest.raises(RuntimeError):
        LogisticFeatureDetector(FAST).score(attacked)


# --------------------------------------------------------------------------- #
# ReconTransformerDetector — the unsupervised learned cell (matched control)
# --------------------------------------------------------------------------- #
def test_recon_transformer_stays_unsupervised(attacked):
    det = ReconTransformerDetector(FAST).fit(attacked)
    assert det.supports_supervision is False
    assert det.supervised_ is False
    scores = det.score(attacked)
    assert scores.min() >= 0.0  # reconstruction MSE


def test_recon_transformer_refuses_supervision(attacked):
    with pytest.raises(ValueError):
        ReconTransformerDetector(FAST).fit(attacked, supervised=True)


def test_recon_matches_transformer_recon_head(attacked):
    """Same encoder, same seed -> identical reconstruction scores as the
    unsupervised base detector; the subclass only fixes the mode."""
    a = ReconTransformerDetector(FAST).fit(attacked).score(attacked)
    b = TransformerDetector(FAST).fit(attacked).score(attacked)
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# HybridDetector — supervised linear head on features ⊕ pooled embedding
# --------------------------------------------------------------------------- #
def test_hybrid_contract_flags():
    assert HybridDetector.consumes_windows is True
    assert HybridDetector.supports_supervision is True


def test_hybrid_scores_are_probabilities(attacked):
    det = HybridDetector(FAST).fit(attacked, supervised=True)
    scores = det.score(attacked)
    assert scores.shape == window_labels(attacked).shape
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_hybrid_learns_training_data(attacked):
    from sklearn.metrics import roc_auc_score

    det = HybridDetector(FAST).fit(attacked, supervised=True)
    # As with Logistic: fitting its own training windows rules out a
    # mis-aligned feature/embedding/label ordering.
    assert roc_auc_score(window_labels(attacked), det.score(attacked)) > 0.9


def test_hybrid_refuses_unsupervised(attacked):
    with pytest.raises(ValueError):
        HybridDetector(FAST).fit(attacked, supervised=False)


def test_hybrid_determinism(attacked):
    a = HybridDetector(FAST).fit(attacked, supervised=True).score(attacked)
    b = HybridDetector(FAST).fit(attacked, supervised=True).score(attacked)
    assert np.allclose(a, b)


def test_hybrid_embedding_dimensions(attacked):
    det = HybridDetector(FAST).fit(attacked, supervised=True)
    emb = det.embed(attacked)
    n_windows = attacked["window_id"].nunique()
    assert emb.shape == (n_windows, FAST.tf_d_model)
    # The head sees 27 hand features + the pooled embedding, nothing else.
    assert det.head_.coef_.shape[1] == len(det.columns_) + FAST.tf_d_model


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def test_ablation_cells_axes():
    cells = ablation_cells(FAST)
    assert set(cells) == {"IForest", "Logistic", "ReconTransformer", "Transformer"}
    axes = {name: (rep, sup) for name, (rep, sup, _) in cells.items()}
    assert axes["IForest"] == ("features", "unsupervised")
    assert axes["Logistic"] == ("features", "supervised")
    assert axes["ReconTransformer"] == ("learned", "unsupervised")
    assert axes["Transformer"] == ("learned", "supervised")


def test_ablation_cells_lstm_optin():
    cells = ablation_cells(FAST, include_lstm=True)
    assert "LSTM-AE" in cells
    rep, sup, _ = cells["LSTM-AE"]
    assert (rep, sup) == ("learned", "unsupervised")


def test_ablation_cells_hybrid_optin():
    cells = ablation_cells(FAST, include_hybrid=True)
    assert "Hybrid" in cells
    rep, sup, make = cells["Hybrid"]
    assert (rep, sup) == ("features+learned", "supervised")
    assert isinstance(make(), HybridDetector)


def test_ablation_table_structure(windows):
    table = run_ablation_table(
        windows, attack_cfg=AttackConfig(), model_cfg=FAST, test_frac=0.4
    )
    for col in ("detector", "representation", "supervision", "attack_type", "pr_auc"):
        assert col in table.columns
    # Every cell appears, each with an ALL row plus at least one attack family.
    assert set(table["detector"]) == set(ablation_cells(FAST))
    overall = table[table["attack_type"] == "ALL"]
    assert len(overall) == len(ablation_cells(FAST))
    assert overall["pr_auc"].notna().all()
    # Axes are consistent per detector.
    assert (
        table.groupby("detector")[["representation", "supervision"]].nunique().max().max()
        == 1
    )


def test_ablation_table_is_held_out(windows):
    """The table scores fewer windows than the full set — held-out vessels only."""
    table = run_ablation_table(
        windows, attack_cfg=AttackConfig(), model_cfg=FAST, test_frac=0.4
    )
    n_scored = table[table["attack_type"] == "ALL"]["n"].iloc[0]
    assert n_scored < windows["window_id"].nunique()


def test_ablation_sweeps_carry_axes(windows):
    sweep = run_ablation_sweeps(
        windows,
        attack_cfg=AttackConfig(seed=7, jump_km=8.0, drift_total_km=3.0,
                                speed_multiplier=6.0),
        model_cfg=FAST,
        cells={k: v for k, v in ablation_cells(FAST).items()
               if k in ("IForest", "Logistic")},  # fast subset
        test_frac=0.4,
    )
    assert {"representation", "supervision"}.issubset(sweep.columns)
    assert set(sweep["detector"]) == {"IForest", "Logistic"}
    # supervised Logistic gets the "supervised" tag everywhere it appears.
    assert set(sweep[sweep["detector"] == "Logistic"]["supervision"]) == {"supervised"}
    assert sweep["pr_auc"].notna().all()
