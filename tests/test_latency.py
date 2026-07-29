"""Detection-latency tests (Phase 5).

What has to be trustworthy is the accounting, not any particular latency
number: alarms must be counted only once the prefix actually contains attacked
points, misses must stay in the table, the per-length threshold must respect
the clean-window FPR budget, and the whole thing must be deterministic. The
gross-severity fixture attacks are detectable enough that the rule and forest
should catch *something*, which is asserted loosely.
"""

import numpy as np
import pandas as pd
import pytest

from wakeUp.attacks import build_attacked_dataset
from wakeUp.config import ModelConfig
from wakeUp.eval.latency import (
    clean_fpr_threshold,
    detection_latency,
    run_detection_latency,
    summarize_latency,
)
from wakeUp.eval.splits import window_labels
from wakeUp.features import build_feature_matrix
from wakeUp.models import IsolationForestDetector, KinematicRuleDetector

FAST = ModelConfig(seed=0, iforest_estimators=50)


@pytest.fixture(scope="module")
def attacked(windows):
    from wakeUp.config import AttackConfig

    return build_attacked_dataset(
        windows,
        AttackConfig(seed=7, jump_km=8.0, drift_total_km=3.0, speed_multiplier=6.0),
    )


@pytest.fixture(scope="module")
def rule_latency(attacked):
    feat, _ = build_feature_matrix(attacked)
    det = KinematicRuleDetector().fit(feat)
    return detection_latency(attacked, det, min_points=8, max_fpr=0.05)


# --------------------------------------------------------------------------- #
# threshold calibration
# --------------------------------------------------------------------------- #
def test_threshold_respects_fpr_budget(attacked):
    feat, labels = build_feature_matrix(attacked)
    scores = KinematicRuleDetector().fit(feat).score(feat)
    thr = clean_fpr_threshold(scores, labels, max_fpr=0.05)
    clean = scores[labels == 0]
    assert (clean > thr).mean() <= 0.05 + 1e-9


def test_threshold_needs_clean_windows():
    with pytest.raises(ValueError):
        clean_fpr_threshold(np.ones(4), np.ones(4))


# --------------------------------------------------------------------------- #
# per-window latency accounting
# --------------------------------------------------------------------------- #
def test_latency_rows_are_attacked_windows_only(attacked, rule_latency):
    labels = window_labels(attacked)
    assert len(rule_latency) == int(labels.sum())
    assert set(rule_latency.columns) >= {
        "window_id", "mmsi", "attack_type", "onset_idx",
        "detected", "latency_points", "latency_s",
    }
    assert (rule_latency["attack_type"] != "none").all()


def test_latency_nonnegative_and_alarm_after_onset(rule_latency, attacked):
    """An alarm on a prefix with no attacked points must not count, so every
    reported latency is >= 0 and bounded by the window length."""
    length = int(attacked["point_idx"].max()) + 1
    hits = rule_latency[rule_latency["detected"]]
    assert len(hits) > 0  # gross attacks: the rule must catch something
    assert (hits["latency_points"] >= 0).all()
    assert (hits["latency_points"] < length).all()
    assert (hits["onset_idx"] + hits["latency_points"] < length).all()


def test_misses_are_reported_not_dropped(rule_latency):
    misses = rule_latency[~rule_latency["detected"]]
    assert misses["latency_points"].isna().all()


def test_latency_seconds_match_cadence(rule_latency, small_cfg):
    hits = rule_latency[rule_latency["detected"]]
    assert np.allclose(
        hits["latency_s"], hits["latency_points"] * small_cfg.cadence_s
    )


def test_latency_determinism(attacked):
    feat, _ = build_feature_matrix(attacked)
    det = IsolationForestDetector(FAST).fit(feat)
    a = detection_latency(attacked, det, min_points=8)
    b = detection_latency(attacked, det, min_points=8)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# harness + summary
# --------------------------------------------------------------------------- #
def test_run_detection_latency_holdout(attacked):
    lat = run_detection_latency(
        attacked,
        detectors={
            "KinematicRule": lambda: KinematicRuleDetector(),
            "IForest": lambda: IsolationForestDetector(FAST),
        },
        test_frac=0.4,
    )
    assert set(lat["detector"]) == {"KinematicRule", "IForest"}
    # Held out: fewer attacked windows scored than exist in the full frame.
    per_det = lat.groupby("detector").size()
    assert (per_det < window_labels(attacked).sum()).all()
    # Held-out vessels only.
    assert lat["mmsi"].nunique() < attacked["mmsi"].nunique()


def test_summarize_latency(attacked):
    lat = run_detection_latency(
        attacked,
        detectors={"KinematicRule": lambda: KinematicRuleDetector()},
        test_frac=0.4,
    )
    summary = summarize_latency(lat)
    assert {"detector", "attack_type", "n", "detection_rate",
            "median_latency_points", "median_latency_s"} <= set(summary.columns)
    assert summary["detection_rate"].between(0, 1).all()
    assert (summary["n"] >= 1).all()
