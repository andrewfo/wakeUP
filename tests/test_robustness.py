"""Robustness-sweep tests (Phase 5).

The sweep is what turns a saturated headline table into a discriminative
benchmark, so it has to be trustworthy in the same way the injectors are: the
curve must isolate severity (same windows corrupted at every point), the
metrics must respond monotonically to severity where physics says they should,
and repeated runs must agree exactly.
"""

import numpy as np
import pytest

from wakeUp.attacks import AttackType
from wakeUp.config import AttackConfig, ModelConfig
from wakeUp.eval.robustness import (
    DEFAULT_SWEEPS,
    SWEEP_PARAM,
    default_detectors,
    run_robustness_sweeps,
    sweep_attack_severity,
)

# Short ladders keep the suite fast; the full ladders live in DEFAULT_SWEEPS.
# Spans chance-level to saturated, so the degradation test is not vacuous.
JUMP_LADDER = [0.001, 0.05, 8.0]


@pytest.fixture(scope="module")
def rule_only():
    return {"KinematicRule": default_detectors()["KinematicRule"]}


def test_sweep_shape_and_columns(windows, rule_only, attack_cfg):
    df = sweep_attack_severity(
        windows, AttackType.POSITION_JUMP, values=JUMP_LADDER,
        attack_cfg=attack_cfg, detectors=rule_only,
    )
    assert len(df) == len(JUMP_LADDER) * len(rule_only)
    for col in ("attack_type", "param", "value", "detector", "pr_auc", "roc_auc", "n", "n_pos"):
        assert col in df.columns
    assert set(df["attack_type"]) == {"position_jump"}
    assert set(df["param"]) == {"jump_km"}
    assert list(df["value"]) == JUMP_LADDER


def test_severity_isolated_from_sampling(windows, rule_only, attack_cfg):
    """Every sweep point corrupts the same windows — only severity varies."""
    df = sweep_attack_severity(
        windows, AttackType.POSITION_JUMP, values=JUMP_LADDER,
        attack_cfg=attack_cfg, detectors=rule_only,
    )
    assert df["n"].nunique() == 1
    assert df["n_pos"].nunique() == 1
    assert df["n_pos"].iloc[0] > 0


def test_detection_degrades_with_subtlety(windows, rule_only, attack_cfg):
    """A gross teleport is trivially caught; a 1 m one is near chance."""
    df = sweep_attack_severity(
        windows, AttackType.POSITION_JUMP, values=JUMP_LADDER,
        attack_cfg=attack_cfg, detectors=rule_only,
    ).sort_values("value")
    pr = df["pr_auc"].to_numpy()
    assert pr[-1] > 0.9                  # 8 km jump: saturated
    assert pr[0] < 0.5                   # 1 m jump: near the base rate
    assert np.all(np.diff(pr) >= -0.05)  # monotone up to small-sample wobble


def test_determinism(windows, rule_only, attack_cfg):
    kw = dict(values=JUMP_LADDER, attack_cfg=attack_cfg, detectors=rule_only)
    a = sweep_attack_severity(windows, AttackType.POSITION_JUMP, **kw)
    b = sweep_attack_severity(windows, AttackType.POSITION_JUMP, **kw)
    assert np.array_equal(a["pr_auc"].to_numpy(), b["pr_auc"].to_numpy())
    assert np.array_equal(a["roc_auc"].to_numpy(), b["roc_auc"].to_numpy())


def test_iforest_sweep_runs(windows, attack_cfg):
    """The learned baseline goes through the same harness uniformly."""
    dets = default_detectors(ModelConfig(seed=0, iforest_estimators=50))
    df = sweep_attack_severity(
        windows, AttackType.GRADUAL_DRIFT, values=[0.5, 5.0],
        attack_cfg=attack_cfg, detectors=dets,
    )
    assert set(df["detector"]) == {"KinematicRule", "IsolationForest"}
    assert df["pr_auc"].notna().all()


def test_unsweepable_attack_raises(windows):
    with pytest.raises(ValueError):
        sweep_attack_severity(windows, AttackType.REPLAY, values=[1.0])


def test_run_all_sweeps_covers_every_knob(windows, rule_only, attack_cfg):
    sweeps = {a: DEFAULT_SWEEPS[p][:2] for a, p in SWEEP_PARAM.items()}
    df = run_robustness_sweeps(
        windows, attack_cfg=attack_cfg, detectors=rule_only, sweeps=sweeps
    )
    assert set(df["param"]) == set(SWEEP_PARAM.values())
    assert len(df) == sum(len(v) for v in sweeps.values()) * len(rule_only)


def test_plot_robustness_curves(windows, rule_only, attack_cfg, tmp_path):
    from wakeUp.eval import plot_robustness_curves

    df = sweep_attack_severity(
        windows, AttackType.POSITION_JUMP, values=JUMP_LADDER,
        attack_cfg=attack_cfg, detectors=rule_only,
    )
    out = plot_robustness_curves(df, tmp_path / "robustness.png")
    assert out.exists() and out.stat().st_size > 0
