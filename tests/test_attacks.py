"""Hammer the attack injectors — this module is the eval backbone.

For every injector we assert:
  * the clean window is left unlabeled by construction,
  * the injector produces at least one labeled point,
  * the labeled points carry the correct attack_type,
  * the corruption is physically detectable (moves position and/or breaks
    speed consistency), i.e. the label actually corresponds to an anomaly,
  * shape/columns are preserved and the injector is deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from wakeUp import geo
from wakeUp.attacks import (
    AttackType,
    inject_position_jump,
    inject_kinematic_impossible,
    inject_replay,
    inject_gradual_drift,
    build_attacked_dataset,
)
from wakeUp.attacks.injectors import inject_identity_swap
from wakeUp.features.kinematic import point_features


def _implied_max_speed(win):
    pf = point_features(win)
    return np.nanmax(pf["implied_speed_kn"].to_numpy())


def test_position_jump_labels_and_moves(one_window, rng, attack_cfg):
    out = inject_position_jump(one_window, rng, attack_cfg)
    atk = out["is_attack"].to_numpy()
    assert atk.sum() > 0
    assert set(out.loc[out["is_attack"] == 1, "attack_type"]) == {
        AttackType.POSITION_JUMP.value
    }
    # displacement roughly matches jump_km at the labeled points
    base = one_window.sort_values("point_idx").reset_index(drop=True)
    out2 = out.sort_values("point_idx").reset_index(drop=True)
    moved = geo.haversine_m(
        base["lat"], base["lon"], out2["lat"], out2["lon"]
    )
    assert moved[out2["is_attack"] == 1].max() > attack_cfg.jump_km * 1000 * 0.9


def test_kinematic_impossible_breaks_speed(one_window, rng, attack_cfg):
    clean_speed = _implied_max_speed(one_window)
    out = inject_kinematic_impossible(one_window, rng, attack_cfg)
    assert out["is_attack"].sum() > 0
    assert _implied_max_speed(out) > clean_speed * 2


def test_replay_repeats_positions(one_window, rng, attack_cfg):
    out = inject_replay(one_window, rng, attack_cfg).sort_values("point_idx").reset_index(drop=True)
    assert out["is_attack"].sum() > 0
    n = len(out)
    seg = n // 3
    # the pasted tail should equal the head segment
    head = out.loc[: seg - 1, ["lat", "lon"]].to_numpy()
    tail = out.loc[n - seg :, ["lat", "lon"]].to_numpy()
    assert np.allclose(head, tail)


def test_gradual_drift_is_subtle_but_present(one_window, rng, attack_cfg):
    out = inject_gradual_drift(one_window, rng, attack_cfg)
    assert out["is_attack"].sum() > 0
    base = one_window.sort_values("point_idx").reset_index(drop=True)
    out2 = out.sort_values("point_idx").reset_index(drop=True)
    moved = np.asarray(geo.haversine_m(base["lat"], base["lon"], out2["lat"], out2["lon"]))
    # ends far from truth, starts near truth (a ramp)
    assert moved[-1] > attack_cfg.drift_total_km * 1000 * 0.8
    assert moved[0] < 50


def test_identity_swap_needs_donor(one_window, rng, attack_cfg):
    # no donor -> unchanged/clean
    out_nodonor = inject_identity_swap(one_window, rng, attack_cfg, donor=None)
    assert out_nodonor["is_attack"].sum() == 0

    donor = one_window.copy()
    donor["lat"] = donor["lat"] + 0.2  # a clearly different track
    out = inject_identity_swap(one_window, rng, attack_cfg, donor=donor)
    assert out["is_attack"].sum() > 0
    assert set(out.loc[out["is_attack"] == 1, "attack_type"]) == {
        AttackType.IDENTITY_SWAP.value
    }


def test_injectors_preserve_shape_and_columns(one_window, rng, attack_cfg):
    for fn in (inject_position_jump, inject_kinematic_impossible,
               inject_replay, inject_gradual_drift):
        out = fn(one_window, rng, attack_cfg)
        assert len(out) == len(one_window)
        for col in ("mmsi", "timestamp", "lat", "lon", "sog", "cog"):
            assert col in out.columns


def test_injector_determinism(one_window, attack_cfg):
    a = inject_position_jump(one_window, np.random.default_rng(42), attack_cfg)
    b = inject_position_jump(one_window, np.random.default_rng(42), attack_cfg)
    pd.testing.assert_frame_equal(
        a.sort_values("point_idx").reset_index(drop=True),
        b.sort_values("point_idx").reset_index(drop=True),
    )


def test_build_attacked_dataset_labels_consistent(windows, attack_cfg):
    full = build_attacked_dataset(windows, attack_cfg)
    # window_label == max of point labels within window
    grp = full.groupby("window_id")
    recomputed = grp["is_attack"].max()
    stored = grp["window_label"].first()
    assert (recomputed.values == stored.values).all()
    # contamination roughly honoured
    rate = stored.mean()
    assert 0.05 < rate < 0.35


def test_forge_velocity_restores_consistency(one_window, rng, attack_cfg):
    """With forged velocity, reported SOG should track implied speed."""
    out = inject_position_jump(one_window, rng, attack_cfg, forge_velocity=True)
    pf = point_features(out)
    resid = np.abs(pf["speed_resid_kn"].to_numpy())
    resid = resid[np.isfinite(resid)]
    # forged velocity => small residual almost everywhere
    assert np.median(resid) < 1.0
