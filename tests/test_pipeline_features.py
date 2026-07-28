"""Pipeline (clean/resample/window) and feature-extraction tests."""

import numpy as np
import pandas as pd

from wakeUp.config import DataConfig
from wakeUp.data.pipeline import clean_ais, resample_track, segment_windows
from wakeUp.features import build_feature_matrix, FEATURE_NAMES
from wakeUp.features.kinematic import point_features


def test_clean_drops_dupes_and_short(small_cfg):
    from wakeUp.data import generate_fleet

    raw = generate_fleet(small_cfg)
    n_raw_dupes = raw.duplicated(subset=["mmsi", "timestamp"]).sum()
    assert n_raw_dupes > 0  # generator injected dupes
    cleaned = clean_ais(raw, small_cfg)
    assert cleaned.duplicated(subset=["mmsi", "timestamp"]).sum() == 0
    counts = cleaned.groupby("mmsi").size()
    assert (counts >= small_cfg.min_track_points).all()


def test_resample_regular_cadence(small_cfg):
    from wakeUp.data import generate_fleet

    raw = generate_fleet(small_cfg)
    cleaned = clean_ais(raw, small_cfg)
    one = cleaned[cleaned["mmsi"] == cleaned["mmsi"].iloc[0]]
    rs = resample_track(one, small_cfg.cadence_s)
    diffs = rs["timestamp"].diff().dropna().dt.total_seconds().unique()
    assert set(diffs) == {float(small_cfg.cadence_s)}
    assert "gap_s" in rs.columns


def test_windows_have_fixed_length(windows, small_cfg):
    sizes = windows.groupby("window_id").size().unique()
    assert set(sizes) == {small_cfg.window_len}
    # point_idx runs 0..L-1
    w0 = windows[windows["window_id"] == windows["window_id"].iloc[0]]
    assert list(w0.sort_values("point_idx")["point_idx"]) == list(range(small_cfg.window_len))


def test_feature_matrix_shape_and_names(windows):
    from wakeUp.attacks import build_attacked_dataset
    from wakeUp.config import AttackConfig

    attacked = build_attacked_dataset(windows, AttackConfig())
    feat, labels = build_feature_matrix(attacked)
    assert list(feat.columns) == FEATURE_NAMES
    assert len(feat) == attacked["window_id"].nunique()
    assert len(labels) == len(feat)
    assert np.isfinite(feat.to_numpy()).all()  # no NaN/inf leak into features


def test_clean_track_has_low_speed_residual(one_window):
    """Sanity: a clean generated window is self-consistent."""
    pf = point_features(one_window)
    resid = np.abs(pf["speed_resid_kn"].to_numpy())
    resid = resid[np.isfinite(resid)]
    assert np.median(resid) < 1.0
