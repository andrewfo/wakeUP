"""Gap-aware track segmentation tests (Phase 1, real-ingest prerequisite).

Real AIS goes silent; the synthetic fleet never does. Resampling interpolates
onto a fixed grid, so a silence that is not split on first becomes fabricated
smooth motion that is then windowed and labelled *clean* — a false negative
manufactured by the pipeline itself. These tests pin the split, and pin that
leaving it disabled reproduces the pre-existing behaviour exactly (every
recorded benchmark number depends on that).
"""

import numpy as np
import pandas as pd

from wakeUp.config import DataConfig
from wakeUp.data.pipeline import (
    build_dataset,
    drop_short_segments,
    resample_all,
    segment_windows,
    split_on_gaps,
)


def _track(mmsi: int, start: str, n: int, lat0: float = 36.0) -> pd.DataFrame:
    """A short, regular, self-consistent stretch of one vessel's track."""
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq="60s")
    return pd.DataFrame(
        {
            "mmsi": mmsi,
            "timestamp": ts,
            "lat": lat0 + np.arange(n) * 1e-3,
            "lon": -75.0,
            "sog": 10.0,
            "cog": 0.0,
            "heading": 0.0,
        }
    )


def _gapped_track(mmsi: int = 1, n: int = 40, gap_hours: float = 2.0):
    """Two runs of ``n`` fixes separated by a silence, far apart in space."""
    a = _track(mmsi, "2024-01-01T00:00:00", n, lat0=36.0)
    start_b = a["timestamp"].iloc[-1] + pd.Timedelta(hours=gap_hours)
    b = _track(mmsi, str(start_b), n, lat0=37.8)
    return pd.concat([a, b], ignore_index=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def test_split_marks_two_segments_across_a_silence():
    df = _gapped_track()
    out = split_on_gaps(df, max_gap_s=600.0)
    assert sorted(out["segment"].unique()) == [0, 1]
    assert (out["segment"].to_numpy()[:40] == 0).all()
    assert (out["segment"].to_numpy()[40:] == 1).all()


def test_split_disabled_yields_one_segment():
    out = split_on_gaps(_gapped_track(), max_gap_s=None)
    assert out["segment"].nunique() == 1


def test_gap_below_threshold_does_not_split():
    df = _gapped_track(gap_hours=0.05)  # 180 s silence
    assert split_on_gaps(df, max_gap_s=600.0)["segment"].nunique() == 1


def test_segments_are_per_vessel():
    """Segment ids restart per MMSI; a vessel boundary is not a gap."""
    df = pd.concat([_gapped_track(mmsi=1), _gapped_track(mmsi=2)], ignore_index=True)
    out = split_on_gaps(df, max_gap_s=600.0)
    for _, g in out.groupby("mmsi"):
        assert sorted(g["segment"].unique()) == [0, 1]


# --------------------------------------------------------------------------- #
# The actual point: no fabricated motion, no straddling windows
# --------------------------------------------------------------------------- #
def test_resampling_does_not_interpolate_across_the_gap():
    cfg = DataConfig(cadence_s=60, min_track_points=20, max_gap_s=600.0)
    df = _gapped_track(n=40, gap_hours=2.0)

    unsplit = resample_all(split_on_gaps(df, None), cfg)
    split = resample_all(split_on_gaps(df, cfg.max_gap_s), cfg)

    # Unsplit, the 2 h silence is filled in with invented points.
    assert len(unsplit) > len(df)
    # Split, every resampled point is bracketed by real observations.
    assert len(split) == len(df)

    silence_start = df["timestamp"].iloc[39]
    silence_end = df["timestamp"].iloc[40]

    def n_inside_silence(frame):
        ts = frame["timestamp"]
        return int(((ts > silence_start) & (ts < silence_end)).sum())

    assert n_inside_silence(unsplit) > 0
    assert n_inside_silence(split) == 0


def test_no_window_straddles_a_gap():
    cfg = DataConfig(
        cadence_s=60, min_track_points=20, max_gap_s=600.0, window_len=32,
        window_stride=16,
    )
    df = _gapped_track(n=40, gap_hours=2.0)
    frame = resample_all(split_on_gaps(df, cfg.max_gap_s), cfg)
    windows = segment_windows(frame, cfg)

    assert len(windows) > 0
    for _, w in windows.groupby("window_id"):
        assert w["segment"].nunique() == 1
        steps = w.sort_values("point_idx")["timestamp"].diff().dropna()
        assert (steps.dt.total_seconds() == cfg.cadence_s).all()


def test_short_segments_are_dropped():
    """A stub segment too short to window is discarded, not padded."""
    a = _track(1, "2024-01-01T00:00:00", 40)
    stub = _track(1, "2024-01-01T06:00:00", 3, lat0=37.0)
    df = split_on_gaps(pd.concat([a, stub], ignore_index=True), max_gap_s=600.0)
    assert df["segment"].nunique() == 2

    kept = drop_short_segments(df, min_points=20)
    assert kept["segment"].nunique() == 1
    assert len(kept) == 40


# --------------------------------------------------------------------------- #
# Backwards compatibility — the recorded benchmark numbers depend on this
# --------------------------------------------------------------------------- #
def test_default_config_leaves_the_synthetic_pipeline_unchanged(small_cfg):
    """max_gap_s defaults to None, and the gap-free fleet is one segment."""
    from wakeUp.data import generate_fleet

    assert small_cfg.max_gap_s is None
    windows = build_dataset(generate_fleet(small_cfg), small_cfg)
    assert windows["segment"].nunique() == 1
    # one window_id per (vessel, slice) exactly as before, fixed length
    assert set(windows.groupby("window_id").size().unique()) == {small_cfg.window_len}


def test_enabling_split_is_a_noop_on_gapfree_data(small_cfg):
    """Synthetic tracks have no silences, so splitting must change nothing."""
    from dataclasses import replace

    from wakeUp.data import generate_fleet

    raw = generate_fleet(small_cfg)
    off = build_dataset(raw, small_cfg)
    on = build_dataset(raw, replace(small_cfg, max_gap_s=600.0))

    assert len(on) == len(off)
    assert on["window_id"].nunique() == off["window_id"].nunique()
    for col in ("lat", "lon", "sog", "cog"):
        np.testing.assert_allclose(on[col].to_numpy(), off[col].to_numpy())


def test_split_is_deterministic():
    df = _gapped_track()
    a = split_on_gaps(df, max_gap_s=600.0)
    b = split_on_gaps(df, max_gap_s=600.0)
    pd.testing.assert_frame_equal(a, b)
