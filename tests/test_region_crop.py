"""Study-area cropping tests (Phase 1, real-ingest prerequisite).

``clean_ais`` used to unpack ``cfg.region_bbox`` and then never filter on it, so
a whole-zone real download would have gone through the benchmark uncropped —
extra vessels in water the run is not about, redefining "normal" for every
detector fitted on it. These tests pin the crop, pin that it stays *off* by
default (unlike the gap split it is not a no-op on the synthetic fleet, so the
recorded numbers depend on it staying off), and pin its interaction with the gap
split, which is what keeps an out-and-back excursion from being interpolated
over.
"""

import numpy as np
import pandas as pd

from wakeUp.config import DataConfig
from wakeUp.data.pipeline import build_dataset, clean_ais, crop_to_bbox, split_on_gaps

BBOX = (36.0, -75.5, 37.5, -73.5)  # lat_min, lon_min, lat_max, lon_max


def _fixes(lats, lons, mmsi: int = 1, start: str = "2024-01-01T00:00:00"):
    """A regular one-minute track through the given positions."""
    n = len(lats)
    return pd.DataFrame(
        {
            "mmsi": mmsi,
            "timestamp": pd.date_range(pd.Timestamp(start), periods=n, freq="60s"),
            "lat": np.asarray(lats, dtype=float),
            "lon": np.asarray(lons, dtype=float),
            "sog": 10.0,
            "cog": 0.0,
            "heading": 0.0,
        }
    )


# --------------------------------------------------------------------------- #
# The crop itself
# --------------------------------------------------------------------------- #
def test_crop_keeps_inside_and_drops_outside():
    df = _fixes(
        lats=[36.5, 36.5, 40.0, 36.5, 20.0],
        lons=[-75.0, -74.0, -74.0, -80.0, -74.0],
    )
    out = crop_to_bbox(df, BBOX)
    assert len(out) == 2
    assert list(out.index) == [0, 1]  # rows kept, not renumbered


def test_crop_bounds_are_inclusive():
    """A fix exactly on a corner is inside the study area, not outside it."""
    corners = _fixes(
        lats=[36.0, 37.5, 36.0, 37.5],
        lons=[-75.5, -73.5, -73.5, -75.5],
    )
    assert len(crop_to_bbox(corners, BBOX)) == 4


def test_crop_is_a_row_filter_not_a_track_filter():
    """A vessel that leaves and returns keeps both stretches, minus the middle."""
    df = _fixes(lats=[36.5] * 4 + [40.0] * 3 + [36.6] * 4, lons=[-74.0] * 11)
    out = crop_to_bbox(df, BBOX)
    assert len(out) == 8
    assert out["mmsi"].nunique() == 1


# --------------------------------------------------------------------------- #
# Wiring into clean_ais
# --------------------------------------------------------------------------- #
def test_clean_does_not_crop_by_default():
    """The default is off, and every recorded benchmark number depends on it."""
    cfg = DataConfig(min_track_points=1, region_bbox=BBOX)
    assert cfg.crop_to_region is False
    df = _fixes(lats=[36.5, 40.0, 36.5], lons=[-74.0, -74.0, -74.0])
    assert len(clean_ais(df, cfg)) == 3


def test_clean_crops_when_enabled():
    cfg = DataConfig(min_track_points=1, region_bbox=BBOX, crop_to_region=True)
    df = _fixes(lats=[36.5, 40.0, 36.5], lons=[-74.0, -74.0, -74.0])
    out = clean_ais(df, cfg)
    assert len(out) == 2
    assert out["lat"].between(36.0, 37.5).all()


def test_crop_runs_before_the_short_track_filter():
    """A track the crop guts is dropped whole, not kept as a stub."""
    cfg = DataConfig(min_track_points=5, region_bbox=BBOX, crop_to_region=True)
    # inside the box for 3 fixes, outside for 20 — 3 < min_track_points
    df = _fixes(lats=[36.5] * 3 + [40.0] * 20, lons=[-74.0] * 23)
    assert len(clean_ais(df, cfg)) == 0
    # ...while a track with enough surviving fixes is kept
    df2 = _fixes(lats=[36.5] * 6 + [40.0] * 20, lons=[-74.0] * 26)
    assert len(clean_ais(df2, cfg)) == 6


# --------------------------------------------------------------------------- #
# The interaction that matters: an excursion is a gap, not smooth motion
# --------------------------------------------------------------------------- #
def test_crop_creates_a_hole_the_gap_split_catches():
    """Cropping out an excursion leaves a silence; splitting must break there."""
    cfg = DataConfig(min_track_points=1, region_bbox=BBOX, crop_to_region=True,
                     max_gap_s=600.0)
    # 30 fixes inside, 40 min outside the box, then 30 fixes back inside
    df = _fixes(lats=[36.5] * 30 + [40.0] * 40 + [36.6] * 30, lons=[-74.0] * 100)

    cleaned = clean_ais(df, cfg)
    assert len(cleaned) == 60

    # With the split on, the excursion reads as the dropout it is.
    split = split_on_gaps(cleaned, cfg.max_gap_s)
    assert sorted(split["segment"].unique()) == [0, 1]
    # Without it, the two stretches are one segment and resampling would
    # interpolate straight across the 40 minutes of missing track.
    assert split_on_gaps(cleaned, None)["segment"].nunique() == 1


# --------------------------------------------------------------------------- #
# Backwards compatibility — the recorded benchmark numbers depend on this
# --------------------------------------------------------------------------- #
def test_default_config_leaves_the_synthetic_pipeline_unchanged(small_cfg, windows):
    from wakeUp.data import generate_fleet

    assert small_cfg.crop_to_region is False
    rebuilt = build_dataset(generate_fleet(small_cfg), small_cfg)
    assert len(rebuilt) == len(windows)
    for col in ("lat", "lon", "sog", "cog"):
        np.testing.assert_allclose(
            rebuilt[col].to_numpy(), windows[col].to_numpy()
        )


def test_enabling_crop_is_not_a_noop_on_synthetic_data(small_cfg):
    """Unlike the gap split, this one really does move the dataset.

    Synthetic vessels start inside the box and integrate outwards, so the crop
    removes real points. Pinning that keeps anyone from flipping the flag on and
    re-reporting the recorded numbers as if nothing changed.
    """
    from dataclasses import replace

    from wakeUp.data import generate_fleet

    raw = generate_fleet(small_cfg)
    off = build_dataset(raw, small_cfg)
    on = build_dataset(raw, replace(small_cfg, crop_to_region=True))

    assert len(on) < len(off)
    assert on["lat"].between(*small_cfg.region_bbox[::2]).all()
    assert on["lon"].between(small_cfg.region_bbox[1], small_cfg.region_bbox[3]).all()
