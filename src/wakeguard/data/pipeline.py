"""Clean → resample → window pipeline (Phase 1).

Source-agnostic: works on any long-format frame with columns
``mmsi, timestamp, lat, lon, sog, cog[, heading]`` — synthetic or real.

A real MarineCadastre CSV can be adapted with :func:`load_marinecadastre_csv`,
which renames the standard AIS columns onto our schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wakeguard.config import DataConfig

SCHEMA = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "heading"]


# --------------------------------------------------------------------------- #
# Real-data adapter (kept minimal; the milestone runs on synthetic data)
# --------------------------------------------------------------------------- #
_MARINECADASTRE_MAP = {
    "MMSI": "mmsi",
    "BaseDateTime": "timestamp",
    "LAT": "lat",
    "LON": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
}


def load_marinecadastre_csv(path: str | Path) -> pd.DataFrame:
    """Load a MarineCadastre AIS CSV into the wakeguard schema."""
    df = pd.read_csv(path)
    df = df.rename(columns=_MARINECADASTRE_MAP)
    keep = [c for c in SCHEMA if c in df.columns]
    df = df[keep].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_ais(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Dedup, sort, drop obviously invalid rows and too-short tracks."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # basic validity: lat/lon in range, non-negative SOG
    lat_min, lon_min, lat_max, lon_max = cfg.region_bbox
    valid = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & (df["sog"] >= 0)
    )
    df = df[valid]

    # dedup exact (mmsi, timestamp) collisions, keep first
    df = df.sort_values(["mmsi", "timestamp"])
    df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="first")

    # drop short tracks
    counts = df.groupby("mmsi")["timestamp"].transform("size")
    df = df[counts >= cfg.min_track_points]

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def resample_track(track: pd.DataFrame, cadence_s: int) -> pd.DataFrame:
    """Resample one vessel's track onto a fixed cadence grid.

    lat/lon/sog/cog are linearly interpolated (cog via unwrapped angle so it
    doesn't average across the 360→0 seam). A ``gap_s`` column records the
    real spacing to the previous *original* fix, so downstream code can still
    see where the sensor actually went quiet.
    """
    track = track.sort_values("timestamp").set_index("timestamp")
    rule = f"{cadence_s}s"

    # numeric interpolation on a regular grid
    num = track[["lat", "lon", "sog"]].resample(rule).mean().interpolate("time")

    # circular columns: unwrap → interpolate → rewrap
    cog_rad = np.radians(track["cog"].to_numpy())
    unwrapped = pd.Series(np.degrees(np.unwrap(cog_rad)), index=track.index)
    cog = unwrapped.resample(rule).mean().interpolate("time") % 360.0

    out = num.copy()
    out["cog"] = cog
    out["heading"] = cog
    out["mmsi"] = track["mmsi"].iloc[0]

    # gap to previous original observation, mapped onto the grid
    orig_times = track.index.to_series()
    grid = out.index.to_series()
    prev_orig = pd.merge_asof(
        grid.rename("grid").reset_index(drop=True),
        orig_times.rename("orig").reset_index(drop=True),
        left_on="grid",
        right_on="orig",
        direction="backward",
    )
    out["gap_s"] = (
        (out.index - pd.DatetimeIndex(prev_orig["orig"])).total_seconds().to_numpy()
    )

    return out.reset_index().rename(columns={"index": "timestamp"})


def resample_all(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    parts = [
        resample_track(g, cfg.cadence_s) for _, g in df.groupby("mmsi", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def segment_windows(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Slice each resampled track into fixed-length overlapping windows.

    Returns a per-point frame with a ``window_id`` column (globally unique)
    and a ``point_idx`` position within the window. Windows that would run off
    the end of a track are dropped rather than padded.
    """
    L, S = cfg.window_len, cfg.window_stride
    out_frames = []
    wid = 0
    for mmsi, g in df.groupby("mmsi", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        for start in range(0, n - L + 1, S):
            w = g.iloc[start : start + L].copy()
            w["window_id"] = wid
            w["point_idx"] = np.arange(L)
            out_frames.append(w)
            wid += 1
    if not out_frames:
        return pd.DataFrame(columns=list(df.columns) + ["window_id", "point_idx"])
    return pd.concat(out_frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_dataset(df_raw: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Full Phase-1 pipeline: clean → resample → window."""
    cleaned = clean_ais(df_raw, cfg)
    resampled = resample_all(cleaned, cfg)
    windows = segment_windows(resampled, cfg)
    return windows
