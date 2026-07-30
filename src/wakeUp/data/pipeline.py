"""Clean → split on gaps → resample → window pipeline (Phase 1).

Source-agnostic: works on any long-format frame with columns
``mmsi, timestamp, lat, lon, sog, cog[, heading]`` — synthetic or real.

A real MarineCadastre CSV can be adapted with :func:`load_marinecadastre_csv`,
which renames the standard AIS columns onto our schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wakeUp.config import DataConfig

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
    """Load a MarineCadastre AIS CSV into the wakeUp schema."""
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
# Gap segmentation
# --------------------------------------------------------------------------- #
def split_on_gaps(df: pd.DataFrame, max_gap_s: float | None) -> pd.DataFrame:
    """Label contiguous observation runs per vessel, breaking at long silences.

    Real AIS goes quiet — the vessel leaves receiver range, the transponder is
    switched off, or a spoofer simply stops transmitting. :func:`resample_track`
    interpolates onto a fixed grid, so without this step a multi-hour silence is
    *filled in* with smooth, entirely fabricated motion, which then gets
    windowed and labelled clean. Splitting first keeps every resampled point
    bracketed by real observations.

    ``max_gap_s=None`` disables splitting (one segment per vessel). The
    synthetic fleet is gap-free by construction, so that is the default and
    leaves the benchmark's recorded numbers untouched.
    """
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    if max_gap_s is None:
        df["segment"] = 0
        return df
    gap = df.groupby("mmsi")["timestamp"].diff().dt.total_seconds()
    starts_segment = (gap > float(max_gap_s)).fillna(False)
    df["segment"] = starts_segment.groupby(df["mmsi"]).cumsum().astype(int)
    return df


def drop_short_segments(df: pd.DataFrame, min_points: int) -> pd.DataFrame:
    """Drop segments with too few observations to window meaningfully.

    Mirrors the short-track filter in :func:`clean_ais`, but on the segment —
    the unit that actually survives into windows once gaps are honoured. With
    splitting disabled every vessel is a single segment, so this is a no-op.
    """
    counts = df.groupby(["mmsi", "segment"])["timestamp"].transform("size")
    return df[counts >= min_points].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def _group_keys(df: pd.DataFrame) -> list[str]:
    """Group by vessel, and by segment too once gaps have been labelled."""
    return ["mmsi", "segment"] if "segment" in df.columns else ["mmsi"]


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
    if "segment" in track.columns:
        out["segment"] = track["segment"].iloc[0]

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
        resample_track(g, cfg.cadence_s)
        for _, g in df.groupby(_group_keys(df), sort=True)
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

    Windows never straddle a gap: once :func:`split_on_gaps` has labelled
    segments, slicing is per segment, so no window mixes points from either
    side of a silence.
    """
    L, S = cfg.window_len, cfg.window_stride
    out_frames = []
    wid = 0
    for _, g in df.groupby(_group_keys(df), sort=True):
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
    """Full Phase-1 pipeline: clean → split on gaps → resample → window."""
    cleaned = clean_ais(df_raw, cfg)
    segmented = split_on_gaps(cleaned, cfg.max_gap_s)
    segmented = drop_short_segments(segmented, cfg.min_track_points)
    resampled = resample_all(segmented, cfg)
    windows = segment_windows(resampled, cfg)
    return windows
