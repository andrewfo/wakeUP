"""Kinematic-consistency features (Phase 3).

Two granularities:

* :func:`point_features` — per-point derived kinematics for one window
  (implied speed from Δposition, acceleration, turn-rate, and the key
  consistency residuals: reported SOG vs position-implied speed, and reported
  COG vs position-implied bearing).

* :func:`window_feature_vector` — a fixed-length summary vector per window
  (robust aggregates of the point features), which is what the classical
  baselines (IsolationForest, threshold rules) consume.

The consistency residuals are the physically meaningful signal: on a genuine
track the reported velocity and the velocity implied by successive positions
agree to within sensor noise; most spoofing breaks that agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wakeUp import geo

# Physical plausibility ceilings for a mixed commercial fleet.
MAX_PLAUSIBLE_SPEED_KN = 40.0
MAX_PLAUSIBLE_ACCEL_MS2 = 1.0
MAX_PLAUSIBLE_TURN_RATE_DPS = 5.0


def point_features(window: pd.DataFrame) -> pd.DataFrame:
    """Per-point kinematic features for one window (sorted by point_idx)."""
    w = window.sort_values("point_idx").reset_index(drop=True)
    lat = w["lat"].to_numpy()
    lon = w["lon"].to_numpy()
    ts = pd.to_datetime(w["timestamp"]).to_numpy()
    n = len(w)

    dt = np.diff(ts) / np.timedelta64(1, "s")
    dt_safe = np.where(dt == 0, np.nan, dt)

    dist = geo.haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    implied_ms = dist / dt_safe
    implied_speed_kn = geo.ms_to_knots(implied_ms)
    implied_bearing = geo.initial_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # pad diffs back to length n (first point has no predecessor)
    def _pad(arr):
        return np.concatenate([[np.nan], arr])

    implied_speed_kn = _pad(implied_speed_kn)
    implied_bearing = _pad(implied_bearing)

    # acceleration from implied speed
    accel_ms2 = np.full(n, np.nan)
    dv = np.diff(geo.knots_to_ms(implied_speed_kn))
    accel_ms2[2:] = dv[1:] / dt_safe[1:]

    # turn rate from reported COG
    cog = w["cog"].to_numpy()
    turn = geo.angular_diff_deg(cog[1:], cog[:-1])
    turn_rate = np.full(n, np.nan)
    turn_rate[1:] = turn / dt_safe

    # consistency residuals
    sog = w["sog"].to_numpy()
    speed_resid = implied_speed_kn - sog                       # kn
    cog_resid = np.abs(geo.angular_diff_deg(implied_bearing, cog))  # deg

    gap_s = w["gap_s"].to_numpy() if "gap_s" in w.columns else np.full(n, np.nan)

    return pd.DataFrame(
        {
            "point_idx": w["point_idx"].to_numpy(),
            "implied_speed_kn": implied_speed_kn,
            "reported_sog_kn": sog,
            "accel_ms2": accel_ms2,
            "turn_rate_dps": turn_rate,
            "speed_resid_kn": speed_resid,
            "cog_resid_deg": cog_resid,
            "gap_s": gap_s,
        }
    )


def _robust_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_max": float(np.max(np.abs(x))),
        f"{prefix}_p90": float(np.percentile(np.abs(x), 90)),
    }


FEATURE_NAMES = [
    f"{p}_{s}"
    for p in (
        "implied_speed",
        "accel",
        "turn_rate",
        "speed_resid",
        "cog_resid",
        "gap",
    )
    for s in ("mean", "std", "max", "p90")
] + [
    "frac_speed_over_limit",
    "frac_accel_over_limit",
    "frac_turn_over_limit",
]


def window_feature_vector(window: pd.DataFrame) -> dict[str, float]:
    """Fixed-length feature summary for one window."""
    pf = point_features(window)
    feats: dict[str, float] = {}
    feats.update(_robust_stats(pf["implied_speed_kn"].to_numpy(), "implied_speed"))
    feats.update(_robust_stats(pf["accel_ms2"].to_numpy(), "accel"))
    feats.update(_robust_stats(pf["turn_rate_dps"].to_numpy(), "turn_rate"))
    feats.update(_robust_stats(pf["speed_resid_kn"].to_numpy(), "speed_resid"))
    feats.update(_robust_stats(pf["cog_resid_deg"].to_numpy(), "cog_resid"))
    feats.update(_robust_stats(pf["gap_s"].to_numpy(), "gap"))

    spd = pf["implied_speed_kn"].to_numpy()
    acc = pf["accel_ms2"].to_numpy()
    trn = pf["turn_rate_dps"].to_numpy()
    feats["frac_speed_over_limit"] = _frac_over(spd, MAX_PLAUSIBLE_SPEED_KN)
    feats["frac_accel_over_limit"] = _frac_over(np.abs(acc), MAX_PLAUSIBLE_ACCEL_MS2)
    feats["frac_turn_over_limit"] = _frac_over(
        np.abs(trn), MAX_PLAUSIBLE_TURN_RATE_DPS
    )
    return feats


def _frac_over(x: np.ndarray, limit: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.mean(x > limit))


def build_feature_matrix(windows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute the per-window feature matrix and window labels.

    Returns ``(features_df, labels)`` where ``features_df`` is indexed by
    ``window_id`` with columns == FEATURE_NAMES, and ``labels`` is the aligned
    per-window attack label (or all-zeros if unlabeled).
    """
    rows = []
    wids = []
    labels = []
    has_label = "window_label" in windows.columns
    for wid, g in windows.groupby("window_id", sort=True):
        rows.append(window_feature_vector(g))
        wids.append(wid)
        labels.append(int(g["window_label"].iloc[0]) if has_label else 0)
    feat_df = pd.DataFrame(rows, index=pd.Index(wids, name="window_id"))
    feat_df = feat_df[FEATURE_NAMES]  # stable column order
    return feat_df, np.asarray(labels, dtype=int)
