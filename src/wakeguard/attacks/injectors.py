"""Synthetic attack injectors — the labeled-evaluation backbone.

Each injector takes one trajectory *window* (a per-point DataFrame for a
single ``window_id``, sorted by ``point_idx``) and returns a corrupted copy
carrying per-point labels:

    is_attack     : int   1 for a spoofed/anomalous point, else 0
    attack_type   : str   the attack name on corrupted points, else "none"

Design choice that makes the labels detectable: injectors alter *position*
(and, for the kinematic case, position spacing) but by default leave the
*reported* SOG/COG fields untouched — mirroring a naive spoofer who moves the
plotted position without also forging a self-consistent velocity report. That
is precisely the SOG↔Δposition disagreement the kinematic features look for.
Set ``forge_velocity=True`` for the harder variant where the injector also
rewrites SOG/COG to match the fake motion.

Every injector is deterministic given its ``rng`` so the benchmark is
reproducible, and each is covered by unit tests in ``tests/test_attacks.py``.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from wakeguard import geo
from wakeguard.config import AttackConfig


class AttackType(str, Enum):
    NONE = "none"
    POSITION_JUMP = "position_jump"
    KINEMATIC_IMPOSSIBLE = "kinematic_impossible"
    IDENTITY_SWAP = "identity_swap"
    REPLAY = "replay"
    GRADUAL_DRIFT = "gradual_drift"


def _prep(window: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with label columns initialised to clean."""
    w = window.sort_values("point_idx").reset_index(drop=True).copy()
    if "is_attack" not in w.columns:
        w["is_attack"] = 0
    if "attack_type" not in w.columns:
        w["attack_type"] = AttackType.NONE.value
    return w


def _pick_span(rng: np.random.Generator, n: int, min_frac=0.25, max_frac=0.6):
    """Pick a contiguous [start, end) span covering a fraction of the window."""
    length = max(2, int(round(rng.uniform(min_frac, max_frac) * n)))
    length = min(length, n)
    start = int(rng.integers(0, n - length + 1))
    return start, start + length


def _recompute_reported_velocity(w: pd.DataFrame) -> None:
    """Overwrite sog/cog in-place to match the (possibly faked) positions."""
    lat = w["lat"].to_numpy()
    lon = w["lon"].to_numpy()
    ts = pd.to_datetime(w["timestamp"]).to_numpy()
    dt = np.diff(ts) / np.timedelta64(1, "s")
    dt[dt == 0] = np.nan
    dist = geo.haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    brg = geo.initial_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    sog_ms = dist / dt
    sog_kn = geo.ms_to_knots(sog_ms)
    # forward-fill the last point from the previous step
    w.loc[: len(w) - 2, "sog"] = np.nan_to_num(sog_kn, nan=0.0)
    w.loc[len(w) - 1, "sog"] = w.loc[len(w) - 2, "sog"] if len(w) > 1 else 0.0
    w.loc[: len(w) - 2, "cog"] = brg % 360.0
    w.loc[len(w) - 1, "cog"] = w.loc[len(w) - 2, "cog"] if len(w) > 1 else 0.0
    w["heading"] = w["cog"]


# --------------------------------------------------------------------------- #
# 1. Position jump / teleport
# --------------------------------------------------------------------------- #
def inject_position_jump(
    window: pd.DataFrame,
    rng: np.random.Generator,
    cfg: AttackConfig,
    forge_velocity: bool = False,
) -> pd.DataFrame:
    """Displace a contiguous span by a large fixed offset (teleport)."""
    w = _prep(window)
    n = len(w)
    start, end = _pick_span(rng, n)

    bearing = rng.uniform(0, 360)
    dist_m = cfg.jump_km * 1000.0
    new_lat, new_lon = geo.destination_point(
        w.loc[start : end - 1, "lat"].to_numpy(),
        w.loc[start : end - 1, "lon"].to_numpy(),
        bearing,
        dist_m,
    )
    w.loc[start : end - 1, "lat"] = new_lat
    w.loc[start : end - 1, "lon"] = new_lon
    w.loc[start : end - 1, "is_attack"] = 1
    w.loc[start : end - 1, "attack_type"] = AttackType.POSITION_JUMP.value

    if forge_velocity:
        _recompute_reported_velocity(w)
    return w


# --------------------------------------------------------------------------- #
# 2. Kinematically impossible speed / acceleration
# --------------------------------------------------------------------------- #
def inject_kinematic_impossible(
    window: pd.DataFrame,
    rng: np.random.Generator,
    cfg: AttackConfig,
    forge_velocity: bool = False,
) -> pd.DataFrame:
    """Inflate the per-step spacing of a span so implied speed is impossible.

    Points in the span are re-placed along the existing course but with the
    step distance multiplied by ``cfg.speed_multiplier``, producing an
    acceleration/speed profile no real vessel can achieve.
    """
    w = _prep(window)
    n = len(w)
    start, end = _pick_span(rng, n, min_frac=0.2, max_frac=0.4)
    if start == 0:
        start = 1  # need a preceding anchor point

    lat = w["lat"].to_numpy().copy()
    lon = w["lon"].to_numpy().copy()
    for i in range(start, end):
        d = geo.haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
        b = geo.initial_bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
        nlat, nlon = geo.destination_point(
            lat[i - 1], lon[i - 1], b, float(d) * cfg.speed_multiplier
        )
        lat[i], lon[i] = float(nlat), float(nlon)
    w["lat"], w["lon"] = lat, lon
    w.loc[start : end - 1, "is_attack"] = 1
    w.loc[start : end - 1, "attack_type"] = AttackType.KINEMATIC_IMPOSSIBLE.value

    if forge_velocity:
        _recompute_reported_velocity(w)
    return w


# --------------------------------------------------------------------------- #
# 3. Identity swap (MMSI reassignment / track splice)
# --------------------------------------------------------------------------- #
def inject_identity_swap(
    window: pd.DataFrame,
    rng: np.random.Generator,
    cfg: AttackConfig,
    donor: pd.DataFrame | None = None,
    forge_velocity: bool = False,
) -> pd.DataFrame:
    """Splice a donor vessel's motion into the tail under the same MMSI.

    Represents a target adopting/rebroadcasting another vessel's identity:
    the MMSI stays constant but the underlying kinematics jump to a different
    track. The donor tail is translated to abut the splice point so the
    anomaly is the *dynamics* mismatch, not a trivial coordinate offset.
    """
    w = _prep(window)
    n = len(w)
    if donor is None or len(donor) < n:
        return w  # no valid donor -> leave clean
    d = donor.sort_values("point_idx").reset_index(drop=True)

    split = n // 2
    # translate donor so its split point coincides with our split point
    dlat = w.loc[split, "lat"] - d.loc[split, "lat"]
    dlon = w.loc[split, "lon"] - d.loc[split, "lon"]
    w.loc[split:, "lat"] = d.loc[split : n - 1, "lat"].to_numpy() + dlat
    w.loc[split:, "lon"] = d.loc[split : n - 1, "lon"].to_numpy() + dlon
    w.loc[split:, "sog"] = d.loc[split : n - 1, "sog"].to_numpy()
    w.loc[split:, "cog"] = d.loc[split : n - 1, "cog"].to_numpy()
    w.loc[split:, "heading"] = d.loc[split : n - 1, "cog"].to_numpy()
    w.loc[split:, "is_attack"] = 1
    w.loc[split:, "attack_type"] = AttackType.IDENTITY_SWAP.value

    if forge_velocity:
        _recompute_reported_velocity(w)
    return w


# --------------------------------------------------------------------------- #
# 4. Replay (paste an earlier segment)
# --------------------------------------------------------------------------- #
def inject_replay(
    window: pd.DataFrame,
    rng: np.random.Generator,
    cfg: AttackConfig,
    forge_velocity: bool = False,
) -> pd.DataFrame:
    """Overwrite the tail with an earlier segment of the same track.

    The vessel appears to teleport back to a previously reported position and
    re-trace it — the classic replay/record-and-rebroadcast signature.
    """
    w = _prep(window)
    n = len(w)
    seg = n // 3
    if seg < 2:
        return w
    src_start = 0
    dst_start = n - seg
    w.loc[dst_start:, "lat"] = w.loc[src_start : src_start + seg - 1, "lat"].to_numpy()
    w.loc[dst_start:, "lon"] = w.loc[src_start : src_start + seg - 1, "lon"].to_numpy()
    w.loc[dst_start:, "sog"] = w.loc[src_start : src_start + seg - 1, "sog"].to_numpy()
    w.loc[dst_start:, "cog"] = w.loc[src_start : src_start + seg - 1, "cog"].to_numpy()
    w.loc[dst_start:, "heading"] = w.loc[dst_start:, "cog"]
    w.loc[dst_start:, "is_attack"] = 1
    w.loc[dst_start:, "attack_type"] = AttackType.REPLAY.value

    if forge_velocity:
        _recompute_reported_velocity(w)
    return w


# --------------------------------------------------------------------------- #
# 5. Gradual drift (subtle offset ramp)
# --------------------------------------------------------------------------- #
def inject_gradual_drift(
    window: pd.DataFrame,
    rng: np.random.Generator,
    cfg: AttackConfig,
    forge_velocity: bool = False,
) -> pd.DataFrame:
    """Add a slowly ramping positional offset (0 → drift_total_km).

    The subtle attack: per-step position error stays tiny so reported SOG/COG
    remain almost consistent, but the track diverges from truth over the
    window. Robustness curves sweep ``drift_total_km``.
    """
    w = _prep(window)
    n = len(w)
    bearing = rng.uniform(0, 360)
    ramp = np.linspace(0.0, cfg.drift_total_km * 1000.0, n)
    new_lat, new_lon = geo.destination_point(
        w["lat"].to_numpy(), w["lon"].to_numpy(), bearing, ramp
    )
    w["lat"], w["lon"] = new_lat, new_lon
    # label the portion where the offset is meaningfully non-zero
    active = ramp > 0.1 * cfg.drift_total_km * 1000.0
    w.loc[active, "is_attack"] = 1
    w.loc[active, "attack_type"] = AttackType.GRADUAL_DRIFT.value

    if forge_velocity:
        _recompute_reported_velocity(w)
    return w


INJECTORS = {
    AttackType.POSITION_JUMP: inject_position_jump,
    AttackType.KINEMATIC_IMPOSSIBLE: inject_kinematic_impossible,
    AttackType.IDENTITY_SWAP: inject_identity_swap,
    AttackType.REPLAY: inject_replay,
    AttackType.GRADUAL_DRIFT: inject_gradual_drift,
}


# --------------------------------------------------------------------------- #
# Dataset-level orchestration
# --------------------------------------------------------------------------- #
def build_attacked_dataset(
    windows: pd.DataFrame,
    cfg: AttackConfig,
    attack_types: list[AttackType] | None = None,
) -> pd.DataFrame:
    """Corrupt a fraction of windows with randomly assigned attacks.

    A ``cfg.contamination`` fraction of windows is selected and each assigned
    one attack type (uniformly from ``attack_types``); the rest stay clean.
    Returns the full per-point frame with ``is_attack`` / ``attack_type``
    labels and a per-window ``window_label`` (1 if the window contains any
    attacked point).
    """
    if attack_types is None:
        attack_types = [
            AttackType.POSITION_JUMP,
            AttackType.KINEMATIC_IMPOSSIBLE,
            AttackType.IDENTITY_SWAP,
            AttackType.REPLAY,
            AttackType.GRADUAL_DRIFT,
        ]
    rng = np.random.default_rng(cfg.seed)
    wids = np.array(sorted(windows["window_id"].unique()))
    n_attack = int(round(cfg.contamination * len(wids)))
    attacked_ids = set(rng.choice(wids, size=n_attack, replace=False).tolist())

    # pre-index windows for donor lookup (identity swap)
    groups = {wid: g for wid, g in windows.groupby("window_id", sort=True)}
    wid_to_mmsi = {wid: g["mmsi"].iloc[0] for wid, g in groups.items()}

    out = []
    for wid in wids:
        w = groups[wid]
        if wid not in attacked_ids:
            w = _prep(w)
            out.append(w)
            continue
        atype = attack_types[int(rng.integers(0, len(attack_types)))]
        fn = INJECTORS[atype]
        if atype is AttackType.IDENTITY_SWAP:
            donor = _pick_donor(rng, wids, wid, wid_to_mmsi, groups)
            corrupted = fn(w, rng, cfg, donor=donor)
        else:
            corrupted = fn(w, rng, cfg)
        out.append(corrupted)

    full = pd.concat(out, ignore_index=True)
    win_label = (
        full.groupby("window_id")["is_attack"].transform("max").astype(int)
    )
    full["window_label"] = win_label
    return full


def _pick_donor(rng, wids, wid, wid_to_mmsi, groups):
    """Pick a window from a different vessel to donate motion."""
    target_mmsi = wid_to_mmsi[wid]
    candidates = [w for w in wids if wid_to_mmsi[w] != target_mmsi]
    if not candidates:
        return None
    return groups[candidates[int(rng.integers(0, len(candidates)))]]
