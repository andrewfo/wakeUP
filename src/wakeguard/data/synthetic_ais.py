"""Synthetic clean-AIS fleet generator.

Phase 1 of the plan ingests real open AIS (MarineCadastre / Danish Maritime
Authority). Real files require a network download, so to keep the benchmark
reproducible *offline* we also ship a physics-based generator that emits
kinematically self-consistent vessel tracks: each vessel follows a smooth
heading with bounded turn-rate and bounded acceleration, and its reported
SOG/COG are derived from the actual motion (plus small sensor noise). That
self-consistency is exactly what the attack injectors later break, which is
what makes the labels meaningful.

The emitted frame matches the columns we keep from real AIS so downstream
code is source-agnostic:

    mmsi, timestamp (UTC), lat, lon, sog (knots), cog (deg), heading (deg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wakeguard import geo
from wakeguard.config import DataConfig

# Broad, plausible cruising speeds (knots) for a mixed fleet.
_SPEED_RANGE_KN = (6.0, 18.0)


def _simulate_one(
    rng: np.random.Generator,
    mmsi: int,
    start_lat: float,
    start_lon: float,
    n_steps: int,
    cadence_s: int,
) -> pd.DataFrame:
    """Integrate one vessel forward with bounded turn-rate and acceleration."""
    dt = float(cadence_s)

    speed_kn = rng.uniform(*_SPEED_RANGE_KN)
    speed_ms = geo.knots_to_ms(speed_kn)
    course = rng.uniform(0.0, 360.0)

    # Per-step dynamics limits (gentle — this is open-water cruising).
    max_turn_rate = 1.5          # deg / s
    max_accel = 0.03             # m / s^2
    target_speed = speed_ms

    lat = np.empty(n_steps)
    lon = np.empty(n_steps)
    sog_kn = np.empty(n_steps)
    cog = np.empty(n_steps)

    lat[0], lon[0] = start_lat, start_lon
    cur_lat, cur_lon = start_lat, start_lon

    # Slowly varying course target so tracks curve realistically.
    course_target = course
    for i in range(n_steps):
        if i > 0:
            # occasionally pick a new course target (a manoeuvre)
            if rng.random() < 0.02:
                course_target = (course_target + rng.uniform(-60, 60)) % 360.0
            # steer toward target within turn-rate limit
            dcourse = geo.angular_diff_deg(course_target, course)
            max_step = max_turn_rate * dt
            course = (course + np.clip(dcourse, -max_step, max_step)) % 360.0

            # accelerate toward target speed within accel limit
            dv = np.clip(target_speed - speed_ms, -max_accel * dt, max_accel * dt)
            speed_ms = max(0.0, speed_ms + dv)
            if rng.random() < 0.01:
                target_speed = geo.knots_to_ms(rng.uniform(*_SPEED_RANGE_KN))

            step_dist = speed_ms * dt
            cur_lat, cur_lon = geo.destination_point(
                cur_lat, cur_lon, course, step_dist
            )
            cur_lat = float(cur_lat)
            cur_lon = float(cur_lon)
            lat[i], lon[i] = cur_lat, cur_lon

        # reported values derive from true motion + small sensor noise
        sog_kn[i] = geo.ms_to_knots(speed_ms) + rng.normal(0.0, 0.05)
        cog[i] = (course + rng.normal(0.0, 0.4)) % 360.0

    sog_kn = np.clip(sog_kn, 0.0, None)
    ts0 = np.datetime64("2024-01-01T00:00:00")
    timestamps = ts0 + (np.arange(n_steps) * cadence_s).astype("timedelta64[s]")

    return pd.DataFrame(
        {
            "mmsi": np.int64(mmsi),
            "timestamp": timestamps,
            "lat": lat,
            "lon": lon,
            "sog": sog_kn,
            "cog": cog,
            "heading": cog,  # assume heading ~ course for the generator
        }
    )


def generate_fleet(cfg: DataConfig) -> pd.DataFrame:
    """Generate a full synthetic fleet as one long-format AIS frame.

    Vessel start points are scattered uniformly inside ``cfg.region_bbox``.
    A small amount of raw jitter is injected (duplicate rows, out-of-cadence
    timestamps) so the cleaning stage in the pipeline has something to do.
    """
    rng = np.random.default_rng(cfg.seed)
    lat_min, lon_min, lat_max, lon_max = cfg.region_bbox
    n_steps = int(round(cfg.duration_hours * 3600 / cfg.cadence_s))

    frames = []
    for k in range(cfg.n_vessels):
        mmsi = 200_000_000 + k  # MID 200 block, synthetic
        start_lat = rng.uniform(lat_min, lat_max)
        start_lon = rng.uniform(lon_min, lon_max)
        df = _simulate_one(rng, mmsi, start_lat, start_lon, n_steps, cfg.cadence_s)

        # inject raw-data messiness: ~1% duplicate rows
        dup_mask = rng.random(len(df)) < 0.01
        if dup_mask.any():
            df = pd.concat([df, df[dup_mask]], ignore_index=True)
        frames.append(df)

    fleet = pd.concat(frames, ignore_index=True)
    # shuffle to mimic interleaved multi-vessel message stream
    fleet = fleet.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return fleet
