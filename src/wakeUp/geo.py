"""Great-circle geometry helpers.

All angles are in degrees on input/output; internal math uses radians.
Distances are metres, bearings are degrees clockwise from true north in
[0, 360), speeds are metres/second unless a helper says otherwise.

These functions are vectorised over NumPy arrays and are the numerical
backbone for both the synthetic track generator and the kinematic feature
extractor, so they are covered directly by unit tests.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
_MS_TO_KNOTS = 1.943_844_49
_KNOTS_TO_MS = 1.0 / _MS_TO_KNOTS


def ms_to_knots(v_ms: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(v_ms) * _MS_TO_KNOTS


def knots_to_ms(v_kn: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(v_kn) * _KNOTS_TO_MS


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two points (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, in [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    y = np.sin(dlam) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlam)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def destination_point(lat, lon, bearing_deg, distance_m):
    """Point reached from (lat, lon) travelling ``distance_m`` along ``bearing_deg``.

    Returns ``(lat2, lon2)`` in degrees. Vectorised.
    """
    lat, lon, bearing_deg, distance_m = map(
        np.asarray, (lat, lon, bearing_deg, distance_m)
    )
    p1 = np.radians(lat)
    l1 = np.radians(lon)
    theta = np.radians(bearing_deg)
    delta = distance_m / EARTH_RADIUS_M
    sin_p2 = np.sin(p1) * np.cos(delta) + np.cos(p1) * np.sin(delta) * np.cos(theta)
    p2 = np.arcsin(np.clip(sin_p2, -1.0, 1.0))
    y = np.sin(theta) * np.sin(delta) * np.cos(p1)
    x = np.cos(delta) - np.sin(p1) * sin_p2
    l2 = l1 + np.arctan2(y, x)
    lat2 = np.degrees(p2)
    lon2 = (np.degrees(l2) + 540.0) % 360.0 - 180.0  # normalise to [-180, 180)
    return lat2, lon2


def angular_diff_deg(a, b):
    """Smallest signed difference ``a - b`` wrapped to (-180, 180]."""
    d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    # map -180 -> 180 for a canonical half-open convention
    d = np.where(d == -180.0, 180.0, d)
    return d
