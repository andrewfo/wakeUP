"""Detection latency: points-from-onset to first alarm (Phase 5).

PR-AUC says *whether* a detector separates attacked windows from clean ones; it
says nothing about *how long* the spoofer operates before the alarm fires. For
a live AIS feed that lag is the operational number: a drift caught 20 points
after onset has already displaced the vessel most of the way.

Attacks are injected as a contiguous span inside a window, so each attacked
window has a well-defined **onset** (the first ``is_attack`` point). The metric
replays each held-out window as a stream: for every prefix length ``t`` the
detector scores the truncated window, and an alarm fires the first time the
score crosses a threshold calibrated — at that same prefix length — to a fixed
false-positive rate on the clean windows. Calibrating per length matters
because score distributions shift as windows grow (aggregate features sharpen,
reconstruction error accumulates); a single full-window threshold would let
short prefixes alarm for free.

Latency for a detected window is ``(first alarming prefix end) - onset`` in
points (and in seconds via the resampling cadence); an alarm on a prefix that
contains no attacked point yet is a false positive and does not count as a
detection. Windows that never alarm are misses, reported through the
``detected`` flag rather than dropped, so detection rate and latency read
together.

The detector is fit **once** on full-length training windows (held-out by
vessel, supervised where supported) and only *scored* on prefixes — the
streaming deployment scenario, where the model is trained offline and applied
to a growing track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wakeUp.eval.metrics import dominant_attack_type
from wakeUp.eval.robustness import DetectorFactories, default_detectors
from wakeUp.eval.splits import train_test_split_by_vessel, window_labels
from wakeUp.features import build_feature_matrix


def clean_fpr_threshold(
    scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05
) -> float:
    """Score threshold whose false-positive rate on clean windows is <= max_fpr."""
    clean = np.asarray(scores)[np.asarray(labels) == 0]
    if clean.size == 0:
        raise ValueError("no clean windows to calibrate the alarm threshold on")
    # method="higher" picks an actual clean score at/above the quantile
    # position, so the strict `score > threshold` alarm rule stays within the
    # FPR budget even with ties or small clean counts.
    return float(np.quantile(clean, 1.0 - max_fpr, method="higher"))


def _cadence_seconds(windows: pd.DataFrame) -> float:
    """Median within-window sampling interval, in seconds."""
    ts = windows.sort_values(["window_id", "point_idx"]).groupby("window_id")[
        "timestamp"
    ].diff()
    if pd.api.types.is_timedelta64_dtype(ts):
        return float(ts.dt.total_seconds().median())
    return float(ts.median())


def detection_latency(
    windows: pd.DataFrame,
    detector,
    min_points: int = 8,
    max_fpr: float = 0.05,
) -> pd.DataFrame:
    """Streaming-prefix latency for one fitted detector on labelled windows.

    Returns one row per **attacked** window: ``window_id, mmsi, attack_type,
    onset_idx, detected, latency_points, latency_s``. ``latency_points`` is
    NaN for misses; 0 means the alarm fired on the first prefix containing an
    attacked point.
    """
    labels = window_labels(windows)
    wids = list(windows.groupby("window_id", sort=True).groups)
    dom = dominant_attack_type(windows).reindex(wids)
    mmsi = windows.groupby("window_id", sort=True)["mmsi"].first().reindex(wids)
    onset = (
        windows[windows["is_attack"] == 1]
        .groupby("window_id")["point_idx"]
        .min()
        .reindex(wids)
    )
    length = int(windows["point_idx"].max()) + 1
    cadence_s = _cadence_seconds(windows)
    consumes = getattr(detector, "consumes_windows", False)

    prefix_lens = list(range(min_points, length + 1))
    alarm = np.zeros((len(prefix_lens), len(wids)), dtype=bool)
    for k, t in enumerate(prefix_lens):
        prefix = windows[windows["point_idx"] < t]
        payload = prefix if consumes else build_feature_matrix(prefix)[0]
        scores = np.asarray(detector.score(payload))
        thr = clean_fpr_threshold(scores, labels, max_fpr)
        alarm[k] = scores > thr

    rows = []
    for i, wid in enumerate(wids):
        if labels[i] == 0:
            continue
        o = int(onset.loc[wid])
        # An alarm only counts once the prefix actually contains attacked
        # points: prefix length t covers indices 0..t-1, so require t > onset.
        valid = [k for k, t in enumerate(prefix_lens) if t > o and alarm[k, i]]
        detected = bool(valid)
        lat = float(prefix_lens[valid[0]] - 1 - o) if detected else float("nan")
        rows.append(
            {
                "window_id": wid,
                "mmsi": mmsi.loc[wid],
                "attack_type": dom.loc[wid],
                "onset_idx": o,
                "detected": detected,
                "latency_points": lat,
                "latency_s": lat * cadence_s,
            }
        )
    return pd.DataFrame(rows)


def run_detection_latency(
    attacked: pd.DataFrame,
    detectors: DetectorFactories | None = None,
    min_points: int = 8,
    max_fpr: float = 0.05,
    test_frac: float = 0.3,
    split_seed: int = 0,
) -> pd.DataFrame:
    """Latency for every detector under the held-out-by-vessel protocol.

    Each detector is fit once on full-length train-vessel windows (with labels
    where it ``supports_supervision``) and replayed over the held-out windows'
    prefixes. Returns the concatenated per-window frames with a ``detector``
    column.
    """
    factories = detectors or default_detectors()
    fit_frame, score_frame = train_test_split_by_vessel(
        attacked, test_frac=test_frac, seed=split_seed
    )
    fit_feat, _ = build_feature_matrix(fit_frame)

    frames = []
    for name, make in factories.items():
        det = make()
        payload = fit_frame if getattr(det, "consumes_windows", False) else fit_feat
        if getattr(det, "supports_supervision", False):
            det.fit(payload, supervised=True)
        else:
            det.fit(payload)
        lat = detection_latency(score_frame, det, min_points=min_points, max_fpr=max_fpr)
        lat.insert(0, "detector", name)
        frames.append(lat)
    return pd.concat(frames, ignore_index=True)


def summarize_latency(latency: pd.DataFrame) -> pd.DataFrame:
    """Detection rate + latency quantiles per (detector, attack family)."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        hits = g[g["detected"]]
        return pd.Series(
            {
                "n": len(g),
                "detection_rate": float(g["detected"].mean()),
                "median_latency_points": float(hits["latency_points"].median()),
                "median_latency_s": float(hits["latency_s"].median()),
            }
        )

    by = ["detector", "attack_type"] if "detector" in latency.columns else ["attack_type"]
    return latency.groupby(by).apply(_agg, include_groups=False).reset_index()
