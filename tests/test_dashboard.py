"""Dashboard tests (Phase 6).

The dashboard is presentation, so what must hold is the data contract: the
payload's blocks stay aligned with the sorted-``window_id`` ordering every
other harness uses, missing artifacts degrade to empty panels rather than
errors, and the rendered document is genuinely self-contained (payload
embedded, no external URLs). Torch-free by design.
"""

import json
import re

import numpy as np
import pytest

from wakeUp.attacks import build_attacked_dataset
from wakeUp.config import AttackConfig, ModelConfig
from wakeUp.eval.dashboard import build_payload, render_dashboard, write_dashboard
from wakeUp.eval.splits import train_test_split_by_vessel, window_labels
from wakeUp.features import build_feature_matrix
from wakeUp.models import IsolationForestDetector, KinematicRuleDetector

FAST = ModelConfig(seed=0, iforest_estimators=50)


@pytest.fixture(scope="module")
def payload(windows, tmp_path_factory):
    attacked = build_attacked_dataset(
        windows,
        AttackConfig(seed=7, jump_km=8.0, drift_total_km=3.0, speed_multiplier=6.0),
    )
    fit_frame, score_frame = train_test_split_by_vessel(attacked, test_frac=0.4)
    fit_feat, _ = build_feature_matrix(fit_frame)
    score_feat, _ = build_feature_matrix(score_frame)
    scores = {
        "KinematicRule": KinematicRuleDetector().fit(fit_feat).score(score_feat),
        "IsolationForest": IsolationForestDetector(FAST).fit(fit_feat).score(score_feat),
    }
    empty_dir = tmp_path_factory.mktemp("no_artifacts")
    return (
        build_payload(
            clean_windows=windows,
            attacked=attacked,
            score_frame=score_frame,
            scores_by_detector=scores,
            artifacts_dir=empty_dir,
            meta={"seed": 1234},
        ),
        attacked,
        score_frame,
    )


def test_payload_meta_and_alignment(payload):
    data, attacked, score_frame = payload
    assert data["meta"]["n_vessels"] == attacked["mmsi"].nunique()
    assert data["meta"]["n_windows"] == attacked["window_id"].nunique()
    n_scored = score_frame["window_id"].nunique()
    assert data["meta"]["n_scored"] == n_scored
    for strip in data["strips"]:
        assert len(strip["rows"]) == n_scored
        # labels in the strip match the canonical ordering
        assert [r[1] for r in strip["rows"]] == window_labels(score_frame).tolist()


def test_payload_tracks_have_attack_segments(payload):
    data, attacked, _ = payload
    assert len(data["tracks"]) == attacked["mmsi"].nunique()
    seg_types = {s["type"] for t in data["tracks"] for s in t["segments"]}
    assert seg_types  # gross attacks: at least one family drew a segment
    assert seg_types <= set(data["attack_colors"])
    for t in data["tracks"]:
        assert len(t["clean"]) > 2
        for s in t["segments"]:
            assert len(s["pts"]) >= 2


def test_payload_missing_artifacts_degrade_gracefully(payload):
    data, _, _ = payload
    assert data["sweeps"] == []
    assert data["latency"] == []


def test_payload_scores_normalised(payload):
    data, _, _ = payload
    for strip in data["strips"]:
        vals = np.array([r[0] for r in strip["rows"]])
        assert vals.min() >= 0.0 and vals.max() <= 1.0


def test_render_is_self_contained(payload, tmp_path):
    data, _, _ = payload
    html = render_dashboard(data)
    # the payload survives the round trip
    m = re.search(r"const D = (\{.*?\});\nconst NS", html, re.S)
    assert m and json.loads(m.group(1))["meta"]["seed"] == 1234
    # no external resources: offline means offline
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html
    out = write_dashboard(data, tmp_path / "dash.html")
    assert out.exists() and out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_sweep_and_latency_panels_read_artifacts(payload, windows, tmp_path):
    import pandas as pd

    data, attacked, score_frame = payload
    art = tmp_path
    pd.DataFrame(
        {
            "attack_type": ["position_jump"] * 4,
            "param": ["jump_km"] * 4,
            "value": [0.01, 0.1, 0.01, 0.1],
            "detector": ["A", "A", "B", "B"],
            "pr_auc": [0.2, 0.9, 0.3, 0.95],
        }
    ).to_csv(art / "robustness.csv", index=False)
    pd.DataFrame(
        {
            "detector": ["A", "A"],
            "window_id": [1, 2],
            "mmsi": [1, 1],
            "attack_type": ["position_jump", "position_jump"],
            "onset_idx": [3, 4],
            "detected": [True, False],
            "latency_points": [1.0, float("nan")],
            "latency_s": [60.0, float("nan")],
        }
    ).to_csv(art / "latency.csv", index=False)

    fresh = build_payload(
        clean_windows=windows,
        attacked=attacked,
        score_frame=score_frame,
        scores_by_detector={"KinematicRule": np.zeros(data["meta"]["n_scored"])},
        artifacts_dir=art,
    )
    assert fresh["sweeps"][0]["param"] == "jump_km"
    assert {s["detector"] for s in fresh["sweeps"][0]["series"]} == {"A", "B"}
    lat = fresh["latency"][0]
    assert lat["detection_rate"] == 0.5
    # NaN-free: the JSON must be parseable by a browser
    json.loads(json.dumps(fresh, allow_nan=False))
