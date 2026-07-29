"""Dashboard builder (Phase 6).

Renders the whole benchmark into one self-contained HTML file — vessel tracks
with the injected spoofs, held-out detector scores, the severity-sweep
degradation curves, and detection latency — with zero runtime dependencies:
open ``figures/dashboard.html`` straight from the filesystem.

Run:
    PYTHONPATH=src python scripts/run_dashboard.py                # rule + iforest + logistic
    PYTHONPATH=src python scripts/run_dashboard.py --transformer  # + Transformer ([learned])

The sweep and latency panels read ``data/processed/{ablation_sweeps,
robustness,latency}.csv`` when present (run the ablation / robustness /
latency scripts first to populate them); the track and score panels are
computed fresh here, held-out by vessel, so the page always reflects the
current config and seed.
"""

from __future__ import annotations

import argparse

from wakeUp.attacks import build_attacked_dataset
from wakeUp.config import REPO_ROOT, load_config, set_global_seed
from wakeUp.data import build_dataset, generate_fleet
from wakeUp.eval.dashboard import build_payload, write_dashboard
from wakeUp.eval.splits import train_test_split_by_vessel
from wakeUp.features import build_feature_matrix
from wakeUp.models import (
    IsolationForestDetector,
    KinematicRuleDetector,
    LogisticFeatureDetector,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument(
        "--transformer",
        action="store_true",
        help="add the supervised Transformer (needs the [learned] extra)",
    )
    ap.add_argument("--out", default=None, help="output HTML path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.data.seed)
    out = (
        REPO_ROOT / cfg.paths["figures"] / "dashboard.html"
        if args.out is None
        else args.out
    )

    print("[1/3] generating synthetic fleet -> windows -> attacks ...")
    raw = generate_fleet(cfg.data)
    windows = build_dataset(raw, cfg.data)
    attacked = build_attacked_dataset(windows, cfg.attack)

    print("[2/3] fitting detectors, scoring held-out vessels ...")
    fit_frame, score_frame = train_test_split_by_vessel(attacked)
    fit_feat, _ = build_feature_matrix(fit_frame)
    score_feat, _ = build_feature_matrix(score_frame)

    detectors = {
        "KinematicRule": KinematicRuleDetector(),
        "IsolationForest": IsolationForestDetector(cfg.model),
        "Logistic": LogisticFeatureDetector(cfg.model),
    }
    if args.transformer:
        from wakeUp.models import TransformerDetector

        detectors["Transformer"] = TransformerDetector(cfg.model)

    scores = {}
    for name, det in detectors.items():
        seq = getattr(det, "consumes_windows", False)
        if getattr(det, "supports_supervision", False):
            det.fit(fit_frame if seq else fit_feat, supervised=True)
        else:
            det.fit(fit_frame if seq else fit_feat)
        scores[name] = det.score(score_frame if seq else score_feat)

    print("[3/3] rendering ...")
    payload = build_payload(
        clean_windows=windows,
        attacked=attacked,
        score_frame=score_frame,
        scores_by_detector=scores,
        artifacts_dir=REPO_ROOT / cfg.paths["processed"],
        meta={"seed": cfg.data.seed},
    )
    path = write_dashboard(payload, out)
    print(f"Dashboard -> {path}")


if __name__ == "__main__":
    main()
