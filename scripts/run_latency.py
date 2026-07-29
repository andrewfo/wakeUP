"""Detection-latency runner (Phase 5).

Replays held-out attacked windows as streams and reports, per detector and
attack family, the detection rate at a fixed clean-window false-positive rate
and the median points-from-onset to first alarm. Complements PR-AUC: the
benchmark table says whether a detector separates attacks, this says how long
the spoofer runs before the alarm.

Run:
    PYTHONPATH=src python scripts/run_latency.py                # rule + iforest + logistic
    PYTHONPATH=src python scripts/run_latency.py --transformer  # + Transformer ([learned])

Detectors are fit once on full-length train-vessel windows (supervised where
supported), then only scored on prefixes — the offline-train / streaming-score
deployment shape. Everything is seeded and reproducible.
"""

from __future__ import annotations

import argparse

from wakeUp.attacks import build_attacked_dataset
from wakeUp.config import REPO_ROOT, load_config, set_global_seed
from wakeUp.data import build_dataset, generate_fleet
from wakeUp.eval import run_detection_latency, summarize_latency
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
    ap.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="clean-window false-positive rate the alarm threshold is calibrated to",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=8,
        help="shortest prefix scored (Δ-based features need a few points)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.data.seed)
    proc_dir = REPO_ROOT / cfg.paths["processed"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] generating synthetic fleet -> windows ...")
    raw = generate_fleet(cfg.data)
    windows = build_dataset(raw, cfg.data)
    attacked = build_attacked_dataset(windows, cfg.attack)
    print(
        f"      windows={attacked['window_id'].nunique():,}  "
        f"vessels={attacked['mmsi'].nunique()}"
    )

    detectors = {
        "KinematicRule": lambda: KinematicRuleDetector(),
        "IsolationForest": lambda: IsolationForestDetector(cfg.model),
        "Logistic": lambda: LogisticFeatureDetector(cfg.model),
    }
    if args.transformer:
        from wakeUp.models import TransformerDetector

        detectors["Transformer"] = lambda: TransformerDetector(cfg.model)

    print(f"[2/3] streaming-prefix latency (held-out, FPR<={args.max_fpr:.0%}) ...")
    latency = run_detection_latency(
        attacked,
        detectors=detectors,
        min_points=args.min_points,
        max_fpr=args.max_fpr,
    )
    out = proc_dir / "latency.csv"
    latency.to_csv(out, index=False)

    print("[3/3] summary — detection rate & median points from onset to alarm:\n")
    summary = summarize_latency(latency)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nPer-window table -> {out}")


if __name__ == "__main__":
    main()
