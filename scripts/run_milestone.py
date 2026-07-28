"""First-milestone end-to-end runner (Phases 0–2 + IsolationForest baseline).

Pipeline:
    generate synthetic clean fleet
      -> clean / resample / window
      -> inject labeled attacks
      -> kinematic feature matrix
      -> KinematicRule + IsolationForest baselines
      -> metrics (overall + per-attack) + figures

Run:  python scripts/run_milestone.py
Everything is seeded, so results and figures are reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wakeUp.config import Config, load_config, set_global_seed, REPO_ROOT
from wakeUp.data import generate_fleet, build_dataset
from wakeUp.attacks import build_attacked_dataset, AttackType, inject_position_jump
from wakeUp.features import build_feature_matrix
from wakeUp.models import KinematicRuleDetector, IsolationForestDetector
from wakeUp.eval import (
    plot_pr_curves,
    plot_robustness_curves,
    plot_score_hist,
    plot_window_example,
)
from wakeUp.eval.metrics import per_attack_metrics, dominant_attack_type, evaluate_scores
from wakeUp.eval.robustness import default_detectors, run_robustness_sweeps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument(
        "--single-attack",
        default=None,
        choices=[a.value for a in AttackType if a is not AttackType.NONE],
        help="restrict to one attack type (defensible slice); default = all five",
    )
    ap.add_argument(
        "--robustness",
        action="store_true",
        help="also run the attack-subtlety sweeps (slower: rebuilds the dataset per severity)",
    )
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.data.seed)

    root = REPO_ROOT
    fig_dir = Path(args.outdir) if args.outdir else root / cfg.paths["figures"]
    proc_dir = root / cfg.paths["processed"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] generating synthetic fleet ...")
    raw = generate_fleet(cfg.data)
    print(f"      raw rows={len(raw):,}  vessels={raw['mmsi'].nunique()}")

    print("[2/6] clean -> resample -> window ...")
    windows = build_dataset(raw, cfg.data)
    n_windows = windows["window_id"].nunique()
    print(f"      windows={n_windows:,}  points/window={cfg.data.window_len}")

    print("[3/6] injecting labeled attacks ...")
    attack_types = None
    if args.single_attack:
        attack_types = [AttackType(args.single_attack)]
        print(f"      single-attack mode: {args.single_attack}")
    attacked = build_attacked_dataset(windows, cfg.attack, attack_types=attack_types)
    attacked.to_parquet(proc_dir / "windows_attacked.parquet", index=False)
    win_pos_rate = (
        attacked.groupby("window_id")["window_label"].first().mean()
    )
    print(f"      attacked-window rate={win_pos_rate:.3f}")

    print("[4/6] building kinematic feature matrix ...")
    feat_df, labels = build_feature_matrix(attacked)
    dom = dominant_attack_type(attacked).reindex(feat_df.index).to_numpy()
    feat_df.to_parquet(proc_dir / "features.parquet")
    print(f"      feature matrix={feat_df.shape}")

    print("[5/6] fitting baselines ...")
    rule = KinematicRuleDetector().fit(feat_df)
    iforest = IsolationForestDetector(cfg.model).fit(feat_df)
    scores = {
        "KinematicRule": rule.score(feat_df),
        "IsolationForest": iforest.score(feat_df),
    }

    print("[6/6] metrics + figures ...")
    results = {}
    for name, sc in scores.items():
        overall = evaluate_scores(labels, sc)
        per_atk = per_attack_metrics(sc, labels, dom)
        results[name] = {
            "overall": overall,
            "per_attack": per_atk.to_dict(orient="records"),
        }
        print(f"\n=== {name} ===")
        print(per_atk.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    (proc_dir / "results.json").write_text(json.dumps(results, indent=2))

    # figures
    plot_pr_curves(labels, scores, fig_dir / "pr_curves.png")
    plot_score_hist(labels, scores["IsolationForest"], fig_dir / "score_hist_iforest.png")

    # illustrative single-window example (position jump) for the paper
    example_wid = int(feat_df.index[0])
    clean_win = windows[windows["window_id"] == example_wid].copy()
    rng = np.random.default_rng(0)
    from wakeUp.config import AttackConfig

    atk_win = inject_position_jump(clean_win, rng, AttackConfig())
    plot_window_example(
        clean_win, atk_win, fig_dir / "attack_example_jump.png",
        title="Position-jump attack (clean vs spoofed)",
    )

    if args.robustness:
        print("\n[+] robustness sweeps over attack subtlety ...")
        sweep = run_robustness_sweeps(
            windows, attack_cfg=cfg.attack, detectors=default_detectors(cfg.model)
        )
        sweep.to_csv(proc_dir / "robustness.csv", index=False)
        plot_robustness_curves(sweep, fig_dir / "robustness_curves.png")
        for param, g in sweep.groupby("param", sort=False):
            print(f"\n=== {g['attack_type'].iloc[0]} ({param}) ===")
            print(
                g.pivot(index="value", columns="detector", values="pr_auc").to_string(
                    float_format=lambda v: f"{v:.3f}"
                )
            )

    print(f"\nDone. Results -> {proc_dir/'results.json'}  Figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
