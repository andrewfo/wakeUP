"""Features-vs-learned-vs-supervision ablation runner (Phase 5).

Decomposes the headline "supervised Transformer buys a decade of subtlety" claim
into its two confounded causes — the learned sequence representation and access
to labels — with a 2×2 grid scored under one held-out-by-vessel protocol:

                    unsupervised            supervised
    hand features   IForest                 Logistic
    learned         ReconTransformer        Transformer

Run:
    PYTHONPATH=src python scripts/run_ablation.py            # held-out table
    PYTHONPATH=src python scripts/run_ablation.py --sweeps   # + subtlety sweeps
    PYTHONPATH=src python scripts/run_ablation.py --lstm     # + LSTM-AE cell

The learned cells need the ``learned`` extra (torch). Everything is seeded, so
the tables and figures are reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wakeUp.config import REPO_ROOT, load_config, set_global_seed
from wakeUp.data import build_dataset, generate_fleet
from wakeUp.eval import (
    ablation_cells,
    plot_robustness_curves,
    run_ablation_sweeps,
    run_ablation_table,
)


def _pivot_print(df, index, columns, values, fmt="{:.3f}"):
    print(
        df.pivot(index=index, columns=columns, values=values).to_string(
            float_format=lambda v: fmt.format(v)
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument(
        "--sweeps",
        action="store_true",
        help="also trace the grid across attack subtlety (slower: retrains per severity)",
    )
    ap.add_argument(
        "--lstm",
        action="store_true",
        help="add the LSTM-AE as a second learned/unsupervised cell",
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

    print("[1/3] generating synthetic fleet -> windows ...")
    raw = generate_fleet(cfg.data)
    windows = build_dataset(raw, cfg.data)
    print(
        f"      windows={windows['window_id'].nunique():,}  "
        f"vessels={windows['mmsi'].nunique()}"
    )

    grid = ablation_cells(cfg.model, include_lstm=args.lstm)

    print("[2/3] held-out ablation table (all five attacks) ...")
    table = run_ablation_table(
        windows, attack_cfg=cfg.attack, model_cfg=cfg.model, cells=grid
    )
    table.to_csv(proc_dir / "ablation_table.csv", index=False)

    # 2x2 grid: PR-AUC by attack family, columns are the four (rep, sup) cells.
    per_atk = table[table["attack_type"] != "ALL"]
    print("\n=== held-out PR-AUC by attack family (rows) x cell (cols) ===")
    print("    representation × supervision:")
    axes = (
        table[["detector", "representation", "supervision"]]
        .drop_duplicates()
        .set_index("detector")
    )
    for det, row in axes.iterrows():
        print(f"      {det:<18} {row['representation']:<9} {row['supervision']}")
    print()
    _pivot_print(per_atk, index="attack_type", columns="detector", values="pr_auc")
    print("\n=== overall held-out metrics per cell ===")
    overall = table[table["attack_type"] == "ALL"].sort_values(
        ["representation", "supervision"]
    )
    print(
        overall[
            ["detector", "representation", "supervision", "pr_auc", "roc_auc"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    if args.sweeps:
        print("\n[3/3] ablation across attack subtlety (held-out sweeps) ...")
        sweep = run_ablation_sweeps(
            windows, attack_cfg=cfg.attack, model_cfg=cfg.model, cells=grid
        )
        sweep.to_csv(proc_dir / "ablation_sweeps.csv", index=False)
        plot_robustness_curves(sweep, fig_dir / "ablation_curves.png")
        for param, g in sweep.groupby("param", sort=False):
            print(f"\n=== {g['attack_type'].iloc[0]} ({param}) — PR-AUC ===")
            _pivot_print(g, index="value", columns="detector", values="pr_auc")
        print(f"\nFigure -> {fig_dir/'ablation_curves.png'}")
    else:
        print("\n(re-run with --sweeps for the subtlety decomposition)")

    print(f"\nDone. Tables -> {proc_dir}/ablation_*.csv")


if __name__ == "__main__":
    main()
