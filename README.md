# wakeUp

Detection of **spoofed / anomalous AIS vessel tracks** — position jumps,
kinematically impossible motion, identity swaps, replay, and gradual drift —
via kinematic-consistency features and (progressively) learned sequence models.
The repo is a *reproducible benchmark*: a labeled synthetic attack generator,
a source-agnostic feature pipeline, baseline detectors, and auto-generated
paper figures.

> Status: **first milestone delivered** — Phases 0–2 + IsolationForest baseline
> + attacks end-to-end, plotted. See [`docs/PLANNING.md`](docs/PLANNING.md) for
> the full phase-by-phase plan and what's next.

## Reproduce in one command

```bash
pip install -e ".[dev]"          # core + test deps (torch is an optional extra)
python scripts/run_milestone.py  # generate → attack → feature → baselines → figures
```

The defensible single-attack slice from the plan:

```bash
python scripts/run_milestone.py --single-attack position_jump
```

Outputs:
- `data/processed/results.json` — overall + per-attack metrics for each detector
- `figures/pr_curves.png`, `figures/score_hist_iforest.png`,
  `figures/attack_example_jump.png`

`make milestone`, `make test`, and `make figures` wrap the same commands.

## What it does

```
synthetic fleet ─► clean / resample / window ─► inject labeled attacks
       │                                              │
       └────────────── (or real MarineCadastre CSV) ──┘
                                    │
                     kinematic-consistency features
                                    │
                 baselines: KinematicRule · IsolationForest
                                    │
              per-attack PR-AUC / ROC-AUC / FPR@recall + figures
```

The benchmark runs fully **offline** on a physics-based synthetic fleet
(kinematically self-consistent tracks), and the *same* pipeline ingests real
open AIS through `data.pipeline.load_marinecadastre_csv`.

## Representative result (synthetic, seed 1234)

| attack | KinematicRule PR-AUC | IsolationForest PR-AUC |
|---|---|---|
| position_jump | 1.00 | 1.00 |
| kinematic_impossible | 1.00 | 1.00 |
| replay | 1.00 | 1.00 |
| identity_swap | 0.77 | 0.97 |
| gradual_drift | 0.64 | 0.97 |

Physics rules saturate on gross violations but miss the subtle attacks; the
learned baseline recovers them. Closing that gap with sequence models
(LSTM-AE, Transformer) and stress-testing it with robustness sweeps over attack
subtlety is the next phase.

## Layout

```
src/wakeUp/
  geo.py            great-circle geometry (haversine, bearing, destination)
  config.py         dataclass config + global seed helper
  data/             synthetic generator + clean/resample/window pipeline
  attacks/          5 labeled attack injectors (the eval backbone)
  features/         kinematic-consistency feature extraction
  models/           baselines (rule detector, IsolationForest)
  eval/             metrics + figure generation
scripts/run_milestone.py   end-to-end runner
configs/default.yaml       experiment config
tests/                     pytest suite (attacks tested hardest)
docs/PLANNING.md           detailed phase-by-phase plan
```

## Tests

```bash
python -m pytest
```

The attack injectors and geometry/feature math carry the bulk of the coverage,
since the labels they produce are what every reported metric depends on.

## License

MIT.
