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

Degradation curves over attack subtlety, and the sequence model:

```bash
python scripts/run_milestone.py --robustness
pip install -e ".[learned]" && python scripts/run_milestone.py --lstm
```

Outputs:
- `data/processed/results.json` — overall + per-attack metrics for each detector
- `data/processed/robustness.csv` — metrics vs attack severity (`--robustness`)
- `figures/pr_curves.png`, `figures/score_hist_iforest.png`,
  `figures/attack_example_jump.png`, `figures/robustness_curves.png`

`make milestone`, `make robustness`, `make test`, and `make figures` wrap the
same commands.

## What it does

```
synthetic fleet ─► clean / resample / window ─► inject labeled attacks
       │                                              │
       └────────────── (or real MarineCadastre CSV) ──┘
                                    │
                     kinematic-consistency features
                                    │
        detectors: KinematicRule · IsolationForest · LSTM-AE
                                    │
              per-attack PR-AUC / ROC-AUC / FPR@recall + figures
                                    │
                  robustness sweeps over attack subtlety
```

The benchmark runs fully **offline** on a physics-based synthetic fleet
(kinematically self-consistent tracks), and the *same* pipeline ingests real
open AIS through `data.pipeline.load_marinecadastre_csv`.

## Representative result (synthetic, seed 1234)

| attack | KinematicRule | IsolationForest | LSTM-AE |
|---|---|---|---|
| position_jump | 1.00 | 1.00 | 0.35 |
| kinematic_impossible | 1.00 | 1.00 | 1.00 |
| replay | 1.00 | 1.00 | 0.66 |
| identity_swap | 0.77 | 0.97 | 0.68 |
| gradual_drift | 0.64 | 0.97 | 0.04 |

(PR-AUC. LSTM-AE via `--lstm`, needs the `learned` extra.)

Physics rules saturate on gross violations but miss the subtle attacks; the
IsolationForest baseline recovers them.

**The LSTM autoencoder loses to both** — reported as-is rather than tuned away.
Its loss converges and the result is stable across `lstm_hidden` 8→64 and
40→150 epochs (more capacity makes drift slightly *worse*), so it is a property
of the objective: reconstruction error flags **discontinuities** but is blind to
**smooth global offsets**. A gradual drift shifts every channel by a
near-constant amount, which a sequence model reconstructs as easily as clean
motion, while the aggregate features see an obvious `speed_resid_mean` shift.
That is why the planned Transformer carries a classification head alongside
reconstruction.

### Robustness: where each detector actually breaks

The table above is one point on a curve. Sweeping attack severity
(`--robustness`) shows the headline settings sit deep in the saturated regime —
the informative region is one to three orders of magnitude subtler:

| knob (subtle → gross) | KinematicRule PR-AUC | IsolationForest PR-AUC |
|---|---|---|
| `jump_km` 0.001 → 8.0 | 0.18 → 1.00 | 0.18 → 0.99 |
| `speed_multiplier` 1.01 → 6.0 | 0.20 → 1.00 | 0.28 → 1.00 |
| `drift_total_km` 0.02 → 3.0 | 0.17 → 0.77 | 0.19 → 0.92 |

IsolationForest leads the physics rule by roughly a decade of severity on
position jumps and across the whole drift ladder, but the two curves *cross* on
`kinematic_impossible` near 1.5× — once a violation is gross, the rule that
tests for it directly wins. Since the synthetic fleet is perfectly
self-consistent, these knees are optimistic; real AIS noise should push them
right. Sequence models (LSTM-AE, Transformer) and real-region ingest are next.

## Layout

```
src/wakeUp/
  geo.py            great-circle geometry (haversine, bearing, destination)
  config.py         dataclass config + global seed helper
  data/             synthetic generator + clean/resample/window pipeline
  attacks/          5 labeled attack injectors (the eval backbone)
  features/         kinematic-consistency feature extraction
  models/           rule detector, IsolationForest, LSTM autoencoder
  eval/             metrics, robustness sweeps, figure generation
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
