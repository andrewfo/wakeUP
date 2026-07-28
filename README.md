# wakeUp

Detection of **spoofed / anomalous AIS vessel tracks** — position jumps,
kinematically impossible motion, identity swaps, replay, and gradual drift —
via kinematic-consistency features and learned sequence models. The repo is a
*reproducible benchmark*: a labeled synthetic attack generator, a
source-agnostic feature pipeline, four detectors, robustness sweeps over attack
subtlety, and auto-generated paper figures.

> Status: **Phases 0–4 delivered** — synthetic fleet, five labeled attacks,
> kinematic features + sequence tensors, and four detectors (physics rule,
> IsolationForest, LSTM autoencoder, Transformer), with per-attack metrics and
> degradation curves. Open: the features-vs-learned-vs-hybrid ablation,
> detection latency, and real-region ingest. See
> [`docs/PLANNING.md`](docs/PLANNING.md) for the phase-by-phase plan.

## Reproduce in one command

```bash
pip install -e ".[dev]"          # core + test deps (torch is an optional extra)
python scripts/run_milestone.py  # generate → attack → feature → baselines → figures
```

The defensible single-attack slice from the plan:

```bash
python scripts/run_milestone.py --single-attack position_jump
```

Degradation curves over attack subtlety, and the sequence models:

```bash
python scripts/run_milestone.py --robustness
pip install -e ".[learned]"                       # torch, for the two below
python scripts/run_milestone.py --lstm            # LSTM autoencoder
python scripts/run_milestone.py --transformer     # Transformer, held-out vessels
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
   detectors: KinematicRule · IsolationForest · LSTM-AE · Transformer
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

**The Transformer reaches 1.00 PR-AUC on all five attacks** on held-out vessels
(train on 28, score on 12 unseen) — including gradual drift, where the LSTM-AE
was at chance. Its classification head is exactly the fix the autoencoder
result predicted. IsolationForest also hits 1.00 at these settings, though, so
see the robustness section below for the comparison that actually separates
them.

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
tests for it directly wins.

Adding the supervised Transformer (`--transformer --robustness`, all detectors
on the same held-out split) moves the knee another order of magnitude subtler:

| knob | value | KinematicRule | IsolationForest | Transformer |
|---|---|---|---|---|
| `jump_km` | 0.005 | 0.14 | 0.27 | **0.74** |
| | 0.050 | 0.25 | 0.93 | **1.00** |
| `speed_multiplier` | 1.01 | 0.15 | 0.25 | **0.98** |
| `drift_total_km` | 0.05 | 0.13 | 0.27 | **0.81** |
| | 0.10 | 0.14 | 0.58 | **0.99** |

It only collapses to the others at a 1 m jump, below any meaningful spoof.
**This is not an apples-to-apples comparison** — the Transformer is supervised
and sees attack labels the other two never do, so the honest claim is
"supervision buys about a decade of subtlety here", not "the Transformer is a
better detector". Separating those is the next step (features vs learned vs
hybrid, with an unsupervised Transformer control).

Since the synthetic fleet is perfectly self-consistent, all these knees are
optimistic; real AIS noise should push them right, which is what the
real-region ingest is for.

## Layout

```
src/wakeUp/
  geo.py            great-circle geometry (haversine, bearing, destination)
  config.py         dataclass config + global seed helper
  data/             synthetic generator + clean/resample/window pipeline
  attacks/          5 labeled attack injectors (the eval backbone)
  features/         kinematic-consistency feature extraction
  models/           rule detector, IsolationForest, LSTM-AE, Transformer
  eval/             metrics, robustness sweeps, splits, figure generation
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
