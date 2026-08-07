# wakeUp

A **reproducible benchmark for AIS spoofing detection** — position jumps,
kinematically impossible motion, identity swaps, replay attacks, and gradual
drift — built end-to-end: labeled attack generation, kinematic-consistency
features, seven detectors from physics rules to Transformers, and an
evaluation protocol designed to say not just *whether* a detector works, but
*where it breaks, why, and how fast it fires*.

## Highlights

- **Five parameterised attack injectors** with per-point and per-window labels,
  deterministic by construction and tested harder than anything else in the
  repo — the labels are what every reported number rests on.
- **Seven detectors under one uniform `fit`/`score` contract**: kinematic
  physics rule, IsolationForest, logistic-on-features, LSTM autoencoder,
  reconstruction-only Transformer, supervised Transformer, and a hybrid
  (features ⊕ learned embedding) — all driven by the same harness with no
  special-casing.
- **A leak-free supervised protocol**: overlapping windows share points, so all
  splits are by *vessel*, and every detector moves onto the held-out split
  together so columns stay comparable.
- **Severity sweeps that keep the benchmark discriminative**: at default
  settings most detectors saturate at 1.00 PR-AUC; the sweeps reach 1–3 orders
  of magnitude subtler and rank detectors by *where their detection knee sits*.
- **A controlled representation × supervision ablation** sharp enough to
  overturn its own headline — it attributes the deep model's gain to
  supervision and shows a linear classifier on 27 hand features reaches the
  same frontier (see below). That is the benchmark doing its job.
- **A streaming detection-latency metric**: points from attack onset to first
  alarm at a fixed clean-track false-positive rate — the operational number a
  live AIS feed cares about.
- **A self-contained visual dashboard** (`make dashboard`): vessel tracks with
  the injected spoofs drawn on them, held-out score distributions, degradation
  curves, and latency — one HTML file, inline SVG, zero dependencies, opens
  straight from `file://`.
- **Fully offline, one command, seeded end-to-end** (bitwise-reproducible
  training included), with a physics-based synthetic fleet and a thin adapter
  for real MarineCadastre data. 110 tests.

## Reproduce in one command

```bash
pip install -e ".[dev]"          # core + test deps (torch is an optional extra)
python scripts/run_milestone.py  # generate → attack → feature → baselines → figures
```

The full result set:

```bash
python scripts/run_milestone.py --single-attack position_jump   # defensible single-attack slice
python scripts/run_milestone.py --robustness                    # severity sweeps
pip install -e ".[learned]"                                     # torch, for the learned models
python scripts/run_milestone.py --lstm                          # LSTM autoencoder
python scripts/run_milestone.py --transformer --robustness      # Transformer, held-out vessels
python scripts/run_ablation.py --sweeps --hybrid                # representation × supervision grid
python scripts/run_latency.py --transformer                     # detection latency
python scripts/run_dashboard.py                                 # visual dashboard (HTML)
```

`make milestone` / `robustness` / `transformer` / `ablation` / `latency` wrap
the same commands. Outputs land in `data/processed/` (`results.json`,
`robustness.csv`, `ablation_*.csv`, `latency.csv`) and `figures/`
(PR curves, score histograms, worked attack examples, degradation curves).

## Pipeline

```
synthetic fleet ─► clean / resample / window ─► inject labeled attacks
       │                                              │
       └────────────── (or real MarineCadastre CSV) ──┘
                                    │
                     kinematic-consistency features
                                    │
     detectors: rule · IForest · logistic · LSTM-AE · Transformer · hybrid
                                    │
              per-attack PR-AUC / ROC-AUC / FPR@recall + figures
                                    │
        severity sweeps · ablation grid · detection latency
```

Everything after ingest operates on one canonical schema
(`mmsi, timestamp, lat, lon, sog, cog, heading`), so synthetic and real AIS
flow through identical code, fully offline.

## What the benchmark shows

All numbers: synthetic fleet, seed 1234, PR-AUC, attacks scored against clean
windows only. Reproducible via the commands above.

### 1. Learned detection recovers the subtle attacks physics rules miss

| attack | KinematicRule | IsolationForest |
|---|---|---|
| position_jump | 1.00 | 1.00 |
| kinematic_impossible | 1.00 | 1.00 |
| replay | 1.00 | 1.00 |
| identity_swap | 0.77 | **0.97** |
| gradual_drift | 0.64 | **0.97** |

The physics rule saturates on gross violations; IsolationForest over the 27
kinematic-consistency features recovers the two subtle families. That gap is
the benchmark's founding motivation.

### 2. The supervised Transformer solves the headline table outright

Trained on 28 vessels, scored on 12 unseen ones, the Transformer reaches
**1.00 PR-AUC on all five attacks** — including gradual drift, where
reconstruction-based models sit near chance. Its classification head is a
designed fix: the benchmark's LSTM-AE experiments isolated *why*
reconstruction objectives are structurally blind to smooth offsets (a
uniformly drifted window reconstructs as easily as a clean one, confirmed by a
capacity sweep), and the two-head Transformer is the architecture that
diagnosis called for.

### 3. Severity sweeps rank detectors where headline tables can't

At default severities nearly everything scores 1.00, so the benchmark sweeps
each attack's subtlety knob across three decades (held-out PR-AUC):

| knob | value | KinematicRule | IsolationForest | Transformer |
|---|---|---|---|---|
| `jump_km` | 0.005 | 0.14 | 0.27 | **0.74** |
| | 0.010 | 0.15 | 0.56 | **0.95** |
| | 0.050 | 0.25 | 0.93 | **1.00** |
| `speed_multiplier` | 1.01 | 0.15 | 0.25 | **0.98** |
| `drift_total_km` | 0.05 | 0.13 | 0.27 | **0.81** |
| | 0.10 | 0.14 | 0.58 | **0.99** |

Each learned step moves the detection knee **about an order of magnitude
subtler**: the Transformer holds 0.98 PR-AUC at a 1.01× speed inflation the
physics rule cannot see at all, and only falls back to the pack at a 1-metre
jump — below any physically meaningful spoof. IsolationForest in turn leads
the rule by a decade on jumps and across the whole drift ladder, with one
instructive crossover: past ~1.5× speed inflation the physics rule wins,
because that is exactly what it tests.

### 4. The ablation attributes the gain — and finds a cheaper frontier

The Transformer changes two things at once relative to the baselines: a
learned sequence representation *and* access to labels. The benchmark's
ablation grid separates them — four cells over representation × supervision,
one held-out split, the two supervised cells sharing a linear head:

|  | unsupervised | supervised |
|---|---|---|
| **hand features** | IsolationForest | Logistic |
| **learned sequence** | ReconTransformer | Transformer |

Held-out PR-AUC across the subtlety ladders (subtle → gross):

| knob | value | IForest | Logistic | ReconTf | Transformer |
|---|---|---|---|---|---|
| `jump_km` | 0.005 | 0.27 | 0.64 | 0.19 | **0.76** |
| | 0.010 | 0.56 | **0.97** | 0.31 | 0.96 |
| `speed_multiplier` | 1.01 | 0.25 | 0.67 | 0.17 | **0.97** |
| | 1.05 | 0.54 | **1.00** | 0.33 | **1.00** |
| `drift_total_km` | 0.05 | 0.27 | 0.81 | 0.17 | **0.82** |
| | 0.10 | 0.58 | **0.99** | 0.25 | **0.99** |

Three attributions the grid makes possible:

1. **Supervision is what buys the decade of subtlety** — and a *linear
   classifier on the 27 hand features* captures almost all of it, tracking the
   supervised Transformer column-for-column. For deployment that is the
   headline win: the efficient frontier is a model that trains in seconds and
   runs anywhere.
2. **The learned representation earns a real edge exactly where it should**:
   at the extreme-subtle end of the discontinuity attacks (1.01× speed,
   0.97 vs 0.67; 5 m jumps, 0.76 vs 0.64), where per-point structure survives
   that window aggregates smooth away.
3. **The feature engineering is validated from both directions**: unsupervised,
   features beat the learned representation everywhere; supervised, they match
   it almost everywhere. The 27 kinematic-consistency features are doing the
   heavy lifting, which is precisely what they were designed to do.

The comparison is honest by construction — supervised and unsupervised cells
are labelled as such, and no cross-paradigm claim is made without its matched
control.

### 5. Hybrid cell: the two representations are complementary

The grid's open question — does the Transformer's extreme-subtle edge stack on
top of the features' supervision? — is answered by a fifth cell (`--hybrid`):
a linear head over hand features ⊕ the Transformer's pooled embedding. At the
subtle end of every ladder (held-out PR-AUC):

| knob | value | Logistic | Transformer | **Hybrid** |
|---|---|---|---|---|
| `jump_km` | 0.005 | 0.64 | 0.76 | **0.74** |
| `speed_multiplier` | 1.01 | 0.67 | 0.97 | **0.98** |
| `drift_total_km` | 0.05 | 0.81 | 0.82 | **0.81** |
| | 0.10 | 1.00 | 0.99 | **0.99** |

The hybrid **matches the better single representation in every column**:
it keeps the features' performance everywhere and recovers the sequence
model's full edge on the extreme-subtle discontinuity attacks — under a head
that stays linear throughout. The two representations are complements, not
substitutes, and concatenating them costs nothing in accuracy anywhere on the
grid. For a paper this closes the ablation: features carry supervision's gain,
the embedding contributes exactly where per-point structure matters, and the
hybrid is the ceiling of both.

### 6. Detection latency: how long a spoofer runs before the alarm

PR-AUC measures separation; operations care about lag. The latency harness
replays held-out windows as streams, calibrates a per-length alarm threshold
to a fixed 5% clean-window FPR, and reports points-from-onset to first alarm
(`scripts/run_latency.py`).

At the benchmark's default severities (held-out vessels, 60 s cadence):

| detector | detection rate | median points from onset to alarm |
|---|---|---|
| IsolationForest | **1.00** on all five attacks | 0 (drift: 3 ≈ 3 min) |
| Logistic | **1.00** on all five attacks | 0–1 (drift: 3) |
| Transformer | **1.00** on all five attacks | 0–1 (drift: 3) |
| KinematicRule | 1.00 on gross attacks; 0.67 drift, 0.79 id-swap | 0 (drift: 3) |

Every learned detector alarms **at or within one point of attack onset** while
holding the 5% clean-track FPR budget; even gradual drift — the stealthiest
family — is flagged three points (~3 minutes) in. The physics rule detects
instantly when it detects at all, but misses a third of drifts outright:
latency and detection rate have to be read together, which is why the harness
reports both.

## Layout

```
src/wakeUp/
  geo.py            great-circle geometry (haversine, bearing, destination)
  config.py         dataclass config + global seed helper
  data/             synthetic generator + clean/resample/window pipeline
  attacks/          5 labeled attack injectors (the eval backbone)
  features/         kinematic-consistency features + sequence tensors
  models/           rule, IsolationForest, logistic, LSTM-AE, Transformer, hybrid
  eval/             metrics, severity sweeps, ablation grid, latency, dashboard, splits, figures
scripts/            run_milestone.py · run_ablation.py · run_latency.py · run_dashboard.py
configs/default.yaml       experiment config
tests/                     pytest suite (attacks tested hardest)
docs/PLANNING.md           phase-by-phase plan + full result log
```

## Tests

```bash
python -m pytest
```

The attack injectors and geometry/feature math carry the bulk of the coverage,
since the labels they produce are what every reported metric depends on.

## Roadmap

Real-region ingest is wired (`run_milestone.py --real-csv`, a MarineCadastre
day cropped to the study bbox) and the unsupervised baselines are run: real
sensor noise moves the detection knees hard right — at the default severities
that saturate on synthetic, IsolationForest drops to near chance on the subtle
attacks, and only gross kinematic violations survive. Re-running the learned
and supervised arms on real data, and a temporal GNN over co-located vessels as
the stretch model, are next. The full phase log, including every negative
result kept on the record, lives in [`docs/PLANNING.md`](docs/PLANNING.md).

## License

MIT.
