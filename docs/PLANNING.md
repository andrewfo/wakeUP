# wakeUp — Detailed Planning Document

**Goal.** Detect spoofed / anomalous AIS vessel tracks — position jumps,
kinematically impossible motion, identity swaps, replay, and gradual drift —
using a sequence model backed by kinematic-consistency features. The
deliverable is a *reproducible benchmark* plus paper-ready results targeting
IEEE OCEANS / IEEE Access.

**Stack.** Python, pandas/Polars, PostGIS (optional), PyTorch, scikit-learn,
PyTorch Geometric (temporal GNN).

**Status legend:** ✅ done · 🔨 in progress · ⬜ planned

---

## Design principles

1. **The attack generator is the eval backbone.** Every reported number is
   only as trustworthy as the labels. The injector module
   (`src/wakeUp/attacks/`) is parameterised, deterministic, and unit-tested
   harder than anything else in the repo.
2. **Source-agnostic pipeline.** Everything after ingest operates on one
   canonical schema (`mmsi, timestamp, lat, lon, sog, cog, heading`), so
   synthetic and real AIS flow through identical code.
3. **Reproducible offline.** A physics-based synthetic fleet generator lets the
   full benchmark run with no network access; real MarineCadastre / DMA data
   drops in through a thin adapter.
4. **Determinism everywhere.** One global seed helper seeds Python, NumPy and
   PyTorch; every injector takes an explicit `rng`.
5. **Honest evaluation.** Per-attack-type breakdowns (each scored vs clean
   windows only), robustness sweeps over attack subtlety, and ablations
   separating hand-built features from learned representations.

---

## Phase 0 — Scaffold ✅

- [x] `src/wakeUp/` package (data / attacks / features / models / eval).
- [x] `configs/` (YAML today; Hydra structured configs planned — dataclass
      config loader already mirrors the Hydra group layout).
- [x] `tests/`, `notebooks/`, `scripts/`.
- [x] `Makefile` targets: `install`, `data`, `features`, `train`, `eval`,
      `figures`, `milestone`, `milestone-jump`, `robustness`, `lstm`,
      `transformer`, `test`, `lint`, `clean`.
- [x] Deterministic seeds (`config.set_global_seed`).
- [ ] Logging: no run logger yet — results land as `results.json` /
      `robustness.csv` artefacts. CSV run logger, then W&B behind an optional
      flag. ⬜
- **Deps:** `pyproject.toml` with a minimal core and `[learned]` (torch),
  `[gnn]` (torch-geometric, Phase 4 stretch only), `[config]`, `[dev]` extras
  so the milestone installs light. The core path — synthetic data, attacks,
  features, rule + IsolationForest, figures — needs no torch.

## Phase 1 — Data ✅ (synthetic) / 🔨 (real ingest)

- [x] Canonical schema + MarineCadastre CSV adapter
      (`data/pipeline.load_marinecadastre_csv`).
- [x] Physics-based synthetic fleet generator with bounded turn-rate /
      acceleration and self-consistent SOG/COG (`data/synthetic_ais.py`).
- [x] **Clean:** validity filter, dedup on `(mmsi, timestamp)`, drop
      `< min_track_points` tracks.
- [x] **Resample:** per-MMSI onto fixed cadence; circular interpolation for
      COG (unwrap → interp → rewrap); `gap_s` preserves true sensor gaps.
- [x] **Window:** fixed-length overlapping windows → per-point frame with
      `window_id` / `point_idx`. Stored as parquet.
- [ ] Ingest one real region (MarineCadastre zone or Danish Maritime
      Authority) end-to-end and re-run the benchmark. ⬜
- [ ] Optional PostGIS load for spatial neighbour queries (Phase 4 GNN). ⬜

## Phase 2 — Synthetic attack generator ✅

Each injector is parameterised + toggleable, emits per-point **and** per-window
labels, and has a `forge_velocity` switch for the harder variant where the
spoofer also fakes a self-consistent SOG/COG.

| Attack | Mechanism | Primary detection signal |
|---|---|---|
| `position_jump` | teleport a contiguous span by `jump_km` | Δposition ≫ reported SOG at boundary |
| `kinematic_impossible` | inflate step spacing ×`speed_multiplier` | implied speed / accel over physical limit |
| `identity_swap` | splice a donor vessel's motion under same MMSI | dynamics discontinuity mid-window |
| `replay` | paste an earlier same-track segment | position back-jump + repeated geometry |
| `gradual_drift` | ramp a 0→`drift_total_km` offset | slow divergence; SOG/COG stay ~consistent (subtle) |

- [x] `build_attacked_dataset` corrupts a `contamination` fraction, assigns
      window labels, handles donor selection for identity swap.
- [x] Unit tests assert labels, physical detectability, shape preservation,
      determinism, and contamination rate (`tests/test_attacks.py`).

## Phase 3 — Features ✅

- [x] Per-point kinematics: implied speed (Δpos), acceleration, turn-rate,
      and the key **consistency residuals** — reported SOG vs implied speed,
      reported COG vs implied bearing — plus gap detection.
- [x] Fixed-length window feature vector (robust aggregates + fraction of
      points over each physical limit): 27 features, stable column order.
- [x] Sequence tensors `(window, point, channel)` normalised per-channel for
      the learned models (`features/sequences.py`: `SequenceTensorizer` with
      train-only fit/transform stats; 7 kinematic channels, tested for shape,
      alignment, finiteness, determinism, and no train/test leakage).

## Phase 4 — Models ✅ (rule / iforest / LSTM-AE / Transformer) / ⬜ (GNN stretch)

- [x] Kinematic-threshold rule detector (worst-violation ratio → ranking).
- [x] IsolationForest over the window feature matrix (scaled).
- [x] LSTM-autoencoder over the sequence tensors, reconstruction error as the
      anomaly score (`models/sequence_ae.py`). Accepts either the per-point
      windows frame (tensorising internally with train-only stats) or a
      prebuilt `(N, L, C)` array; CPU + explicit-`rng` batching so runs are
      bitwise reproducible. Opt-in via `--lstm`; needs the `learned` extra.
- [x] **Main:** Transformer encoder over track windows with **both** a
      reconstruction head and a classification head (`models/transformer.py`).
      Fixed sinusoidal positional encoding, `norm_first` pre-LN blocks,
      class-rebalanced BCE. `fit(X)` trains reconstruction only and stays
      unsupervised (so the robustness harness drives it unchanged);
      `fit(X, supervised=True)` adds the classification loss and **must** be
      scored on held-out vessels.
- [ ] **Stretch:** temporal GNN over co-located vessels for spatial context. ⬜

All detectors expose `fit` / `score` where larger score == more anomalous, so
the eval harness treats them uniformly. Two opt-in class attributes let the
harnesses adapt without special-casing: `consumes_windows` (wants the
per-point frame, not the aggregated feature matrix) and `supports_supervision`
(can train with labels when the harness runs a held-out protocol).

## Phase 5 — Eval 🔨

- [x] Per-attack-type PR-AUC / ROC-AUC and FPR at fixed recall (each attack
      scored against clean windows only).
- [x] Overall metrics + JSON results dump.
- [x] Robustness curves: sweep `jump_km`, `speed_multiplier`,
      `drift_total_km` → degradation curves (`eval/robustness.py`:
      `sweep_attack_severity` / `run_robustness_sweeps`, plotted by
      `plot_robustness_curves`, run via `--robustness` / `make robustness`).
      Same windows corrupted at every severity, so the curve isolates
      subtlety rather than sampling noise.
- [x] Held-out protocol for the supervised arm (`eval/splits.py`). Splits **by
      vessel**, not by window: `window_stride < window_len`, so consecutive
      windows share points and a window-level split would leak. `--transformer`
      runs every detector under one held-out protocol so the supervised column
      stays comparable; `sweep_attack_severity(holdout=True)` does the same
      inside the sweeps.
- [ ] Detection latency (points-from-onset to first alarm). ⬜
- [ ] Ablations: features-only vs learned vs hybrid. ⬜

## Phase 6 — Deliverable 🔨

- [x] `figures/` auto-generated: `pr_curves.png`, `score_hist_iforest.png`,
      `attack_example_jump.png`, and `robustness_curves.png` (log-x degradation
      panel per swept knob, written by `--robustness`).
- [x] README with reproduce-in-one-command.
- [ ] Paper skeleton: methods, benchmark table, robustness plots. ⬜

---

## First milestone (defensible slice) ✅

**Phases 0–2 + IsolationForest baseline + one attack type end-to-end, plotted.**

Delivered and reproducible via `python scripts/run_milestone.py`
(`--single-attack position_jump` for the strict single-attack slice). Produces
the metrics table, PR curves, score histogram, and the worked position-jump
example figure. The full five-attack run is included as well.

**Representative result (synthetic, seed 1234).** On clean self-consistent
tracks the rule baseline already saturates on gross violations
(jump / kinematic / replay ≈ 1.00 PR-AUC) but degrades on the subtle attacks
(gradual drift ≈ 0.64, identity swap ≈ 0.77), while IsolationForest recovers
them (≈ 0.97 each). That gap is the intended story and the motivation for the
learned sequence models in Phase 4 — and the robustness sweeps in Phase 5 are
what make the benchmark discriminative rather than saturated.

**LSTM-autoencoder result (synthetic, seed 1234, `--lstm`) — a negative one.**
Per-attack PR-AUC / ROC-AUC: kinematic_impossible 1.00 / 1.00, identity_swap
0.68 / 0.95, replay 0.66 / 0.99, position_jump 0.35 / 0.93, gradual_drift
0.04 / 0.58 (overall 0.74 / 0.89). It is **beaten by IsolationForest on every
attack**, and on gradual drift it is at chance.

This is a property of the objective, not an untrained model: the loss
converges (0.81 → 0.50), and sweeping capacity from `lstm_hidden` 8 to 64 and
epochs 40 to 150 leaves drift PR-AUC pinned at 0.04–0.05 while *more* capacity
makes it slightly worse (0.049 → 0.039) — the textbook autoencoder-for-anomaly-
detection failure, where the model learns to reconstruct the anomalies too.

The per-attack pattern is the informative part: reconstruction error catches
**discontinuities** (kinematic_impossible saturates, replay/identity_swap reach
0.99/0.95 ROC) and is blind to **smooth global offsets** — a gradual drift
perturbs every channel by a near-constant amount, which a sequence model
reconstructs exactly as easily as clean motion. Aggregate features see that
offset as an obvious shift in `speed_resid_mean`; a reconstruction objective
cannot. That argues the Transformer should carry a *classification* head
alongside reconstruction, and it is the concrete motivation for the
features-vs-learned-vs-hybrid ablation.

**Transformer result (synthetic, seed 1234, `--transformer --robustness`).**
Trained on 28 vessels (1232 windows), scored on 12 unseen vessels (528
windows, 72 positives). At the default severities it reaches **1.00 PR-AUC on
all five attacks** — including `gradual_drift`, where the LSTM-AE was at
chance (0.04). The classification head is exactly the fix the AE result
predicted: reconstruction alone cannot see a smooth global offset, a
discriminative head can.

But IsolationForest *also* scores 1.00 on every attack at those settings, so
the headline table cannot separate them. The severity sweep can (held-out
PR-AUC, all detectors on the same split):

| knob | value | KinematicRule | IsolationForest | Transformer |
|---|---|---|---|---|
| `jump_km` | 0.001 | 0.13 | 0.17 | 0.13 |
| | 0.005 | 0.14 | 0.27 | **0.74** |
| | 0.010 | 0.15 | 0.56 | **0.95** |
| | 0.050 | 0.25 | 0.93 | **1.00** |
| `speed_multiplier` | 1.01 | 0.15 | 0.25 | **0.98** |
| | 1.10 | 0.26 | 0.67 | **1.00** |
| `drift_total_km` | 0.02 | 0.12 | 0.19 | **0.30** |
| | 0.05 | 0.13 | 0.27 | **0.81** |
| | 0.10 | 0.14 | 0.58 | **0.99** |

The Transformer moves the detection knee roughly **one order of magnitude
subtler** on every family, and on `kinematic_impossible` it is essentially
saturated (0.98) at a 1.01× speed inflation that the physics rule cannot see at
all. It only collapses to the others at `jump_km` 0.001 — a 1 m displacement,
below any physically meaningful spoof.

**The comparison is not apples-to-apples, and should not be reported as one.**
The Transformer is *supervised*: it sees attack labels during training, which
the rule and IsolationForest never do. The honest claim is "supervision buys
about a decade of attack subtlety on this benchmark", not "the Transformer is
a better anomaly detector". The features-vs-learned-vs-hybrid ablation is what
would decompose that gain, and an unsupervised Transformer arm
(`fit()` without `supervised=True`) is the matched control.

**Robustness result (synthetic, seed 1234, `--robustness`).** PR-AUC vs attack
severity, base rate 0.15. The headline settings sit far into the saturated
regime; the informative region is one to three orders of magnitude subtler:

| knob | detector | subtle end → gross end |
|---|---|---|
| `jump_km` 0.001 → 8.0 | KinematicRule | 0.18 → 1.00 (knee ≈ 0.1 km) |
| `jump_km` 0.001 → 8.0 | IsolationForest | 0.18 → 0.99 (knee ≈ 0.01 km) |
| `speed_multiplier` 1.01 → 6.0 | KinematicRule | 0.20 → 1.00 (knee ≈ 1.25×) |
| `speed_multiplier` 1.01 → 6.0 | IsolationForest | 0.28 → 1.00 (knee ≈ 1.1×) |
| `drift_total_km` 0.02 → 3.0 | KinematicRule | 0.17 → 0.77 |
| `drift_total_km` 0.02 → 3.0 | IsolationForest | 0.19 → 0.92 |

Two things the flat table hid. (a) IsolationForest leads the rule by roughly a
decade of severity on `position_jump` and dominates across the whole
`gradual_drift` ladder — the learned-vs-physics gap is a *shift in the
detection threshold*, not a fixed offset. (b) On `kinematic_impossible` the
curves cross near 1.5×: the physics rule is strictly better once the violation
is gross, since that is exactly what it tests. Because the synthetic fleet is
perfectly self-consistent, these knees are optimistic; real AIS position noise
should push them right, and quantifying that shift is the point of the Phase 1
real-data ingest.

## Next actions

1. **Ablation: features-only vs learned vs hybrid** (Phase 5), including an
   *unsupervised* Transformer arm. This is now the highest-value step: the
   supervised Transformer's decade-of-subtlety gain is currently confounded
   with its access to labels, and only the ablation separates "sequence model"
   from "supervision".
2. Detection latency: points-from-onset to first alarm (Phase 5). Pure
   numpy over the existing per-point `is_attack` labels, so unblocked.
3. Real MarineCadastre region ingest and re-run, including the robustness
   sweeps, to see how far sensor noise moves the knees (Phase 1).
4. **Stretch:** temporal GNN over co-located vessels (Phase 4). Needs the
   `gnn` extra and probably the PostGIS neighbour queries from Phase 1.
