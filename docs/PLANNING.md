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
      `transformer`, `ablation`, `latency`, `dashboard`, `pages`, `test`,
      `lint`, `clean`.
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
- [x] **Split on gaps** (`split_on_gaps` / `drop_short_segments`, config
      `max_gap_s`): break each track at silences longer than the threshold and
      resample/window per segment, so no resampled point and no window spans a
      real receiver dropout. Without it, resampling *fills in* a multi-hour
      silence with fabricated smooth motion which is then labelled clean — a
      false negative manufactured by the pipeline. `max_gap_s=None` (the
      default) disables splitting: the synthetic fleet is gap-free, so every
      recorded number below is unaffected, and the milestone reproduces
      exactly. Real ingest should set it (~600 s).
- [ ] Ingest one real region (MarineCadastre zone or Danish Maritime
      Authority) end-to-end and re-run the benchmark. ⬜ Two known blockers to
      clear first: (a) `clean_ais` unpacks `cfg.region_bbox` but never filters
      on it, so a whole-zone download would not be cropped — enabling that
      filter changes dataset composition, so it needs its own step; (b) real
      position noise, which is what the reported knees are optimistic about.
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

## Phase 5 — Eval ✅ (on synthetic; re-run on real data pending Phase 1)

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
- [x] Detection latency (`eval/latency.py`, `scripts/run_latency.py`,
      `make latency`): held-out windows replayed as streams; per prefix length
      the alarm threshold is recalibrated to a fixed clean-window FPR (score
      distributions shift as windows grow, so a single full-window threshold
      would let short prefixes alarm for free), and latency is
      points-from-onset to the first alarming prefix. Alarms on prefixes with
      no attacked points yet are false positives, not detections; misses stay
      in the table so detection rate and latency read together. Detectors fit
      once on full-length train windows, scored on prefixes — the
      offline-train / streaming-score deployment shape.
- [x] Ablation: features-only vs learned × unsupervised vs supervised, as a 2×2
      under one held-out split (`eval/ablation.py`, `scripts/run_ablation.py`).
      Decomposes the Transformer's gain into a learned-representation axis and a
      supervision axis; the two supervised cells share a linear head so the
      comparison holds the classifier fixed. `--hybrid` adds the fifth cell:
      `HybridDetector`, a logistic head over the 27 hand features ⊕ the
      Transformer's mean-pooled encoder embedding (the exact vector `cls_head`
      sees), encoder trained identically to the supervised Transformer cell —
      so any gain over Logistic is the appended embedding, any gain over the
      Transformer is the appended features, classifier linear throughout.

## Phase 6 — Deliverable 🔨

- [x] `figures/` auto-generated: `pr_curves.png`, `score_hist_iforest.png`,
      `attack_example_jump.png`, `robustness_curves.png` (log-x degradation
      panel per swept knob, written by `--robustness`), and
      `ablation_curves.png` (the same panel over the ablation grid, written by
      `run_ablation.py --sweeps`).
- [x] README with reproduce-in-one-command.
- [x] Visual dashboard (`eval/dashboard.py`, `scripts/run_dashboard.py`,
      `make dashboard`): one self-contained HTML file (inline SVG + vanilla
      JS, no network, opens from `file://`) with four panels — vessel tracks
      with spoof segments overlaid (click-to-focus), held-out score strips +
      per-attack PR-AUC table, severity-sweep curves, and latency. Track/score
      panels are computed fresh and held-out; sweep/latency panels read the
      `data/processed/*.csv` artifacts when present and degrade to hints when
      absent. Torch-free.
- [x] Published via GitHub Pages from `main:/docs` — `make pages` renders the
      dashboard and copies it to `docs/index.html`. Note the `dashboard` target
      does *not* pass `--transformer`, so the published page carries score
      strips for the rule / IsolationForest / Logistic cells only (its sweep
      and latency panels still show the Transformer, since those read the CSVs).
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

**Ablation result (synthetic, seed 1234, `run_ablation.py --sweeps`) — the one
that reframes the headline.** The 2×2 over representation × supervision, all
four cells on one held-out split. At the default (saturated) severities three of
the four cells hit 1.00 PR-AUC on every attack; only the *unsupervised learned*
cell lags (overall 0.75), so the grid has to be read on the sweeps.

Held-out PR-AUC across the subtlety ladders (subtle → gross):

| knob | value | IForest (feat/uns) | Logistic (feat/sup) | ReconTf (seq/uns) | Transformer (seq/sup) |
|---|---|---|---|---|---|
| `jump_km` | 0.005 | 0.27 | **0.64** | 0.19 | **0.76** |
| | 0.010 | 0.56 | **0.97** | 0.31 | **0.96** |
| | 0.050 | 0.93 | **1.00** | 0.72 | **1.00** |
| `speed_multiplier` | 1.01 | 0.25 | 0.67 | 0.17 | **0.97** |
| | 1.05 | 0.54 | **1.00** | 0.33 | **1.00** |
| `drift_total_km` | 0.05 | 0.27 | **0.81** | 0.17 | **0.82** |
| | 0.10 | 0.58 | **0.99** | 0.25 | **0.99** |

The decomposition it forces:

1. **The decade-of-subtlety gain is supervision, not the learned
   representation.** A *linear classifier on the 27 hand features* (Logistic)
   tracks the supervised Transformer almost column-for-column — jump, kinematic,
   and drift knees all land within noise of each other. The earlier
   "Transformer ≫ IsolationForest" sweep was a supervised model measured against
   unsupervised baselines; with the matched supervised-features control in the
   grid, most of that gap moves onto the *supervision* axis. The honest headline
   is now "**supervision** buys ~a decade of subtlety here, and hand features
   capture almost all of it" — the sequence model is not what earns it.
2. **The learned representation's marginal value is small and localised.**
   Holding supervision fixed (Logistic → Transformer), the encoder adds a real
   but narrow edge only at the extreme-subtle end of the *discontinuity* attacks
   — `speed_multiplier` 1.01× (0.97 vs 0.67) and `jump_km` 0.005 (0.76 vs 0.64),
   where per-point structure survives that the window aggregates smooth away. On
   `gradual_drift` the two are indistinguishable.
3. **Unsupervised, the learned arm is the worst cell everywhere.** ReconTf sits
   below unsupervised IsolationForest on every ladder and is near-blind to smooth
   offsets (drift 0.15 → 0.70 across its whole range), saturating only on
   `kinematic_impossible`. This reproduces the LSTM-AE blind spot exactly:
   reconstruction sees discontinuities and cannot see a uniform shift.

So the efficient frontier on this benchmark is **supervised hand features**, not
the Transformer — a cheaper model reaching the same knees. The Transformer earns
its place only where per-point detail matters at the very subtle end, and that
is the defensible, narrow claim to make for it. The open question the ablation
raises is whether a **hybrid** (features ⊕ pooled encoder embedding) recovers
that thin learned edge on top of the features' supervision without paying full
sequence-model cost — the next ablation slice. All knees are optimistic on the
self-consistent synthetic fleet; the Phase 1 real-data ingest is what tests
whether the features-vs-learned gap widens once positions carry sensor noise.

**Hybrid result (synthetic, seed 1234, `run_ablation.py --sweeps --hybrid`) —
the ablation's answer.** The fifth cell (logistic head on the 27 hand features
⊕ the supervised Transformer's mean-pooled embedding, one held-out split with
the rest of the grid) matches the **better** of the two supervised single-
representation cells on every rung of every ladder:

| knob | value | Logistic | Transformer | Hybrid |
|---|---|---|---|---|
| `jump_km` | 0.005 | 0.64 | 0.76 | 0.74 |
| | 0.010 | 0.97 | 0.96 | 0.96 |
| `speed_multiplier` | 1.01 | 0.67 | 0.97 | **0.98** |
| `drift_total_km` | 0.05 | 0.81 | 0.82 | 0.81 |
| | 0.10 | 1.00 | 0.99 | 0.99 |

So the representations are **complements, not substitutes**: the embedding
carries per-point discontinuity structure the window aggregates smooth away
(the 1.01× / 5 m regime), the features carry everything else, and a linear
head over the concatenation attains the ceiling of both with no loss anywhere.
Caveats: the hybrid still pays full encoder training cost (its win is
accuracy-dominance, not cheapness), and its edge over plain Logistic lives
only in the extreme-subtle discontinuity regime — at every other severity
supervised hand features alone remain the efficient frontier.

**Detection-latency result (synthetic, seed 1234, `run_latency.py
--transformer`, held-out vessels, 5% clean-window FPR, 60 s cadence).** Median
points from onset to first alarm / detection rate, per attack family:

| detector | jump | kinematic | replay | id-swap | drift |
|---|---|---|---|---|---|
| KinematicRule | 0 / 1.00 | 0 / 1.00 | 0 / 1.00 | 0 / **0.79** | 3 / **0.67** |
| IsolationForest | 0 / 1.00 | 0 / 1.00 | 0 / 1.00 | 0 / 1.00 | 3 / 1.00 |
| Logistic | 0 / 1.00 | 1 / 1.00 | 0 / 1.00 | 0 / 1.00 | 3 / 1.00 |
| Transformer | 0 / 1.00 | 1 / 1.00 | 0 / 1.00 | 0 / 1.00 | 3 / 1.00 |

At the (saturated) default severities every learned detector alarms at or
within one point of onset while holding the FPR budget, and even gradual drift
is caught 3 points (~3 min) in. The rule detects instantly *when it detects*
but misses a third of drifts and a fifth of identity swaps — the metric's
point is exactly that latency and detection rate must be read together. The
informative follow-up once real data lands: latency at *subtle* severities,
where the families should finally separate.

## Next actions

1. ~~**Ablation: features-only vs learned**, including an *unsupervised*
   Transformer arm.~~ **Done** — the 2×2 (`scripts/run_ablation.py`) separates
   the learned representation from supervision; results above.
2. ~~**Hybrid cell** (logistic over hand features ⊕ pooled encoder
   embedding).~~ **Done** — `--hybrid` / `HybridDetector`; results above.
3. ~~Detection latency: points-from-onset to first alarm (Phase 5).~~
   **Done** — `eval/latency.py` streaming-prefix harness; results above.
4. ~~Gap-aware track segmentation, so resampling cannot invent motion across a
   receiver dropout (Phase 1, real-ingest prerequisite).~~ **Done** —
   `split_on_gaps` / `max_gap_s`, off by default so synthetic results are
   unchanged (`tests/test_gap_segmentation.py` pins both the split and the
   no-op).
5. **Region-crop `clean_ais`.** It unpacks `cfg.region_bbox` and then ignores
   it, so a real whole-zone download would not be cropped. Small change, but it
   alters which points survive cleaning, so it lands as its own step with the
   synthetic impact measured before anything is re-reported. (Synthetic vessels
   integrate for 12 h and drift well outside the 1.5°×2.0° box, so switching
   this on *would* move the numbers — it is not a no-op like the gap split.)
6. Real MarineCadastre region ingest and re-run, including the robustness
   sweeps, the ablation, and latency-at-subtle-severities, to see how far
   sensor noise moves the knees (Phase 1). Now unblocked on the gap side;
   needs (5) and a chosen zone/date.
7. Paper skeleton: methods, benchmark table, robustness plots (Phase 6) — the
   last ⬜ before the deliverable, and draftable now on synthetic results.
8. **Stretch:** temporal GNN over co-located vessels (Phase 4). Needs the
   `gnn` extra and probably the PostGIS neighbour queries from Phase 1.
