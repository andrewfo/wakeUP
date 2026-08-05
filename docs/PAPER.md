# wakeUp: A Reproducible Benchmark for AIS Spoofing Detection

*Paper skeleton — target venue: IEEE OCEANS / IEEE Access.*

All numbers below are the repo's logged synthetic results (fleet seed 1234) as
recorded in [`PLANNING.md`](PLANNING.md); they are transcribed here as the
draft's placeholders, not produced fresh for this skeleton. Every table is
reproducible via the one-command entry points in the README. Cells marked
**[TODO]** are gated on the Phase 1 real-region ingest and are left empty
deliberately rather than filled with invented values.

---

## Abstract

*[~200 words. To finalise once the real-data column lands.]*

Automatic Identification System (AIS) tracks are trivially forgeable, and
spoofed or anomalous tracks — teleport-style position jumps, kinematically
impossible motion, identity swaps, replayed segments, and slow positional
drift — undermine any downstream maritime-domain-awareness system that trusts
them. Progress on detection is bottlenecked less by models than by *evaluation*:
public AIS lacks ground-truth spoofing labels, so reported numbers are hard to
trust or compare. We present **wakeUp**, a fully offline, deterministic
benchmark that generates labeled attacks over a physics-based synthetic fleet
(and drops real MarineCadastre data through the same pipeline), scores seven
detectors under one uniform contract, and — crucially — reports not just
whether a detector works but *where its detection knee sits, why, and how fast
it fires*. A controlled representation × supervision ablation attributes the
deep model's advantage to **supervision** rather than the learned sequence
representation, and shows a linear classifier on 27 kinematic-consistency
features reaches the same detection frontier. We release the generator,
detectors, and evaluation protocol as the primary contribution.

## 1. Introduction

- **Problem.** AIS is unauthenticated; spoofing is cheap and operationally
  consequential (illicit fishing, sanctions evasion, collision-avoidance
  poisoning). Detection must span gross violations *and* stealthy drift.
- **Gap.** No labeled public benchmark. Papers report metrics on
  incomparable, often saturated, private setups; the community cannot rank
  methods or locate where each breaks.
- **Contributions.**
  1. A parameterised, deterministic, unit-tested attack generator producing
     per-point and per-window labels over five attack families (§3).
  2. A source-agnostic pipeline on one canonical schema, so synthetic and real
     AIS flow through identical code (§3.1).
  3. An honest evaluation protocol: per-attack PR-AUC/ROC-AUC against clean
     windows only, leak-free by-vessel splits, severity sweeps, a
     representation × supervision ablation, and a streaming detection-latency
     metric (§5).
  4. A finding the benchmark's own ablation forces: the deep model's edge is
     **supervision**, and hand features capture almost all of it (§6.4).

## 2. Related work

*[Skeleton — to populate.]*

- AIS anomaly detection: rule/kinematic-consistency methods, trajectory
  clustering, autoencoders, sequence models.
- Spoofing/GPS-spoofing detection in the maritime and general GNSS literature.
- Benchmarking practice and the labeled-data gap that motivates a synthetic
  generator.

## 3. Benchmark design

### 3.1 Canonical pipeline

Everything after ingest operates on one schema —
`mmsi, timestamp, lat, lon, sog, cog, heading` — so synthetic and real AIS
share code. Stages: **clean** (validity filter, dedup on `(mmsi, timestamp)`,
short-track drop), **resample** (per-MMSI onto a fixed 60 s cadence, circular
interpolation for COG, `gap_s` preserving true sensor gaps), **window**
(length 32, stride 16 → 50% overlap, per-point frame with `window_id` /
`point_idx`). Two real-AIS options are off by default and documented as such:
gap-aware segmentation (`max_gap_s ≈ 600`), which stops resampling from
fabricating motion across a receiver dropout, and a study-area crop
(`crop_to_region`), which is *not* a no-op on the synthetic fleet (§7).

### 3.2 Synthetic fleet

Physics-based generator: bounded turn-rate and acceleration, self-consistent
SOG/COG, no position noise (so reported knees are optimistic — §7). Default
fleet: 40 vessels, 12 h, 60 s cadence, seed 1234.

### 3.3 Attack families

Each injector is parameterised, toggleable, deterministic (explicit `rng`), and
emits per-point and per-window labels, with a `forge_velocity` variant where
the spoofer also fakes self-consistent SOG/COG.

| Attack | Mechanism | Primary detection signal |
|---|---|---|
| `position_jump` | teleport a contiguous span by `jump_km` | Δposition ≫ reported SOG at boundary |
| `kinematic_impossible` | inflate step spacing ×`speed_multiplier` | implied speed / accel over physical limit |
| `identity_swap` | splice a donor vessel's motion under same MMSI | dynamics discontinuity mid-window |
| `replay` | paste an earlier same-track segment | position back-jump + repeated geometry |
| `gradual_drift` | ramp a 0→`drift_total_km` offset | slow divergence; SOG/COG stay consistent |

## 4. Features and detectors

### 4.1 Kinematic-consistency features

Per-point: implied speed (Δpos), acceleration, turn-rate, and the key
**consistency residuals** — reported SOG vs implied speed, reported COG vs
implied bearing — plus gap detection. Aggregated to a 27-dimensional per-window
vector (robust aggregates + fraction of points over each physical limit,
stable column order). For the sequence models, seven kinematic channels are
tensorised `(window, point, channel)` and normalised per-channel with
train-only statistics.

### 4.2 Detectors (one `fit`/`score` contract, larger = more anomalous)

| Detector | Representation | Supervision |
|---|---|---|
| KinematicRule | worst-violation ratio | unsupervised |
| IsolationForest | 27 hand features | unsupervised |
| LogisticFeatureDetector | 27 hand features | supervised (linear) |
| LSTM-AE | sequence tensors | unsupervised (reconstruction) |
| ReconTransformer | sequence tensors | unsupervised (reconstruction) |
| Transformer | sequence tensors | supervised (recon + linear cls head) |
| Hybrid | features ⊕ pooled encoder embedding | supervised (linear) |

Transformer: `d_model=64`, 4 heads, 2 pre-LN (`norm_first`) layers,
`dim_feedforward=128`, fixed sinusoidal positional encoding, class-rebalanced
BCE; CPU + explicit-`rng` batching for bitwise reproducibility.

## 5. Evaluation protocol

- **Per-attack.** PR-AUC / ROC-AUC / FPR@recall, each attack scored against
  clean windows only (so one family's difficulty isn't diluted by others).
- **Leak-free splits.** By vessel, never by window: `stride < window_len`, so a
  window-level split leaks near-duplicate windows across the boundary.
- **Held-out supervised protocol.** All detectors move onto one by-vessel split
  together so the supervised column stays comparable.
- **Severity sweeps.** Same windows corrupted at every severity, isolating
  subtlety rather than sampling noise; ladders reach 1–3 orders of magnitude
  subtler than the (saturated) defaults.
- **Ablation grid.** Representation × supervision, 2×2, the two supervised
  cells sharing a linear head so the classifier is held fixed.
- **Detection latency.** Held-out windows replayed as streaming prefixes;
  per-length threshold recalibrated to a fixed clean-window FPR; points from
  onset to first alarm, with pre-onset alarms counted as false positives.

## 6. Results (synthetic, seed 1234)

### 6.1 Learned detection recovers the subtle attacks physics rules miss

| attack | KinematicRule | IsolationForest |
|---|---|---|
| position_jump | 1.00 | 1.00 |
| kinematic_impossible | 1.00 | 1.00 |
| replay | 1.00 | 1.00 |
| identity_swap | 0.77 | **0.97** |
| gradual_drift | 0.64 | **0.97** |

### 6.2 Severity sweeps rank detectors where headline tables can't

| knob | value | KinematicRule | IsolationForest | Transformer |
|---|---|---|---|---|
| `jump_km` | 0.005 | 0.14 | 0.27 | **0.74** |
| | 0.010 | 0.15 | 0.56 | **0.95** |
| | 0.050 | 0.25 | 0.93 | **1.00** |
| `speed_multiplier` | 1.01 | 0.15 | 0.25 | **0.98** |
| `drift_total_km` | 0.05 | 0.13 | 0.27 | **0.81** |
| | 0.10 | 0.14 | 0.58 | **0.99** |

### 6.3 The LSTM-AE negative result (kept on the record)

Reconstruction error catches discontinuities but is structurally blind to
smooth global offsets: gradual_drift 0.04 PR-AUC (at chance), pinned at
0.04–0.05 across a capacity sweep (`lstm_hidden` 8→64, epochs 40→150), while
kinematic_impossible saturates. This is *why* the Transformer carries a
classification head alongside reconstruction.

### 6.4 Ablation — the deep model's gain is supervision, not representation

| knob | value | IForest | Logistic | ReconTf | Transformer |
|---|---|---|---|---|---|
| `jump_km` | 0.005 | 0.27 | 0.64 | 0.19 | **0.76** |
| | 0.010 | 0.56 | **0.97** | 0.31 | 0.96 |
| `speed_multiplier` | 1.01 | 0.25 | 0.67 | 0.17 | **0.97** |
| | 1.05 | 0.54 | **1.00** | 0.33 | **1.00** |
| `drift_total_km` | 0.05 | 0.27 | 0.81 | 0.17 | **0.82** |
| | 0.10 | 0.58 | **0.99** | 0.25 | **0.99** |

1. **Supervision** buys ~a decade of attack subtlety; a linear classifier on
   the 27 hand features captures almost all of it.
2. The learned representation's marginal value is small and localised — the
   extreme-subtle end of the *discontinuity* attacks (1.01× speed, 5 m jump).
3. Unsupervised, the learned arm (ReconTf) is the worst cell everywhere.

### 6.5 Hybrid — the representations are complementary

| knob | value | Logistic | Transformer | Hybrid |
|---|---|---|---|---|
| `jump_km` | 0.005 | 0.64 | 0.76 | 0.74 |
| `speed_multiplier` | 1.01 | 0.67 | 0.97 | **0.98** |
| `drift_total_km` | 0.05 | 0.81 | 0.82 | 0.81 |

The hybrid matches the better single representation in every column: features
carry supervision's gain, the embedding contributes exactly where per-point
structure matters, at no cost anywhere.

### 6.6 Detection latency (5% clean-window FPR, 60 s cadence)

| detector | detection rate | median points onset→alarm |
|---|---|---|
| IsolationForest | 1.00 on all five | 0 (drift: 3 ≈ 3 min) |
| Logistic | 1.00 on all five | 0–1 (drift: 3) |
| Transformer | 1.00 on all five | 0–1 (drift: 3) |
| KinematicRule | 1.00 gross; 0.67 drift, 0.79 id-swap | 0 (drift: 3) |

### 6.7 Real MarineCadastre region — **[TODO]**

Gated on the Phase 1 ingest. The claim to test: sensor noise pushes every knee
right, and it is where the features-vs-learned gap may widen.

## 7. Threats to validity

- **Optimistic knees.** The synthetic fleet is perfectly self-consistent (no
  position noise), so all reported detection knees are lower bounds on real
  difficulty. Quantifying the shift is the purpose of §6.7.
- **Study-area crop is not a no-op.** Synthetic vessels start inside the bbox
  and integrate out, so enabling the crop keeps 41.8% of fixes / 46.4% of
  windows and drops 4/40 vessels — every recorded number here is
  `crop_to_region=False`. Crop and gap-split belong on together for real AIS.
- **Supervised comparison is labelled as such.** No cross-paradigm claim is
  made without its matched control; "the Transformer is a better anomaly
  detector" is *not* a claim we make.

## 8. Conclusion

wakeUp is a fully offline, deterministic, honestly-scored AIS spoofing
benchmark whose own ablation reframes the deep-learning headline. Roadmap:
real-region ingest to move the knees under sensor noise (§6.7), and a temporal
GNN over co-located vessels as the stretch model.

## Reproducibility

Every table above maps to one command (README "Reproduce in one command");
outputs land in `data/processed/` and `figures/`. 110 tests, attacks tested
hardest. Full phase log and every negative result kept on the record:
[`PLANNING.md`](PLANNING.md).

---

### Figure/table checklist for submission

- [ ] Fig 1 — pipeline diagram (have: README ASCII; needs vector version).
- [x] Fig 2 — worked attack example (`figures/attack_example_jump.png`).
- [x] Fig 3 — PR curves (`figures/pr_curves.png`).
- [x] Fig 4 — robustness/degradation curves (`figures/robustness_curves.png`).
- [x] Fig 5 — ablation curves (`figures/ablation_curves.png`).
- [ ] Table I — headline per-attack (§6.1). Table II — sweeps (§6.2).
      Table III — ablation grid (§6.4). Table IV — latency (§6.6).
- [ ] §2 Related work prose.
- [ ] §6.7 real-data column.
</content>
