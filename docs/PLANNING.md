# wakeguard — Detailed Planning Document

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
   (`src/wakeguard/attacks/`) is parameterised, deterministic, and unit-tested
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

- [x] `src/wakeguard/` package (data / attacks / features / models / eval).
- [x] `configs/` (YAML today; Hydra structured configs planned — dataclass
      config loader already mirrors the Hydra group layout).
- [x] `tests/`, `notebooks/`, `scripts/`.
- [x] `Makefile` targets: `data`, `features`, `train`, `eval`, `figures`,
      `milestone`, `test`.
- [x] Deterministic seeds (`config.set_global_seed`).
- [ ] Logging: CSV logger now; W&B behind an optional flag. ⬜
- **Deps:** `pyproject.toml` with a minimal core and `[learned]`, `[config]`,
  `[dev]` extras so the milestone installs light.

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

## Phase 3 — Features ✅ (kinematic) / ⬜ (sequence tensors)

- [x] Per-point kinematics: implied speed (Δpos), acceleration, turn-rate,
      and the key **consistency residuals** — reported SOG vs implied speed,
      reported COG vs implied bearing — plus gap detection.
- [x] Fixed-length window feature vector (robust aggregates + fraction of
      points over each physical limit): 27 features, stable column order.
- [ ] Sequence tensors `(window, point, channel)` normalised per-channel for
      the learned models. ⬜

## Phase 4 — Models 🔨

- [x] Kinematic-threshold rule detector (worst-violation ratio → ranking).
- [x] IsolationForest over the window feature matrix (scaled).
- [ ] LSTM-autoencoder (reconstruction error as anomaly score). ⬜
- [ ] **Main:** Transformer encoder over track windows, reconstruction and
      classification heads. ⬜
- [ ] **Stretch:** temporal GNN over co-located vessels for spatial context. ⬜

All detectors expose `fit` / `score` where larger score == more anomalous, so
the eval harness treats them uniformly.

## Phase 5 — Eval 🔨

- [x] Per-attack-type PR-AUC / ROC-AUC and FPR at fixed recall (each attack
      scored against clean windows only).
- [x] Overall metrics + JSON results dump.
- [ ] Detection latency (points-from-onset to first alarm). ⬜
- [ ] Robustness curves: sweep `drift_total_km`, `jump_km`,
      `speed_multiplier` → degradation curves. ⬜
- [ ] Ablations: features-only vs learned vs hybrid. ⬜

## Phase 6 — Deliverable 🔨

- [x] `figures/` auto-generated (PR curves, score histogram, worked example).
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
what will make the benchmark discriminative rather than saturated.

## Next actions

1. Sequence tensors + LSTM-autoencoder (Phase 3/4).
2. Robustness sweep harness over attack subtlety (Phase 5).
3. Real MarineCadastre region ingest and re-run (Phase 1).
