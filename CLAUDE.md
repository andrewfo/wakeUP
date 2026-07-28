# CLAUDE.md — working notes for wakeUp

Detection of spoofed/anomalous AIS vessel tracks. The repo is a *benchmark*, so
the labels and the evaluation protocol matter more than any single model.

## Running things

The package is **not pip-installed** in the current dev environment — tests and
scripts need `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_milestone.py [--single-attack position_jump]
                                               [--lstm] [--transformer] [--robustness]
```

`make test` / `make milestone` / `make robustness` / `make lstm` /
`make transformer` wrap the same commands (and assume an installed package).

`--lstm` and `--transformer` need torch (`pip install -e ".[learned]"`). The
installed build is **CPU-only** — PyPI's default Windows wheel no longer bundles
CUDA. Everything else runs without torch; `wakeUp.models.__getattr__` imports
the torch-backed detectors lazily so `import wakeUp.models` never requires it.

The Transformer severity sweep is slow (~15 min on CPU: 21 severities × 3
detectors, retraining each time). Run it in the background.

## Canonical schema

Everything after ingest is `mmsi, timestamp, lat, lon, sog, cog, heading`
(+ `gap_s` after resampling, + `window_id` / `point_idx` after windowing,
+ `is_attack` / `attack_type` / `window_label` after injection). Synthetic and
real AIS flow through identical code; real data enters via
`data.pipeline.load_marinecadastre_csv`.

## Invariants — do not break these

**Detector contract.** `fit` / `score` where **larger score == more anomalous**.
The eval and robustness harnesses depend on this being uniform; sklearn's
`score_samples` is higher-for-inliers, hence the negation in
`IsolationForestDetector.score`.

Two opt-in class attributes let harnesses adapt without special-casing names:
- `consumes_windows = True` — hand it the per-point windows frame, not the
  aggregated 27-column feature matrix (the sequence models).
- `supports_supervision = True` — it can train with labels when the harness
  runs a held-out protocol (the Transformer only).

**Determinism.** Every injector takes an explicit `rng`; models seed from
`cfg.seed` and batch via a numpy permutation rather than a DataLoader.
`config.set_global_seed` covers Python/NumPy/torch. Learned models default to
`device="cpu"` because cuDNN's LSTM/attention kernels are not deterministic —
switching to cuda trades reproducibility for speed.

**Ordering.** `build_feature_matrix`, the sequence tensorizer, and
`window_labels` all group by `window_id` with `sort=True`. That shared ordering
is what keeps scores aligned with labels across detectors — preserve it.

**Splits are by vessel, never by window.** `window_stride (16) < window_len
(32)`, so consecutive windows of one track share points; a window-level split
puts near-duplicates on both sides and leaks. Use
`eval.splits.train_test_split_by_vessel`.

**Supervised models need the held-out protocol.** The Transformer's
classification head is the only supervised component. Scoring it on its own
training windows measures memorisation. `--transformer` and
`sweep_attack_severity(holdout=True)` move *all* detectors onto the split
together, so the columns stay comparable.

## Evaluation gotchas

Per-attack metrics score each attack family **against clean windows only**, so
one attack's difficulty isn't diluted by the others.

The default attack severities are **saturated** — most detectors hit ~1.00
PR-AUC there, and the headline table cannot rank them. The informative
comparison is the severity sweep (`DEFAULT_SWEEPS` in `eval/robustness.py`),
whose ladders reach 1–3 orders of magnitude subtler. If a change appears to
make no difference, check it on the sweep before concluding anything.

The synthetic fleet is *perfectly* self-consistent (no position noise), so all
reported knees are optimistic; real AIS noise should push them right.

When comparing the Transformer to the others, state that it is supervised. The
defensible claim is "supervision buys ~a decade of attack subtlety here", not
"the Transformer is a better anomaly detector".

## Reporting results

Only cite metrics actually produced in the current session. Negative results
stay in the docs — the LSTM-AE's failure on `gradual_drift` (0.04 PR-AUC, at
chance) is recorded deliberately, with the capacity sweep showing it is a
property of the reconstruction objective rather than an undertrained model. It
is the reason the Transformer has a classification head.
