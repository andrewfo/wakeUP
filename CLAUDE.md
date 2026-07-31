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
PYTHONPATH=src python scripts/run_ablation.py  [--sweeps] [--lstm] [--hybrid]
PYTHONPATH=src python scripts/run_latency.py   [--transformer] [--max-fpr 0.05]
```

`make test` / `make milestone` / `make robustness` / `make lstm` /
`make transformer` / `make ablation` wrap the same commands (and assume an
installed package).

`--lstm` and `--transformer` need torch (`pip install -e ".[learned]"`). The
installed build is **CPU-only** — PyPI's default Windows wheel no longer bundles
CUDA. Everything else runs without torch; `wakeUp.models.__getattr__` imports
the torch-backed detectors lazily so `import wakeUp.models` never requires it.

The Transformer severity sweep is slow (~15 min on CPU: 21 severities × 3
detectors, retraining each time). Run it in the background.

## Canonical schema

Everything after ingest is `mmsi, timestamp, lat, lon, sog, cog, heading`
(+ `segment` after gap splitting, + `gap_s` after resampling, + `window_id` /
`point_idx` after windowing, + `is_attack` / `attack_type` / `window_label`
after injection). Synthetic and real AIS flow through identical code; real data
enters via `data.pipeline.load_marinecadastre_csv`.

**Gap splitting is off by default.** `DataConfig.max_gap_s=None` means one
segment per vessel, which is what every recorded number was produced under —
the synthetic fleet has no silences, so turning it on is a verified no-op
there. Set it (~600 s) for real AIS, where resampling across a dropout
fabricates smooth motion that then gets labelled clean. Grouping in
`resample_all` / `segment_windows` goes through `_group_keys`, so it degrades
to plain `mmsi` on frames that predate the column.

**The study-area crop is off by default too, and it is NOT a no-op.**
`DataConfig.crop_to_region=False` skips the `region_bbox` filter in
`clean_ais`. Turn it on for real AIS (a zone-wide download is not the study
area), but never turn it on and then compare against a recorded number:
synthetic vessels only *start* inside the box and then integrate out, so on the
default fleet the crop keeps 41.8% of cleaned fixes and 46.4% of windows
(1760 → 817) and loses 4 of 40 vessels entirely. Crop and gap split belong on
together — cropping an out-and-back excursion leaves a hole that resampling
would otherwise interpolate straight across.

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
ablation (below) sharpened the defensible claim: **supervision** buys ~a decade
of attack subtlety here, and a *linear classifier on the hand features* captures
almost all of it — so the claim is about supervision, not the sequence model.
The Transformer's own marginal edge over supervised features is small and shows
up only at the extreme-subtle end of the discontinuity attacks (a 1.01× speed
inflation, a 5 m jump). Never report "the Transformer is a better anomaly
detector"; unsupervised, its reconstruction arm is the *worst* cell in the grid.

**The ablation is what earns that claim.** `scripts/run_ablation.py` runs a 2×2
over *representation* (hand features vs learned sequence) × *supervision*
(unsupervised vs supervised), all four cells under one held-out-by-vessel split:

|                  | unsupervised       | supervised                |
|------------------|--------------------|---------------------------|
| hand features    | IsolationForest    | `LogisticFeatureDetector` |
| learned sequence | `ReconTransformer` | `TransformerDetector`     |

The two supervised cells share a **linear** head (sklearn logistic vs the
Transformer's linear `cls_head` on the pooled encoding), so left↔right isolates
supervision and top↔bottom isolates the learned representation with the
classifier held fixed. `ReconTransformerDetector` is the matched control: same
encoder as the Transformer but `supports_supervision = False`, so the held-out
harness fits it reconstruction-only even while it hands the supervised cells
their labels. `LogisticFeatureDetector` sets `consumes_windows = True` and builds
features internally, so it slots into the same harness branch as the sequence
models with no special-casing.

`--hybrid` adds a fifth cell, `HybridDetector` ("features+learned" ×
"supervised"): a logistic head over the 27 hand features ⊕ the Transformer's
mean-pooled encoder embedding (the exact vector `cls_head` sees), with the
encoder trained identically to the supervised Transformer cell. Gain over
Logistic ⇒ the embedding adds something; gain over Transformer ⇒ the features
do; the classifier stays linear throughout. It is supervised-only (`fit`
raises without labels) and needs the windows frame, not the feature matrix.
Result: the hybrid matches the *better* single representation on every rung of
every ladder — the honest claim is "the representations are complementary; the
embedding adds only in the extreme-subtle discontinuity regime", not "the
hybrid is a better detector overall" (it still pays full encoder training
cost, and elsewhere plain Logistic already sits on the frontier).

**Detection latency** (`eval/latency.py`, `scripts/run_latency.py`) replays
held-out windows as streaming prefixes. The alarm threshold is recalibrated
per prefix length to a fixed clean-window FPR (`clean_fpr_threshold`, quantile
`method="higher"` so ties can't blow the budget); alarms before the prefix
contains any attacked point are false positives, not detections; misses stay
in the table. Detectors fit once on full-length train windows and only score
prefixes.

## Reporting results

Only cite metrics actually produced in the current session. Negative results
stay in the docs — the LSTM-AE's failure on `gradual_drift` (0.04 PR-AUC, at
chance) is recorded deliberately, with the capacity sweep showing it is a
property of the reconstruction objective rather than an undertrained model. It
is the reason the Transformer has a classification head.
