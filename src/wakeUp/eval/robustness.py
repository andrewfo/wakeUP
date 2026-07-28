"""Robustness sweeps over attack subtlety (Phase 5).

The headline benchmark reports each detector at *one* attack severity, which
makes gross attacks look solved (PR-AUC ≈ 1.00) and says nothing about where a
detector actually breaks. A robustness sweep re-runs the same evaluation across
a ladder of severities for one attack knob at a time and reports the
**degradation curve**: detection quality as a function of how subtle the
spoofer is willing to be.

Three knobs are swept, one per attack family whose difficulty is continuously
parameterised:

    position_jump         -> jump_km           (how far the teleport is)
    kinematic_impossible  -> speed_multiplier  (how inflated the step spacing is)
    gradual_drift         -> drift_total_km    (how far the ramp diverges)

``identity_swap`` and ``replay`` have no continuous subtlety knob (they splice
whole segments), so they are not swept.

Each sweep point rebuilds the attacked dataset from the *same* clean windows
with only the severity changed. ``build_attacked_dataset`` draws its window
selection from ``cfg.seed``, so the identical set of windows is corrupted at
every severity — the curve isolates severity, not sampling noise.

Detectors are re-fit at each sweep point. They are unsupervised and fit on the
same contaminated matrix they score, exactly as in ``scripts/run_milestone.py``,
so the numbers are comparable with the headline table.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Mapping

import pandas as pd

from wakeUp.attacks import AttackType, build_attacked_dataset
from wakeUp.config import AttackConfig, ModelConfig
from wakeUp.eval.metrics import evaluate_scores
from wakeUp.features import build_feature_matrix
from wakeUp.models import IsolationForestDetector, KinematicRuleDetector

# Which AttackConfig knob controls each attack family's subtlety.
SWEEP_PARAM: dict[AttackType, str] = {
    AttackType.POSITION_JUMP: "jump_km",
    AttackType.KINEMATIC_IMPOSSIBLE: "speed_multiplier",
    AttackType.GRADUAL_DRIFT: "drift_total_km",
}

# Severity ladders, subtle -> gross. These are log-spaced and reach much lower
# than the benchmark's headline settings on purpose: the synthetic fleet is
# *perfectly* self-consistent (no position noise), so consistency residuals
# still separate a 100 m teleport at 1.00 PR-AUC and the informative region is
# below that. Real AIS position noise will push these knees to the right, which
# is itself a result worth reporting rather than a ladder to re-tune away.
DEFAULT_SWEEPS: dict[str, list[float]] = {
    "jump_km": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 8.0],
    "speed_multiplier": [1.01, 1.05, 1.1, 1.25, 1.5, 3.0, 6.0],
    "drift_total_km": [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0],
}

# A detector spec is a zero-arg factory, so every sweep point gets a fresh
# unfitted instance rather than silently reusing fitted state.
DetectorFactories = Mapping[str, Callable[[], object]]


def default_detectors(
    model_cfg: ModelConfig | None = None, include_lstm: bool = False
) -> dict[str, Callable[[], object]]:
    """The milestone's baselines, as factories.

    ``include_lstm`` adds the LSTM autoencoder; it retrains at every sweep
    point, so it is opt-in and needs the ``learned`` extra.
    """
    cfg = model_cfg or ModelConfig()
    factories: dict[str, Callable[[], object]] = {
        "KinematicRule": lambda: KinematicRuleDetector(),
        "IsolationForest": lambda: IsolationForestDetector(cfg),
    }
    if include_lstm:
        from wakeUp.models import LSTMAutoencoderDetector

        factories["LSTM-AE"] = lambda: LSTMAutoencoderDetector(cfg)
    return factories


def sweep_attack_severity(
    windows: pd.DataFrame,
    attack_type: AttackType | str,
    values: Iterable[float] | None = None,
    attack_cfg: AttackConfig | None = None,
    detectors: DetectorFactories | None = None,
    recall_target: float = 0.90,
) -> pd.DataFrame:
    """Degradation curve for one attack family over its subtlety knob.

    Returns one row per ``(severity value, detector)`` with the standard
    metrics from :func:`~wakeUp.eval.metrics.evaluate_scores` — the binary task
    at each point is {this attack} vs clean windows, since only this attack
    type is injected.
    """
    atype = AttackType(attack_type)
    if atype not in SWEEP_PARAM:
        raise ValueError(
            f"{atype.value} has no continuous subtlety knob; "
            f"sweepable types: {[a.value for a in SWEEP_PARAM]}"
        )
    param = SWEEP_PARAM[atype]
    ladder = list(values) if values is not None else list(DEFAULT_SWEEPS[param])
    base = attack_cfg or AttackConfig()
    factories = detectors or default_detectors()

    rows = []
    for v in ladder:
        cfg_v = replace(base, **{param: float(v)})
        attacked = build_attacked_dataset(windows, cfg_v, attack_types=[atype])
        feat_df, labels = build_feature_matrix(attacked)
        for name, make in factories.items():
            det = make()
            # Sequence models want the per-point frame; both groupings sort by
            # window_id, so scores stay aligned with `labels` either way.
            payload = attacked if getattr(det, "consumes_windows", False) else feat_df
            det.fit(payload)
            metrics = evaluate_scores(labels, det.score(payload), recall_target)
            rows.append(
                {
                    "attack_type": atype.value,
                    "param": param,
                    "value": float(v),
                    "detector": name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_robustness_sweeps(
    windows: pd.DataFrame,
    attack_cfg: AttackConfig | None = None,
    detectors: DetectorFactories | None = None,
    sweeps: Mapping[AttackType, Iterable[float]] | None = None,
    recall_target: float = 0.90,
) -> pd.DataFrame:
    """Run every sweepable attack family and concatenate the curves."""
    if sweeps is None:
        sweeps = {a: DEFAULT_SWEEPS[p] for a, p in SWEEP_PARAM.items()}
    frames = [
        sweep_attack_severity(
            windows,
            atype,
            values=values,
            attack_cfg=attack_cfg,
            detectors=detectors,
            recall_target=recall_target,
        )
        for atype, values in sweeps.items()
    ]
    return pd.concat(frames, ignore_index=True)
