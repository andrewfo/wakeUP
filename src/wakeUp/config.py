"""Lightweight config + determinism helpers.

The full plan targets Hydra; for the first-milestone slice we keep a small
dataclass-backed loader with zero required third-party deps so the pipeline
runs anywhere. YAML is loaded if PyYAML is present, otherwise the built-in
defaults are used.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for the milestone slice
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class DataConfig:
    seed: int = 1234
    n_vessels: int = 40
    duration_hours: float = 12.0
    cadence_s: int = 60          # resample cadence
    min_track_points: int = 30   # drop shorter tracks
    window_len: int = 32         # points per trajectory window
    window_stride: int = 16      # overlap between windows
    region_bbox: tuple[float, float, float, float] = (
        # (lat_min, lon_min, lat_max, lon_max) — a patch off the US east coast
        36.0, -75.5, 37.5, -73.5,
    )


@dataclass
class AttackConfig:
    seed: int = 7
    # fraction of windows to corrupt, per attack type
    contamination: float = 0.15
    jump_km: float = 8.0
    drift_total_km: float = 3.0
    speed_multiplier: float = 6.0


@dataclass
class ModelConfig:
    seed: int = 0
    iforest_estimators: int = 300
    iforest_contamination: float = 0.1
    # LSTM autoencoder (Phase 4). Small by design: the windows are 32 points of
    # 7 channels, so a wider model overfits the attacked minority it is meant
    # to reconstruct badly.
    lstm_hidden: int = 32
    lstm_epochs: int = 40
    lstm_lr: float = 1e-3
    lstm_batch_size: int = 64
    lstm_contamination: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: dict[str, str] = field(
        default_factory=lambda: {
            "raw": "data/raw",
            "interim": "data/interim",
            "processed": "data/processed",
            "figures": "figures",
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a Config, optionally overlaying a YAML file if PyYAML is installed."""
    cfg = Config()
    if path is None:
        return cfg
    try:
        import yaml
    except Exception:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for section in ("data", "attack", "model"):
        if section in raw and raw[section]:
            current = getattr(cfg, section)
            for k, v in raw[section].items():
                if hasattr(current, k):
                    setattr(current, k, v)
    if "paths" in raw and raw["paths"]:
        cfg.paths.update(raw["paths"])
    return cfg
