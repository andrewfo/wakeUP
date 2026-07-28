import numpy as np
import pandas as pd
import pytest

from wakeUp.config import DataConfig, AttackConfig
from wakeUp.data import generate_fleet, build_dataset


@pytest.fixture(scope="session")
def small_cfg():
    cfg = DataConfig(
        seed=1234,
        n_vessels=6,
        duration_hours=3.0,
        cadence_s=60,
        min_track_points=20,
        window_len=32,
        window_stride=16,
    )
    return cfg


@pytest.fixture(scope="session")
def windows(small_cfg):
    raw = generate_fleet(small_cfg)
    return build_dataset(raw, small_cfg)


@pytest.fixture(scope="session")
def one_window(windows):
    wid = sorted(windows["window_id"].unique())[0]
    return windows[windows["window_id"] == wid].copy()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def attack_cfg():
    return AttackConfig(seed=7, jump_km=8.0, drift_total_km=3.0, speed_multiplier=6.0)
