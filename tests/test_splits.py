"""Train/test split tests (Phase 4/5 supervised arm).

The supervised Transformer's numbers are only meaningful if the split really
holds out unseen vessels — overlapping windows make a naive window-level split
leak — so the disjointness guarantee is tested directly.
"""

import numpy as np

from wakeUp.config import AttackConfig
from wakeUp.attacks import build_attacked_dataset
from wakeUp.eval.splits import train_test_split_by_vessel, window_labels


def test_vessels_are_disjoint(windows):
    train, test = train_test_split_by_vessel(windows, test_frac=0.3, seed=0)
    assert not (set(train["mmsi"]) & set(test["mmsi"]))
    assert len(train) > 0 and len(test) > 0
    assert len(train) + len(test) == len(windows)


def test_no_window_spans_the_split(windows):
    """Every window_id lands wholly on one side (no shared points)."""
    train, test = train_test_split_by_vessel(windows, test_frac=0.3, seed=0)
    assert not (set(train["window_id"]) & set(test["window_id"]))


def test_split_is_deterministic(windows):
    a, _ = train_test_split_by_vessel(windows, seed=3)
    b, _ = train_test_split_by_vessel(windows, seed=3)
    c, _ = train_test_split_by_vessel(windows, seed=4)
    assert set(a["mmsi"]) == set(b["mmsi"])
    assert set(a["mmsi"]) != set(c["mmsi"])  # seed actually varies the split


def test_train_side_never_empty(windows):
    """An extreme test_frac must still leave vessels to train on."""
    train, test = train_test_split_by_vessel(windows, test_frac=0.99, seed=0)
    assert train["mmsi"].nunique() >= 1
    assert test["mmsi"].nunique() >= 1


def test_window_labels_align_with_sorted_ids(windows):
    attacked = build_attacked_dataset(windows, AttackConfig())
    labels = window_labels(attacked)
    wids = sorted(attacked["window_id"].unique())
    assert len(labels) == len(wids)
    # spot-check alignment against a direct lookup
    for i in (0, len(wids) // 2, len(wids) - 1):
        expected = attacked.loc[attacked["window_id"] == wids[i], "window_label"].iloc[0]
        assert labels[i] == int(expected)
    assert set(np.unique(labels)) <= {0, 1}
