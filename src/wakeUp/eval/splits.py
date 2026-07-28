"""Train/test splitting for the supervised arm (Phase 4/5).

Every detector up to now has been unsupervised: fit and score run on the same
contaminated set, which is a legitimate protocol because labels never enter
training. The Transformer's classification head breaks that — a supervised
model scored on the windows it trained on reports its memorisation, not its
detection ability — so it needs a held-out split.

The split must be **by vessel**, not by window. Windows are built with
``window_stride < window_len``, so consecutive windows of one track share
points; splitting on ``window_id`` would put near-duplicate windows on both
sides and leak. Grouping by ``mmsi`` keeps whole tracks on one side.

The unsupervised detectors are re-scored on the same test split so the
comparison table is like-for-like rather than mixing protocols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_by_vessel(
    windows: pd.DataFrame,
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a windows frame into (train, test) with disjoint vessels.

    ``test_frac`` is the fraction of *vessels* held out; the resulting window
    fraction is close but not exact, since tracks differ in length.
    """
    rng = np.random.default_rng(seed)
    vessels = np.array(sorted(windows["mmsi"].unique()))
    if len(vessels) < 2:
        raise ValueError("need at least 2 vessels to split by vessel")
    n_test = max(1, int(round(test_frac * len(vessels))))
    n_test = min(n_test, len(vessels) - 1)  # never leave the train side empty
    test_vessels = set(rng.choice(vessels, size=n_test, replace=False).tolist())

    is_test = windows["mmsi"].isin(test_vessels)
    return windows[~is_test].copy(), windows[is_test].copy()


def window_labels(windows: pd.DataFrame) -> np.ndarray:
    """Per-window labels in sorted ``window_id`` order.

    Matches the ordering used by ``build_feature_matrix`` and the sequence
    tensorizer, so labels align with any detector's scores.
    """
    return (
        windows.groupby("window_id", sort=True)["window_label"]
        .first()
        .to_numpy()
        .astype(int)
    )
