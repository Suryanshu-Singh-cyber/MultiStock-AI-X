"""Walk-forward splitter — the ONLY sanctioned way to make train/test splits.

Windows roll forward over the trading calendar. No shuffle, no random split,
no 'train 80% / test 20%'. Every window is (train, valid, test) with strict
temporal ordering enforced by guards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import load_yaml
from src.leakage.guards import assert_train_before_test


@dataclass(frozen=True)
class Split:
    train: np.ndarray   # positional indices into the feature matrix
    valid: np.ndarray
    test: np.ndarray
    train_dates: pd.DatetimeIndex
    valid_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def _window_bounds(dates: pd.DatetimeIndex, train_days: int, valid_days: int, test_days: int):
    """Yield (train_slice, valid_slice, test_slice) positional bounds."""
    n = len(dates)
    min_start = train_days
    step = test_days  # roll forward by one test window each iteration
    start = min_start
    while start + train_days + valid_days + test_days <= n:
        tr = slice(start - train_days, start)
        va = slice(start, start + valid_days)
        te = slice(start + valid_days, start + valid_days + test_days)
        yield tr, va, te
        start += step


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_days: int | None = None,
    valid_days: int | None = None,
    test_days: int | None = None,
    max_splits: int | None = None,
) -> list[Split]:
    """Generate rolling (train, valid, test) splits over `dates` (sorted)."""
    wf = load_yaml("config.yaml").get("model", {}).get("walk_forward", {})
    train_days = train_days or wf.get("train_window_days", 1260)
    valid_days = valid_days or wf.get("valid_window_days", 126)
    test_days = test_days or wf.get("test_window_days", 21)

    dates = pd.DatetimeIndex(dates).sort_values()
    assert dates.is_unique, "dates must be unique trading days"

    splits: list[Split] = []
    for tr, va, te in _window_bounds(dates, train_days, valid_days, test_days):
        train_idx = np.arange(tr.start, tr.stop)
        valid_idx = np.arange(va.start, va.stop)
        test_idx = np.arange(te.start, te.stop)

        # mechanical guard: strict temporal ordering
        assert_train_before_test(dates[train_idx[-1]], dates[valid_idx[0]], context="train->valid")
        assert_train_before_test(dates[valid_idx[-1]], dates[test_idx[0]], context="valid->test")
        assert_train_before_test(dates[train_idx[-1]], dates[test_idx[0]], context="train->test")

        splits.append(Split(
            train=train_idx, valid=valid_idx, test=test_idx,
            train_dates=dates[train_idx], valid_dates=dates[valid_idx],
            test_dates=dates[test_idx],
        ))
        if max_splits and len(splits) >= max_splits:
            break
    return splits
