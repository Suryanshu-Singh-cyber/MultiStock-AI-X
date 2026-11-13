"""Domain-specific causality guards: news, graph edges, regimes, hyperparameters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.leakage.guards import LeakageError


def assert_news_first_available(first_available, feature_cutoff, *, headline: str = "") -> None:
    """A news item may only influence features at dates STRICTLY AFTER its
    first-available timestamp. If the article was first available at 14:00 on
    day d, features stamped d (computed at close) may use it; features stamped
    < d may not. We enforce first_available < feature_cutoff.
    """
    if pd.Timestamp(first_available) >= pd.Timestamp(feature_cutoff):
        raise LeakageError(
            f"news '{headline}': first_available {first_available} "
            f">= feature_cutoff {feature_cutoff}"
        )


def assert_graph_edge_causal(edge_date, data_through_date, *, context: str = "") -> None:
    """A graph edge/weight at time t must be computed from data with
    timestamp <= data_through_date, and data_through_date must be < edge_date
    (edges at t use info up to t-1 at the latest, typically close of t-1).
    """
    if pd.Timestamp(data_through_date) >= pd.Timestamp(edge_date):
        raise LeakageError(
            f"{context}: edge at {edge_date} built from data through "
            f"{data_through_date} (must be strictly earlier)"
        )


def assert_regime_causal(regime_date, data_through_date, *, context: str = "") -> None:
    """Regime label stamped `regime_date` must derive from data strictly
    before that date (HMM filtered/smoothed state at t uses data <= t-1 when
    predicting t)."""
    if pd.Timestamp(data_through_date) >= pd.Timestamp(regime_date):
        raise LeakageError(
            f"{context}: regime at {regime_date} uses data through "
            f"{data_through_date}"
        )


def assert_hp_selected_on_train(val_scores: dict, train_end, test_start, *, context: str = "") -> None:
    """Hyperparameter selection metadata guard: the validation scores used to
    pick hyperparameters must come from a window ending before test_start."""
    if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
        raise LeakageError(
            f"{context}: HP selection window ends {train_end} >= test start {test_start}"
        )
