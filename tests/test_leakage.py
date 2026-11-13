"""Leakage framework tests — these PASS when guards work correctly.

Each test constructs a violation and asserts LeakageError is raised, plus a
clean case asserting no error. If a guard silently stops firing, tests fail.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.target import build_target_panel, forecast_date, latest_available_date
from src.leakage.domain_guards import (
    assert_graph_edge_causal,
    assert_news_first_available,
    assert_regime_causal,
)
from src.leakage.guards import (
    LeakageError,
    assert_feature_time_before_target,
    assert_fit_data_before_transform,
    assert_train_before_test,
)
from src.leakage.scaler import LeakageCheckingImputer, TrainOnlyScaler
from src.leakage.split import walk_forward_splits

DATES = pd.bdate_range("2020-01-01", periods=400)


# ------------------------------------------------------------- core guards

def test_feature_before_target_passes():
    assert_feature_time_before_target(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), context="t->t+1")


def test_feature_before_target_fires():
    with pytest.raises(LeakageError):
        assert_feature_time_before_target(
            pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03"), context="same-day")
    with pytest.raises(LeakageError):
        assert_feature_time_before_target(
            pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-03"), context="future")


def test_train_before_test_fires():
    with pytest.raises(LeakageError):
        assert_train_before_test(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01"))
    with pytest.raises(LeakageError):
        assert_train_before_test(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-06-01"))


def test_fit_data_before_transform_fires():
    with pytest.raises(LeakageError):
        assert_fit_data_before_transform(
            pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-01"))


# ------------------------------------------------------------- domain guards

def test_news_guard():
    # news at 14:00 day d -> feature cutoff end-of-day d is fine? we require
    # first_available < cutoff; equal timestamps FAIL (conservative).
    assert_news_first_available("2024-01-02 14:00", "2024-01-03")
    with pytest.raises(LeakageError):
        assert_news_first_available("2024-01-03 09:00", "2024-01-03")
    with pytest.raises(LeakageError):
        assert_news_first_available("2024-01-04", "2024-01-03")


def test_graph_edge_guard():
    assert_graph_edge_causal("2024-01-03", "2024-01-02")
    with pytest.raises(LeakageError):
        assert_graph_edge_causal("2024-01-03", "2024-01-03")  # same-day = leak
    with pytest.raises(LeakageError):
        assert_graph_edge_causal("2024-01-03", "2024-01-04")


def test_regime_guard():
    assert_regime_causal("2024-01-03", "2024-01-02")
    with pytest.raises(LeakageError):
        assert_regime_causal("2024-01-03", "2024-01-03")


# ------------------------------------------------------------- scaler

def test_train_only_scaler_blocks_future_fit():
    X = pd.DataFrame(np.random.randn(100, 3), columns=list("abc"))
    dates = DATES[:100]
    cutoff = dates[80]
    # fitting on data that reaches the cutoff must fail
    scaler = TrainOnlyScaler()
    with pytest.raises(LeakageError):
        scaler.fit_dated(X.iloc[:81], dates[:81], cutoff)
    # fitting strictly before cutoff works, transform on later data works
    scaler2 = TrainOnlyScaler()
    scaler2.fit_dated(X.iloc[:80], dates[:80], cutoff)
    out = scaler2.transform(X.iloc[80:90])
    assert out.shape == (10, 3)
    # statistics come from train only
    assert np.allclose(scaler2.mean_, X.iloc[:80].mean().values)


def test_imputer_train_only():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
    dates = DATES[:4]
    imp = LeakageCheckingImputer().fit_dated(X, dates[:3], dates[3])
    assert imp._values["a"] == 2.0  # median of [1,3] -> nan excluded? check


# ------------------------------------------------------------- split

def test_walk_forward_split_properties():
    splits = walk_forward_splits(DATES, train_days=100, valid_days=20, test_days=10)
    assert len(splits) > 0
    for s in splits:
        # strict ordering
        assert s.train_dates.max() < s.valid_dates.min()
        assert s.valid_dates.max() < s.test_dates.min()
        # windows contiguous, non-overlapping
        assert s.train[-1] + 1 == s.valid[0]
        assert s.valid[-1] + 1 == s.test[0]
        # all indices unique across a split
        all_idx = np.concatenate([s.train, s.valid, s.test])
        assert len(np.unique(all_idx)) == len(all_idx)


def test_walk_forward_uses_config_defaults():
    # 400 days < 1260 train -> zero splits (correct: not enough history)
    assert walk_forward_splits(DATES) == []


# ------------------------------------------------------------- real data

@pytest.fixture(scope="module")
def panel():
    return build_target_panel()


def test_real_target_alignment(panel):
    """target_ret[t] must equal ln(C[t+1]/C[t]) — re-verify mechanically."""
    expected = np.log(panel["Close"].shift(-1) / panel["Close"])
    a = panel["target_ret"].to_numpy()
    b = expected.to_numpy()
    mask = ~np.isnan(a) & ~np.isnan(b)
    assert np.allclose(a[mask], b[mask])


def test_forecast_date_is_future(panel):
    fdate = forecast_date()
    latest = latest_available_date()
    assert fdate > latest
    assert fdate.dayofweek < 5  # weekday


def test_walk_forward_on_real_panel(panel):
    # Contract: pass only dates that HAVE a label (target_ret not NaN).
    # The final row is inference-only (no t+1 close yet) and must be excluded.
    label_dates = panel.index[panel["target_ret"].notna()]
    splits = walk_forward_splits(label_dates, max_splits=3)
    assert len(splits) == 3
    for s in splits:
        assert s.test_dates.max() <= label_dates.max()
