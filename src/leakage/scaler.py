"""Scaler/encoder wrapper that can ONLY be fit on training data.

Any attempt to fit on data extending to/past the transform cutoff raises
LeakageError. This is the mechanical enforcement of 'fit on train only'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.leakage.guards import LeakageError, assert_fit_data_before_transform


class TrainOnlyScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that refuses to see the future.

    Usage:
        scaler = TrainOnlyScaler()
        Xtr_s = scaler.fit_transform_dated(X_train, train_dates)
        Xte_s = scaler.transform(X_test)          # transform never fits

    `fit` requires the index max of the fit data to be < the cutoff you will
    transform at. Pass cutoff explicitly via fit_dated.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fit_max: pd.Timestamp | None = None

    def fit(self, X, y=None):
        # Plain fit allowed only for internal/sklearn compat; dated path preferred.
        self._scaler.fit(X)
        return self

    def fit_dated(self, X: pd.DataFrame, dates: pd.DatetimeIndex,
                  cutoff: pd.Timestamp) -> "TrainOnlyScaler":
        """Fit on X whose dates are all strictly < cutoff."""
        assert_fit_data_before_transform(pd.Timestamp(dates.max()), cutoff,
                                         context="TrainOnlyScaler.fit")
        self._scaler.fit(X)
        self._fit_max = pd.Timestamp(dates.max())
        return self

    def transform(self, X):
        return self._scaler.transform(X)

    def fit_transform_dated(self, X: pd.DataFrame, dates: pd.DatetimeIndex,
                            cutoff: pd.Timestamp):
        self.fit_dated(X, dates, cutoff)
        return self.transform(X)

    # sklearn compat
    def fit_transform(self, X, y=None):
        return self._scaler.fit_transform(X)

    @property
    def mean_(self):
        return self._scaler.mean_

    @property
    def scale_(self):
        return self._scaler.scale_


class LeakageCheckingImputer:
    """Median imputer that enforces train-only fitting on dated data."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._fit_max: pd.Timestamp | None = None

    def fit_dated(self, X: pd.DataFrame, dates: pd.DatetimeIndex,
                  cutoff: pd.Timestamp) -> "LeakageCheckingImputer":
        assert_fit_data_before_transform(pd.Timestamp(dates.max()), cutoff,
                                         context="Imputer.fit")
        self._values = {c: float(X[c].median()) for c in X.columns}
        self._fit_max = pd.Timestamp(dates.max())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(pd.Series(self._values)).astype(float)
