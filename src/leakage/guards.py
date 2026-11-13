"""Core leakage guards. Every guard raises LeakageError — never warns."""
from __future__ import annotations


class LeakageError(RuntimeError):
    """Raised when any component attempts to use information from the future."""


def assert_feature_time_before_target(feature_time, target_time, *, context: str = "") -> None:
    """A feature row stamped `feature_time` must be knowable strictly before
    the target it predicts is realized.

    For next-day return targets: feature date t is knowable at close of t,
    target r_{t+1} is realized at close of t+1, so feature_time <= t < t+1.
    We enforce feature_time < target_time where target_time is the date the
    target return *ends* (t+1). Feature at t predicting target ending t+1
    passes; anything stamped >= target end fails.
    """
    if feature_time >= target_time:
        raise LeakageError(
            f"{context}: feature_time {feature_time} >= target_time {target_time}"
        )


def assert_train_before_test(train_max, test_min, *, context: str = "") -> None:
    """Last training timestamp must be strictly before first test timestamp."""
    if train_max >= test_min:
        raise LeakageError(
            f"{context}: train_max {train_max} >= test_min {test_min}"
        )


def assert_fit_data_before_transform(data_max, cutoff, *, context: str = "") -> None:
    """A scaler/encoder/imputer fitted on `data` may only see rows with
    timestamp < cutoff (the first timestamp it will transform for eval)."""
    if data_max >= cutoff:
        raise LeakageError(
            f"{context}: fit data extends to {data_max} >= cutoff {cutoff}"
        )
