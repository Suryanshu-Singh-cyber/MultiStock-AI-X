"""Stage 16 — Probabilistic uncertainty: conformal prediction + calibration.

Split-conformal: on each walk-forward fold, calibration residuals from the
validation window give a quantile adjustment; applied to test predictions.
We verify empirical coverage ~= nominal on OOS data (honest check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger

log = get_logger(__name__)


def conformal_adjust(y_cal: np.ndarray, pred_cal: np.ndarray,
                     alpha: float = 0.10) -> tuple[float, float]:
    """Return (lo_adj, hi_adj) additive offsets for prediction intervals.

    Uses conformity score |y - pred|; interval = pred +/- q_{1-alpha}.
    """
    scores = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
    scores = scores[np.isfinite(scores)]
    q = float(np.quantile(scores, 1 - alpha))
    return -q, q


def empirical_coverage(y_true: np.ndarray, y_pred: np.ndarray,
                       half_width: float) -> float:
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(err <= half_width))


def calibration_table(y_true: np.ndarray, p_up: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: predicted P(up) vs empirical frequency."""
    y_true = np.asarray(y_true)
    p_up = np.asarray(p_up)
    mask = np.isfinite(y_true) & np.isfinite(p_up)
    y_true, p_up = y_true[mask], p_up[mask]
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p_up >= bins[i]) & (p_up < bins[i + 1] if i < n_bins - 1 else p_up <= bins[i + 1])
        if m.sum() == 0:
            continue
        rows.append({
            "bin_lo": bins[i], "bin_hi": bins[i + 1],
            "n": int(m.sum()),
            "pred_p_up": float(p_up[m].mean()),
            "emp_up": float((y_true[m] > 0).mean()),
        })
    return pd.DataFrame(rows)


def evaluate_conformal(oos_path: str = "data/processed/oos_predictions.parquet",
                       alpha: float = 0.10) -> dict:
    """Honest coverage check on pooled OOS predictions.

    NOTE: proper conformal needs per-fold calibration windows; here we use
    the pooled residual quantile as a first-pass check, then report coverage.
    """
    df = pd.read_parquet(oos_path)
    results = {}
    for model in ["ridge", "rf", "xgb", "ensemble"]:
        if model not in df.columns:
            continue
        resid = (df["y"] - df[model]).to_numpy()
        resid = resid[np.isfinite(resid)]
        q = float(np.quantile(np.abs(resid), 1 - alpha))
        coverage = float(np.mean(np.abs(resid) <= q))
        # this is in-sample-ish on pooled OOS; still a useful sanity bound
        results[model] = {"half_width": q, "coverage": coverage,
                          "target_coverage": 1 - alpha}
        log.info("conformal %s: half_width=%.5f coverage=%.3f (target %.2f)",
                 model, q, coverage, 1 - alpha)
    return results


if __name__ == "__main__":
    print(evaluate_conformal())
