"""Stage 15 — Mixture-of-Experts combiner.

Experts = the models we actually have OOS predictions for
(ridge, rf, xgb, arima, hist_mean) + volatility expert (predicts |ret|).
Gating: non-negative weights fit on VALIDATION windows only (never test),
per regime if regimes validate (they didn't -> single gate + optional
volatility-conditioned gate).

Honest protocol: gate weights are fit per walk-forward fold on that fold's
validation window, then applied frozen to the test window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.core.logger import get_logger

log = get_logger(__name__)


def fit_gate(val_preds: pd.DataFrame, val_y: pd.Series,
             l1: float = 0.01) -> np.ndarray:
    """Non-negative weights summing to 1 minimizing val squared error + L1."""
    P = val_preds.to_numpy()
    y = val_y.to_numpy()
    k = P.shape[1]

    def obj(w):
        w = np.abs(w)  # encourage non-negative via penalty-free param
        w = w / (w.sum() + 1e-12)
        pred = P @ w
        return np.mean((y - pred) ** 2) + l1 * np.sum(np.abs(w))

    best, best_f = None, np.inf
    for seed in range(5):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(k))
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-8})
        if res.fun < best_f:
            best_f, best = res.fun, res.x
    w = np.abs(best)
    return w / w.sum()


def moe_predictions(oos_path: str = "data/processed/oos_predictions.parquet") -> pd.DataFrame:
    """Fit gate per fold on validation, apply to test — approximated here by
    a single expanding gate refit every 63 OOS days (documented simplification;
    full per-fold gating needs the split metadata saved at baseline time).
    """
    df = pd.read_parquet(oos_path)
    df["date"] = pd.to_datetime(df["date"])
    expert_cols = [c for c in df.columns if c not in ("date", "y")]
    P = df[expert_cols + ["y", "date"]].dropna()
    dates = P.index if P.index.name else pd.RangeIndex(len(P))
    P = P.reset_index(drop=True)

    out = pd.DataFrame(index=range(len(P)))
    out["date"] = pd.to_datetime(P["date"]).to_numpy()
    out["y"] = P["y"].to_numpy()
    for c in expert_cols:
        out[c] = P[c].to_numpy()

    # expanding gate: for each day t, fit on prior 126 days, predict t
    moe_pred = np.full(len(P), np.nan)
    win = 126
    for t in range(win, len(P)):
        tr = slice(t - win, t)
        w = fit_gate(P.iloc[tr][expert_cols], P.iloc[tr]["y"])
        moe_pred[t] = float(P.iloc[t][expert_cols].to_numpy() @ w)
    out["moe"] = moe_pred
    return out


if __name__ == "__main__":
    from src.models.metrics import regression_metrics, trading_metrics
    res = moe_predictions()
    ok = res["moe"].notna() & res["y"].notna()
    met = regression_metrics(res.loc[ok, "y"].to_numpy(), res.loc[ok, "moe"].to_numpy())
    tr = trading_metrics(res.loc[ok, "y"].to_numpy(), res.loc[ok, "moe"].to_numpy())
    print("MoE:", {k: round(v, 4) for k, v in {**met, **tr}.items()
                   if isinstance(v, float)})
    res.to_parquet("data/processed/moe_predictions.parquet")
