"""Stage 17 — Risk engine.

From forecast distribution (mu, sigma) + trailing historical windows <= t-1:
  VaR_95/99 (parametric Gaussian + historical-simulation cross-check)
  Expected Shortfall
  rolling beta vs NIFTY, rolling vol, max drawdown of equity curve
All windows are trailing — nothing uses data after t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy import stats


def var_es_gaussian(mu: float, sigma: float,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Parametric VaR/ES for return ~ N(mu, sigma^2). Returns are negative-side."""
    z = stats.norm.ppf(alpha)
    var = mu + sigma * z
    # ES = mu - sigma * phi(z)/alpha
    es = mu - sigma * stats.norm.pdf(z) / alpha
    return float(var), float(es)


def var_es_historical(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Historical-simulation VaR/ES on a trailing window."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return np.nan, np.nan
    var = float(np.quantile(r, alpha))
    tail = r[r <= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def risk_snapshot(forecast_mu: float, forecast_sigma: float,
                  hist_returns: np.ndarray,
                  nifty_returns: np.ndarray | None = None,
                  window: int = 252) -> dict:
    """One-date risk summary."""
    r = np.asarray(hist_returns, float)[-window:]
    r = r[np.isfinite(r)]
    out = {
        "vol_annual": float(np.std(r) * np.sqrt(252)) if len(r) > 1 else np.nan,
        "vol_21": float(np.std(r[-21:])) if len(r) >= 21 else np.nan,
    }
    for a in (0.05, 0.01):
        v_g, e_g = var_es_gaussian(forecast_mu, max(forecast_sigma, 1e-6), a)
        v_h, e_h = var_es_historical(r, a)
        out[f"var{int((1-a)*100)}_gauss"] = v_g
        out[f"es{int((1-a)*100)}_gauss"] = e_g
        out[f"var{int((1-a)*100)}_hist"] = v_h
        out[f"es{int((1-a)*100)}_hist"] = e_h
    if nifty_returns is not None:
        nr = np.asarray(nifty_returns, float)[-window:]
        ok = np.isfinite(r[-len(nr):]) & np.isfinite(nr)
        if ok.sum() > 30 and np.std(nr[ok]) > 0:
            out["beta_252"] = float(np.cov(r[-len(nr):][ok], nr[ok])[0, 1]
                                    / np.var(nr[ok]))
    # drawdown of trailing equity
    eq = np.cumprod(1 + np.nan_to_num(r, nan=0.0))
    peak = np.maximum.accumulate(eq)
    out["max_drawdown_window"] = float(np.max(1 - eq / peak)) if len(eq) else 0.0
    return out


def rolling_risk_series(hist_returns: pd.Series,
                        forecast_mu: float = 0.0,
                        forecast_sigma: float | None = None,
                        window: int = 252) -> pd.DataFrame:
    """VaR/ES time series using trailing windows (for backtest/report)."""
    r = hist_returns.dropna()
    rows = []
    for i in range(window, len(r)):
        w = r.iloc[i - window:i].to_numpy()
        sig = forecast_sigma if forecast_sigma is not None else float(np.std(w))
        v95, e95 = var_es_gaussian(forecast_mu, sig, 0.05)
        v99, e99 = var_es_gaussian(forecast_mu, sig, 0.01)
        vh95, eh95 = var_es_historical(w, 0.05)
        rows.append({"date": r.index[i], "vol": np.std(w),
                     "var95": v95, "es95": e95, "var99": v99, "es99": e99,
                     "var95_hist": vh95, "es95_hist": eh95})
    return pd.DataFrame(rows).set_index("date")
