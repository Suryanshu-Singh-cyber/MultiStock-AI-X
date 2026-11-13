"""Stage 18 — Stress tests, scenarios, Monte Carlo.

Historical stress: replay known shock windows (2008 GFC, 2020 COVID,
2022 rate shock) through the return distribution.
Hypothetical scenarios: user-defined shocks to market/sector/VIX.
Monte Carlo: N=10,000 paths of next-day (and 21-day) returns from the
forecast Gaussian + fat-tail adjustment (Student-t with nu=5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger

log = get_logger(__name__)

STRESS_WINDOWS = {
    "gfc_2008": ("2008-09-01", "2008-12-31"),
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "russia_2022": ("2022-02-01", "2022-06-30"),
    "taper_2013": ("2013-05-01", "2013-09-30"),
}


def historical_stress(returns: pd.Series) -> pd.DataFrame:
    """Realized stats during historical stress windows."""
    rows = []
    for name, (a, b) in STRESS_WINDOWS.items():
        w = returns.loc[a:b].dropna()
        if len(w) < 5:
            continue
        eq = (1 + w).cumprod()
        rows.append({
            "scenario": name, "n_days": len(w),
            "total_ret": float(eq.iloc[-1] - 1),
            "max_dd": float(np.max(1 - eq / np.maximum.accumulate(eq))),
            "worst_day": float(w.min()),
            "vol": float(w.std() * np.sqrt(252)),
        })
    return pd.DataFrame(rows)


def hypothetical_scenario(base_mu: float, base_sigma: float,
                          mkt_shock: float = 0.0,
                          beta: float = 1.0,
                          vol_mult: float = 1.0) -> dict:
    """ shocked forecast: mu' = mu + beta*mkt_shock ; sigma' = sigma*vol_mult """
    mu2 = base_mu + beta * mkt_shock
    sig2 = base_sigma * vol_mult
    return {"mu": float(mu2), "sigma": float(sig2),
            "p10": float(mu2 + np.quantile(np.random.standard_t(5, 10000), 0.10) * sig2),
            "p90": float(mu2 + np.quantile(np.random.standard_t(5, 10000), 0.90) * sig2)}


def monte_carlo(mu: float, sigma: float, n_paths: int = 10_000,
                horizon: int = 21, nu: float = 5.0,
                seed: int = 42) -> dict:
    """Fat-tailed Monte Carlo for `horizon` days ahead.

    Daily returns ~ Student-t(nu) scaled to sigma (fat tails vs Gaussian).
    Returns VaR/ES/drawdown distribution summaries.
    """
    rng = np.random.default_rng(seed)
    # standardize t to unit variance then scale
    t = rng.standard_t(nu, size=(n_paths, horizon))
    t = t / np.sqrt(nu / (nu - 2))
    rets = mu + sigma * t
    paths = np.cumprod(1 + rets, axis=1)
    final = paths[:, -1] - 1
    # max drawdown per path
    peaks = np.maximum.accumulate(paths, axis=1)
    dd = (1 - paths / peaks).max(axis=1)

    def q(x, a):
        return float(np.quantile(x, a))

    return {
        "horizon_days": horizon, "n_paths": n_paths, "nu": nu,
        "final_ret_p05": q(final, 0.05), "final_ret_p50": q(final, 0.50),
        "final_ret_p95": q(final, 0.95),
        "prob_loss": float(np.mean(final < 0)),
        "var95_1d": float(np.quantile(rets[:, 0], 0.05)),
        "es95_1d": float(rets[:, 0][rets[:, 0] <= np.quantile(rets[:, 0], 0.05)].mean()),
        "maxdd_p50": q(dd, 0.50), "maxdd_p95": q(dd, 0.95),
        "maxdd_p99": q(dd, 0.99),
        "final_hist": {
            "edges": np.histogram(final, bins=24, range=(-0.15, 0.15))[1].tolist(),
            "counts": np.histogram(final, bins=24, range=(-0.15, 0.15))[0].tolist(),
        },
    }


if __name__ == "__main__":
    from src.data.target import load_raw
    from src.core.config import get_symbols
    r = np.log(load_raw(get_symbols()["target"])["Close"]).diff().dropna()
    print(historical_stress(r).to_string(index=False))
    print(monte_carlo(mu=0.0003, sigma=0.018))
