"""Stage 7b — Volatility forecasting (GARCH/EWMA) for honest risk bands.

Regimes failed for DIRECTION, but vol clustering is real and predictable.
We forecast next-day variance with:
  - EWMA (RiskMetrics lambda=0.94) — always works
  - GARCH(1,1) via `arch` — if installed, preferred
Used to widen/narrow the forecast sigma and VaR bands honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger

log = get_logger(__name__)


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA variance -> annualized vol series."""
    r2 = returns.pow(2)
    var = r2.ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * 252)


def garch_vol_forecast(returns: pd.Series, horizon: int = 1) -> dict | None:
    """GARCH(1,1) next-day vol forecast. Returns None if arch missing/fails."""
    try:
        from arch import arch_model
    except ImportError:
        return None
    r = returns.dropna() * 100  # arch prefers %-scale
    if len(r) < 500:
        return None
    try:
        am = arch_model(r, vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=horizon, reindex=False)
        var_next = float(fc.variance.iloc[-1, 0]) / 1e4  # back to decimal
        return {"vol_next": float(np.sqrt(var_next)),
                "persistence": float(res.params.get("alpha[1]", 0)
                                     + res.params.get("beta[1]", 0)),
                "model": "GARCH-t"}
    except Exception as e:
        log.warning("garch fit failed: %s", e)
        return None


def vol_forecast(returns: pd.Series, window: int = 252) -> dict:
    """Combined: realized vol (21d, 63d), EWMA nowcast, GARCH forecast."""
    r = returns.dropna()
    out = {
        "vol_21d": float(r.tail(21).std() * np.sqrt(252)),
        "vol_63d": float(r.tail(63).std() * np.sqrt(252)),
        "vol_252d": float(r.tail(window).std() * np.sqrt(252)),
        "ewma_now": float(ewma_vol(r).iloc[-1]),
    }
    g = garch_vol_forecast(r)
    if g:
        out.update(g)
        out["vol_next_annual"] = float(g["vol_next"] * np.sqrt(252))
    else:
        out["vol_next_annual"] = out["ewma_now"]
        out["model"] = "EWMA"
    return out


if __name__ == "__main__":
    from src.data.target import load_raw
    from src.core.config import get_symbols
    import numpy as np
    r = np.log(load_raw(get_symbols()["target"])["Close"]).diff()
    print(vol_forecast(r))
