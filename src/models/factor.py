"""Stage 11 — Factor model integration.

We don't have commercial Fama-French India factors, so we CONSTRUCT factors
from our universe (all causal, trailing):
  MKT   = NIFTY 50 return
  SEC   = CNX Energy return
  MOM   = peer-basket momentum long-minus-short (top vs bottom tercile by mom_21)
  VOL   = -VIX change (low-vol anomaly proxy)
  FX    = USDINR return
  CRUDE = WTI return

The alpha layer: regress target daily return on factors (rolling 252d,
coefficients fitted through t-1). Incremental value = does adding factor
predictions improve OOS dir-acc vs alpha-only baseline?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import get_symbols
from src.core.logger import get_logger
from src.data.target import load_raw

log = get_logger(__name__)


def build_factors() -> pd.DataFrame:
    symbols = get_symbols()
    idx = load_raw(symbols["target"]).index

    def lr(sym):
        return np.log(load_raw(sym)["Close"]).diff().reindex(idx)

    f = pd.DataFrame(index=idx)
    f["MKT"] = lr(symbols["indices"]["nifty_50"])
    f["SEC"] = lr(symbols["indices"]["sector_energy"])
    f["FX"] = lr(symbols["macro"]["usd_inr"])
    f["CRUDE"] = lr(symbols["macro"]["crude"])
    vix = load_raw(symbols["indices"]["india_vix"])["Close"].reindex(idx)
    f["VOL"] = -vix.pct_change().fillna(0)  # rising VIX -> negative factor ret

    # MOM: peer momentum L/S — at each t rank peers by mom_21, long top half
    peer_rets = {}
    peer_px = {}
    for p in symbols.get("peers", []):
        c = load_raw(p)["Close"].reindex(idx)
        peer_px[p] = c
        peer_rets[p] = np.log(c).diff()
    mom = pd.DataFrame({p: c / c.shift(21) - 1 for p, c in peer_px.items()})
    rets = pd.DataFrame(peer_rets)
    rank = mom.rank(axis=1, pct=True)
    long_leg = (rank >= 0.5).astype(float).div((rank >= 0.5).sum(axis=1), axis=0)
    short_leg = (rank < 0.5).astype(float).div((rank < 0.5).sum(axis=1), axis=0)
    f["MOM"] = (long_leg * rets).sum(axis=1) - (short_leg * rets).sum(axis=1)
    return f


def factor_alpha_returns(window: int = 252) -> pd.DataFrame:
    """Target return decomposed into factor-predicted + residual (alpha).

    Coefficients at row t fitted on [t-window, t-1] — causal.
    """
    symbols = get_symbols()
    tgt_r = np.log(load_raw(symbols["target"])["Close"]).diff()
    F = build_factors()
    idx = F.index
    y = tgt_r.reindex(idx)

    out = pd.DataFrame(index=idx)
    cols = ["fitted", "alpha", *F.columns]
    for c in cols:
        out[c] = np.nan
    out[F.columns] = F

    Y = y.to_numpy()
    X = F.to_numpy()
    n = len(idx)
    for t in range(window, n):
        lo = t - window
        Xw, yw = X[lo:t], Y[lo:t]
        ok = np.isfinite(Xw).all(axis=1) & np.isfinite(yw)
        if ok.sum() < window // 2:
            continue
        Xd = np.column_stack([np.ones(ok.sum()), Xw[ok]])
        try:
            coef, *_ = np.linalg.lstsq(Xd, yw[ok], rcond=None)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(X[t]).all() and np.isfinite(Y[t]):
            out.iloc[t, out.columns.get_loc("fitted")] = np.concatenate([[1.0], X[t]]) @ coef
            out.iloc[t, out.columns.get_loc("alpha")] = Y[t] - out.iloc[t, out.columns.get_loc("fitted")]
    return out


if __name__ == "__main__":
    fa = factor_alpha_returns()
    print(fa[["fitted", "alpha"]].describe().round(5))
    print(f"alpha var share = {fa['alpha'].var() / fa['alpha'].add(fa['fitted']).var():.3f}")
    fa.to_parquet("data/processed/factor_alpha.parquet")
