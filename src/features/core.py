"""Core feature set — small, leakage-safe, used by baselines (Stage 4) and
as a sanity subset of the full library (Stage 5).

TIMESTAMP RULE: every feature at row t uses only data with timestamp <= t
(close of day t). The target y_t = r_{t+1} is realized at close of t+1.
All rolling windows are trailing (min_periods enforced). No feature may
reference Close[t+1] or later — enforced by construction + Stage 3 tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import get_symbols, load_yaml
from src.data.target import load_raw, raw_path


def _rets(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def single_symbol_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Price/volume features for one symbol's raw OHLCV frame."""
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    r = _rets(c)
    feats = pd.DataFrame(index=df.index)

    p = prefix
    for k in (1, 2, 3, 5, 10, 21):
        feats[f"{p}ret_lag{k}"] = r.shift(k)
    for w in (5, 10, 21, 63):
        feats[f"{p}vol_{w}"] = r.rolling(w, min_periods=max(3, w // 2)).std()
        feats[f"{p}mom_{w}"] = c / c.shift(w) - 1.0
    feats[f"{p}rsi_14"] = _rsi(c, 14)
    feats[f"{p}sma_ratio_21"] = c / c.rolling(21, min_periods=10).mean() - 1.0
    feats[f"{p}vwap_dev"] = c / ((h + l + c) / 3).rolling(20, min_periods=10).mean() - 1.0
    feats[f"{p}vol_z_21"] = (v - v.rolling(21, min_periods=10).mean()) / (
        v.rolling(21, min_periods=10).std() + 1e-9)
    feats[f"{p}hl_range"] = (h - l) / c
    feats[f"{p}oc_ret"] = np.log(c / o)          # intraday return, known at close t
    feats[f"{p}gap"] = np.log(o / c.shift(1))    # overnight gap
    feats[f"{p}downside_vol_21"] = r.where(r < 0, 0).rolling(
        21, min_periods=10).std()
    return feats


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_core_features() -> pd.DataFrame:
    """Core feature frame for the target + market context.

    Returns DataFrame indexed by target trading dates with columns:
      target features + market (NIFTY) + sector + VIX + macro features.
    Rows where target doesn't trade are absent (inner join on target dates).
    """
    symbols = get_symbols()
    target = symbols["target"]
    tgt = load_raw(target)

    feats = single_symbol_features(tgt, prefix="tgt_")

    # market context: NIFTY return features (aligned by date, ffill not used —
    # indices trade same days; residual NaN handled downstream)
    nifty = load_raw(symbols["indices"]["nifty_50"])
    nr = _rets(nifty["Close"])
    mkt = pd.DataFrame(index=tgt.index)
    mkt["mkt_ret"] = nr.reindex(tgt.index)
    mkt["mkt_ret_lag1"] = nr.shift(1).reindex(tgt.index)
    mkt["mkt_vol_21"] = nr.rolling(21, min_periods=10).std().reindex(tgt.index)
    mkt["mkt_mom_21"] = (nifty["Close"] / nifty["Close"].shift(21) - 1).reindex(tgt.index)

    # sector index
    sec = load_raw(symbols["indices"]["sector_energy"])
    sr = _rets(sec["Close"])
    mkt["sec_ret"] = sr.reindex(tgt.index)
    mkt["sec_mom_21"] = (sec["Close"] / sec["Close"].shift(21) - 1).reindex(tgt.index)

    # VIX (level + change — known at close t)
    vix = load_raw(symbols["indices"]["india_vix"])
    mkt["vix"] = vix["Close"].reindex(tgt.index)
    mkt["vix_chg_5"] = vix["Close"].pct_change(5).reindex(tgt.index)

    # macro
    fx = load_raw(symbols["macro"]["usd_inr"])
    crude = load_raw(symbols["macro"]["crude"])
    mkt["usd_inr_ret_5"] = fx["Close"].pct_change(5).reindex(tgt.index)
    mkt["crude_ret_5"] = crude["Close"].pct_change(5).reindex(tgt.index)

    # target-relative-to-market (known at close t)
    tgt_r = _rets(tgt["Close"])
    mkt["tgt_mkt_spread_21"] = (
        (tgt_r.rolling(21, min_periods=10).mean()
         - nr.reindex(tgt.index).rolling(21, min_periods=10).mean()))

    out = pd.concat([feats, mkt], axis=1)
    out.index.name = "date"
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns]
