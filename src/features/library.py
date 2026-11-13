"""Stage 5 — Candidate feature library (~300 features).

Categories (every feature documents lookback; all timestamps <= t):
  A. price momentum & mean-reversion      (~60)
  B. volatility (multi-window, estimators) (~50)
  C. volume & liquidity                    (~40)
  D. trend / oscillators                   (~40)
  E. market & sector relationship          (~50)
  F. peer cross-section                    (~40)
  G. macro                                 (~20)
  H. calendar                              (~10)

TIMESTAMP RULE (unchanged): feature at row t uses data with timestamp <= t.
All windows are trailing .rolling()/.shift() — nothing references t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import get_symbols
from src.data.target import load_raw
from src.features.core import _rsi

RET_LAGS = [1, 2, 3, 5, 10, 21, 42, 63]
MOM_WINDOWS = [5, 10, 21, 42, 63, 126, 252]
VOL_WINDOWS = [5, 10, 21, 42, 63, 126]


def _logret(c: pd.Series) -> pd.Series:
    return np.log(c).diff()


def _block_price_vol(df: pd.DataFrame, p: str) -> pd.DataFrame:
    """Categories A+B for one OHLCV frame."""
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    r = _logret(c)
    f = pd.DataFrame(index=df.index)

    # A. momentum / reversion
    for k in RET_LAGS:
        f[f"{p}ret_lag{k}"] = r.shift(k)
    for w in MOM_WINDOWS:
        f[f"{p}mom_{w}"] = c / c.shift(w) - 1.0
        f[f"{p}logmom_{w}"] = np.log(c / c.shift(w))
    f[f"{p}ret_skew_63"] = r.rolling(63, min_periods=30).skew()
    f[f"{p}ret_kurt_63"] = r.rolling(63, min_periods=30).kurt()
    f[f"{p}max_ret_21"] = r.rolling(21, min_periods=10).max()
    f[f"{p}min_ret_21"] = r.rolling(21, min_periods=10).min()
    f[f"{p}up_days_21"] = (r > 0).rolling(21, min_periods=10).mean()

    # B. volatility estimators
    for w in VOL_WINDOWS:
        f[f"{p}vol_{w}"] = r.rolling(w, min_periods=max(3, w // 3)).std()
        f[f"{p}downside_vol_{w}"] = r.where(r < 0, 0.0).rolling(
            w, min_periods=max(3, w // 3)).std()
        f[f"{p}parkinson_{w}"] = np.sqrt(
            (np.log(h / l) ** 2).rolling(w, min_periods=max(3, w // 3)).mean() / (4 * np.log(2)))
        f[f"{p}gk_{w}"] = np.sqrt(
            (0.5 * np.log(h / l) ** 2
             - (2 * np.log(2) - 1) * np.log(c / o) ** 2)
            .rolling(w, min_periods=max(3, w // 3)).mean().clip(lower=0))
    f[f"{p}vol_ratio_5_63"] = f[f"{p}vol_5"] / (f[f"{p}vol_63"] + 1e-9)
    f[f"{p}vol_of_vol_21"] = f[f"{p}vol_21"].rolling(21, min_periods=10).std()
    f[f"{p}atr_pct_14"] = ((h - l) / c).rolling(14, min_periods=7).mean()
    f[f"{p}hl_range_5"] = ((h - l) / c).rolling(5, min_periods=3).mean()

    # C. volume
    lv = np.log(v.replace(0, np.nan))
    for w in (5, 21, 63):
        f[f"{p}vol_z_{w}"] = (v - v.rolling(w, min_periods=max(3, w // 3)).mean()) / (
            v.rolling(w, min_periods=max(3, w // 3)).std() + 1e-9)
        f[f"{p}vol_mom_{w}"] = v / v.rolling(w, min_periods=max(3, w // 3)).mean() - 1
    f[f"{p}vol_price_corr_21"] = r.rolling(21, min_periods=10).corr(lv.diff())
    f[f"{p}obv_slope_21"] = np.sign(r).rolling(21, min_periods=10).sum() / 21
    f[f"{p}vol_trend_63"] = lv.rolling(63, min_periods=30).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 2 else np.nan,
        raw=True)

    # D. trend / oscillators
    for w in (10, 21, 50, 200):
        f[f"{p}sma_ratio_{w}"] = c / c.rolling(w, min_periods=max(5, w // 3)).mean() - 1
        f[f"{p}ema_ratio_{w}"] = c / c.ewm(span=w, adjust=False).mean() - 1
    f[f"{p}rsi_14"] = _rsi(c, 14)
    f[f"{p}rsi_7"] = _rsi(c, 7)
    # MACD
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f[f"{p}macd"] = macd / c
    f[f"{p}macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
    f[f"{p}macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    # efficiency ratio (Kaufman)
    direction = (c - c.shift(10)).abs()
    vol_path = r.abs().rolling(10, min_periods=5).sum()
    f[f"{p}eff_ratio_10"] = direction / (vol_path + 1e-9)
    # drawdown / position-in-range (known at t)
    roll_max = c.rolling(252, min_periods=60).max()
    roll_min = c.rolling(252, min_periods=60).min()
    f[f"{p}dist_52w_high"] = c / roll_max - 1
    f[f"{p}dist_52w_low"] = c / roll_min - 1
    f[f"{p}pos_in_52w_range"] = (c - roll_min) / (roll_max - roll_min + 1e-9)
    # current drawdown from rolling peak
    f[f"{p}drawdown_63"] = c / c.rolling(63, min_periods=20).max() - 1
    # streaks
    up = (r > 0).astype(float)
    f[f"{p}consec_up"] = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    f[f"{p}consec_up"] = f[f"{p}consec_up"].where(up == 1, 0)
    dn = (r < 0).astype(float)
    f[f"{p}consec_dn"] = dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1
    f[f"{p}consec_dn"] = f[f"{p}consec_dn"].where(dn == 1, 0)
    # expanding z-score of today's return (adaptive normalization)
    f[f"{p}ret_z_expanding"] = r / r.expanding(min_periods=60).std()
    # close vs VWAP-proxy (typical price) multi-window
    tp = (h + l + c) / 3
    for w in (10, 21):
        f[f"{p}tp_dev_{w}"] = c / tp.rolling(w, min_periods=5).mean() - 1
    # intraday microstructure (all known at close t)
    f[f"{p}oc_ret"] = np.log(c / o)
    f[f"{p}gap"] = np.log(o / c.shift(1))
    f[f"{p}body_ratio"] = (c - o).abs() / (h - l + 1e-9)
    f[f"{p}upper_shadow"] = (h - np.maximum(c, o)) / (h - l + 1e-9)
    f[f"{p}lower_shadow"] = (np.minimum(c, o) - l) / (h - l + 1e-9)
    f[f"{p}close_loc"] = (c - l) / (h - l + 1e-9)
    return f


def _block_market_context(feat_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Category E: relationship to NIFTY / sector / VIX."""
    symbols = get_symbols()
    idx = feat_index
    out = pd.DataFrame(index=idx)

    def aligned_close(sym):
        return load_raw(sym)["Close"].reindex(idx)

    tgt_c = aligned_close(symbols["target"])
    nifty_c = aligned_close(symbols["indices"]["nifty_50"])
    sec_c = aligned_close(symbols["indices"]["sector_energy"])
    vix_c = aligned_close(symbols["indices"]["india_vix"])

    tr, nr, sr = _logret(tgt_c), _logret(nifty_c), _logret(sec_c)

    # rolling beta of target vs NIFTY (cov/var, trailing)
    for w in (21, 63, 126):
        cov = tr.rolling(w, min_periods=w // 2).cov(nr)
        var = nr.rolling(w, min_periods=w // 2).var()
        out[f"beta_nifty_{w}"] = cov / (var + 1e-12)
        out[f"corr_nifty_{w}"] = tr.rolling(w, min_periods=w // 2).corr(nr)
        out[f"corr_sector_{w}"] = tr.rolling(w, min_periods=w // 2).corr(sr)
        out[f"rel_ret_{w}"] = tr.rolling(w, min_periods=w // 2).mean() - \
                             nr.rolling(w, min_periods=w // 2).mean()

    # market state
    out["mkt_ret"] = nr
    out["mkt_ret_lag1"] = nr.shift(1)
    out["mkt_mom_5"] = nifty_c / nifty_c.shift(5) - 1
    out["mkt_mom_21"] = nifty_c / nifty_c.shift(21) - 1
    out["mkt_vol_21"] = nr.rolling(21, min_periods=10).std()
    out["sec_ret"] = sr
    out["sec_mom_21"] = sec_c / sec_c.shift(21) - 1
    out["vix"] = vix_c
    out["vix_ret_1"] = _logret(vix_c)
    out["vix_chg_5"] = vix_c.pct_change(5)
    out["vix_chg_21"] = vix_c.pct_change(21)
    out["vix_rank_252"] = vix_c.rolling(252, min_periods=60).rank(pct=True)
    # target idiosyncratic vol vs market (residual of rolling beta approx)
    out["idio_vol_21"] = (tr - out["beta_nifty_21"] * nr).rolling(
        21, min_periods=10).std()
    return out


def _block_peers(feat_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Category F: peer returns/momentum, aligned to target dates."""
    symbols = get_symbols()
    idx = feat_index
    out = pd.DataFrame(index=idx)
    peer_rets = {}
    for peer in symbols.get("peers", []):
        c = load_raw(peer)["Close"].reindex(idx)
        r = _logret(c)
        peer_rets[peer] = r
        short = peer.split(".")[0].lower()
        out[f"peer_{short}_ret"] = r
        out[f"peer_{short}_mom21"] = c / c.shift(21) - 1
        out[f"peer_{short}_mom5"] = c / c.shift(5) - 1
        out[f"peer_{short}_vol21"] = r.rolling(21, min_periods=10).std()
        out[f"peer_{short}_ret_lag1"] = r.shift(1)
    if peer_rets:
        rdf = pd.DataFrame(peer_rets)
        out["peer_median_ret"] = rdf.median(axis=1)
        out["peer_dispersion_21"] = rdf.rolling(21, min_periods=10).std().mean(axis=1)
        out["peer_breadth_up"] = (rdf > 0).rolling(21, min_periods=10).mean().mean(axis=1)
        # target vs peer-basket spread
        tgt_r = _logret(load_raw(symbols["target"])["Close"].reindex(idx))
        out["tgt_vs_peers_21"] = tgt_r.rolling(21, min_periods=10).mean() - \
                                rdf.mean(axis=1).rolling(21, min_periods=10).mean()
    return out


def _block_macro(feat_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Category G: FX, crude, and their relation to target."""
    symbols = get_symbols()
    idx = feat_index
    out = pd.DataFrame(index=idx)
    for name, sym in symbols.get("macro", {}).items():
        c = load_raw(sym)["Close"].reindex(idx)
        r = _logret(c)
        key = name.replace("-", "_")
        out[f"{key}_ret_1"] = r
        out[f"{key}_ret_5"] = c.pct_change(5)
        out[f"{key}_mom_21"] = c / c.shift(21) - 1
        out[f"{key}_vol_21"] = r.rolling(21, min_periods=10).std()
    return out


def _block_calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Category H: day-of-week / month effects (known at t)."""
    out = pd.DataFrame(index=idx)
    dow = idx.dayofweek
    for d in range(5):
        out[f"dow_{d}"] = (dow == d).astype(float)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    out["day_of_month"] = idx.day / 31.0
    out["is_month_end"] = idx.is_month_end.astype(float)
    return out


def build_feature_library() -> pd.DataFrame:
    symbols = get_symbols()
    tgt = load_raw(symbols["target"])
    idx = tgt.index  # target trading dates only

    blocks = [
        _block_price_vol(tgt, "tgt_"),
        _block_market_context(idx),
        _block_peers(idx),
        _block_macro(idx),
        _block_calendar(idx),
    ]
    X = pd.concat(blocks, axis=1)
    X.index.name = "date"
    # drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")
    return X


if __name__ == "__main__":
    X = build_feature_library()
    print(f"feature library: {X.shape[1]} features x {X.shape[0]} rows")
    print(f"NaN fraction overall: {X.isna().mean().mean():.3f}")
    print("first 10 cols:", list(X.columns[:10]))
