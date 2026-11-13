"""Target construction + data cleaning + trading calendar.

CRITICAL ALIGNMENT RULE
-----------------------
For supervised learning we want, at each feature date t:
    X_t  = features computable from data with timestamp <= t (close of day t)
    y_t  = ln(Close_{t+1} / Close_t)          # return earned t -> t+1

So the target column is stored AT row t but is only *known* after close of t+1.
Any feature built with data from t+1 or later is leakage. Stage 3 enforces this.

Fill rules (documented, conservative):
- Raw parquet: untouched (source of truth).
- Cleaned panel: dates = union of target + index trading days (NSE calendar).
  Missing OHLCV for a symbol on a market day -> NaN (never forward-filled
  silently into features; ffill allowed ONLY inside feature windows and
  always tagged, see src/features/).
- Target y_t is NaN on the last available date (no t+1 yet) — that row is
  used for INFERENCE, never for training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ROOT, get_symbols, load_yaml
from src.core.logger import get_logger

log = get_logger(__name__)

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def raw_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol.replace('^', 'IDX_').replace('=', '_')}.parquet"


def load_raw(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(raw_path(symbol))
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()


# ---------------------------------------------------------------- calendar

_CAL = None


def nse_calendar():
    """pandas_market_calendars NSE calendar (real holidays), cached."""
    global _CAL
    if _CAL is None:
        import pandas_market_calendars as mcal

        _CAL = mcal.get_calendar("NSE")
    return _CAL


def market_days(start: str, end: str | None = None) -> pd.DatetimeIndex:
    """All NSE trading days in [start, end]. end=None -> ~6 months ahead
    (enough to find the next trading day; holidays handled by calendar)."""
    cal = nse_calendar()
    end = end or (pd.Timestamp.today() + pd.Timedelta(days=210)).strftime("%Y-%m-%d")
    days = cal.valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days).tz_localize(None).normalize()


def latest_available_date(symbol: str | None = None) -> pd.Timestamp:
    symbol = symbol or get_symbols()["target"]
    return pd.Timestamp(load_raw(symbol).index.max())


def forecast_date() -> pd.Timestamp:
    """Next NSE trading day strictly after latest_available_date.

    Uses the real exchange calendar -> skips weekends AND NSE holidays.
    """
    from datetime import timedelta

    latest = pd.Timestamp(latest_available_date()).normalize()
    cal = nse_calendar()
    start_d = (latest.date() + timedelta(days=1)).isoformat()
    end_d = (latest.date() + timedelta(days=15)).isoformat()
    days = cal.valid_days(start_date=start_d, end_date=end_d)
    if len(days) == 0:
        raise RuntimeError("no trading day found within 15 days of latest data")
    return pd.Timestamp(days[0]).tz_localize(None).normalize()


# ------------------------------------------------------------------ target

def build_target_panel(symbol: str | None = None) -> pd.DataFrame:
    """Cleaned daily panel for the target with the aligned target column.

    Columns: Open High Low Close Adj_Close Volume target_ret
    target_ret at row t = ln(Close_{t+1}/Close_t); NaN on last row.
    """
    symbol = symbol or get_symbols()["target"]
    df = load_raw(symbol).copy()
    df = df.rename(columns={"Adj Close": "Adj_Close"})

    # target: return from t to t+1, stored at t
    df["target_ret"] = np.log(df["Close"].shift(-1) / df["Close"])

    # sanity: last row must have NaN target (no future close exists)
    assert pd.isna(df["target_ret"].iloc[-1]), "last target_ret must be NaN"

    out = PROC_DIR / f"{symbol.replace('^', 'IDX_').replace('=', '_')}_panel.parquet"
    df.to_parquet(out)
    log.info("panel saved %s rows=%d last=%s", out.name, len(df), df.index.max().date())
    return df


def main() -> None:
    panel = build_target_panel()
    latest = latest_available_date()
    fdate = forecast_date()
    print(f"latest_available_date = {latest.date()}")
    print(f"forecast_date          = {fdate.date()}  ({fdate.day_name()})")
    print(f"target rows            = {len(panel)}")
    print(f"target_ret stats (ex-last): mean={panel['target_ret'].iloc[:-1].mean():.6f} "
          f"std={panel['target_ret'].iloc[:-1].std():.6f}")
    print(f"non-null targets       = {panel['target_ret'].notna().sum()}")


if __name__ == "__main__":
    main()
