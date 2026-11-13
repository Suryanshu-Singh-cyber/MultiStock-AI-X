"""Historical OHLCV ingestion via yfinance.

Downloads every tracked symbol from configs/symbols.yaml over
[history_start, today] and stores one parquet per symbol under data/raw/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.core.config import ROOT, get_all_symbols, get_symbols, load_yaml
from src.core.logger import get_logger

log = get_logger(__name__)

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass
class IngestReport:
    symbol: str
    rows: int
    first_date: str
    last_date: str
    gaps: int

    def __str__(self) -> str:
        return f"{self.symbol:15s} rows={self.rows:5d}  {self.first_date} -> {self.last_date}  gaps={self.gaps}"


def _symbol_kind(symbols_cfg: dict, symbol: str) -> str:
    if symbol == symbols_cfg["target"]:
        return "target"
    if symbol in symbols_cfg.get("peers", []):
        return "peer"
    if symbol in symbols_cfg.get("indices", {}).values():
        return "index"
    return "macro"


def download_symbol(
    symbol: str,
    start: str,
    end: str | None = None,
    kind: str = "unknown",
) -> pd.DataFrame:
    """Download one symbol. end=None -> latest available (rolling)."""
    log.info("downloading %s (%s) %s -> %s", symbol, kind, start, end or "today")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    # yfinance may return MultiIndex columns for single ticker in some versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[OHLCV_COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    out = RAW_DIR / f"{symbol.replace('^', 'IDX_').replace('=', '_')}.parquet"
    df.to_parquet(out)
    log.info("saved %s (%d rows)", out.name, len(df))
    return df


def ingest_all() -> list[IngestReport]:
    hist = load_yaml("config.yaml")["history"]
    start = hist["start"]
    end = hist.get("end")  # None => rolling through today
    symbols_cfg = get_symbols()

    reports: list[IngestReport] = []
    for symbol in get_all_symbols():
        kind = _symbol_kind(symbols_cfg, symbol)
        df = download_symbol(symbol, start=start, end=end, kind=kind)
        # count internal gaps > 7 calendar days (holiday tolerant)
        gaps = int((df.index.to_series().diff().dt.days > 7).sum())
        reports.append(
            IngestReport(
                symbol=symbol,
                rows=len(df),
                first_date=str(df.index.min().date()),
                last_date=str(df.index.max().date()),
                gaps=gaps,
            )
        )
    return reports


def latest_available_date(symbol: str | None = None) -> pd.Timestamp:
    """max(date) across cleaned raw data — NEVER hardcoded."""
    symbol = symbol or get_symbols()["target"]
    path = RAW_DIR / f"{symbol.replace('^', 'IDX_').replace('=', '_')}.parquet"
    df = pd.read_parquet(path, columns=["Close"])
    return pd.Timestamp(df.index.max())


def main() -> None:
    reports = ingest_all()
    print("\n=== INGEST REPORT ===")
    for r in reports:
        print(r)
    latest = latest_available_date()
    print(f"\nlatest_available_date (target) = {latest.date()}")


if __name__ == "__main__":
    main()
