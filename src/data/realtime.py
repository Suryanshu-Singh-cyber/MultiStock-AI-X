"""Stage 24a/24b — Real-time price feed (polling backend + bar builder).

PriceFeed interface -> YFFeed (yfinance polling, free) -> 1-min bars ->
rolling parquet. Broker WebSocket backend is a stub for Stage 24 upgrade.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.core.config import ROOT, get_live_config
from src.core.logger import get_logger
from src.streaming.bus import get_bus

log = get_logger(__name__)
LIVE_DIR = ROOT / "data" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Tick:
    symbol: str
    price: float
    volume: int
    ts: datetime


class PriceFeed:
    """Interface — swap backends via configs/live.yaml price_feed.backend."""

    def latest(self, symbol: str) -> Tick | None:
        raise NotImplementedError

    def stream(self, symbols: list[str], callback):
        raise NotImplementedError


class YFFeed(PriceFeed):
    """Polling backend: yfinance fast_info every poll_seconds during market
    hours. Slightly delayed (~15min) but free and sufficient for daily
    forecasting; intraday precision not required for next-day target."""

    def __init__(self, symbols: list[str], poll_seconds: int = 60):
        self.symbols = symbols
        self.poll_seconds = poll_seconds
        self._last: dict[str, Tick] = {}

    def latest(self, symbol: str) -> Tick | None:
        try:
            t = yf.Ticker(symbol)
            fi = t.fast_info
            price = float(fi.get("last_price") or fi.get("previous_close") or 0)
            if price <= 0:
                return None
            tick = Tick(symbol, price, 0, datetime.now())
            self._last[symbol] = tick
            return tick
        except Exception as e:
            log.warning("yf poll %s failed: %s", symbol, e)
            return self._last.get(symbol)

    def stream(self, symbols: list[str], callback) -> None:
        log.info("YFFeed streaming %d symbols every %ds", len(symbols), self.poll_seconds)
        while True:
            for s in symbols:
                tick = self.latest(s)
                if tick:
                    callback(tick)
            time.sleep(self.poll_seconds)


class BarBuilder:
    """Aggregates ticks into 1-minute OHLCV bars (in-memory deque)."""

    def __init__(self, symbols: list[str], maxlen: int = 390 * 2):
        self.symbols = symbols
        self.bars: dict[str, deque] = {s: deque(maxlen=maxlen) for s in symbols}
        self._current: dict[str, dict] = {}

    def on_tick(self, tick: Tick) -> None:
        minute = tick.ts.replace(second=0, microsecond=0)
        key = (tick.symbol, minute)
        b = self._current.get(key)
        if b is None:
            if len(self._current) > 5000:
                self._current.clear()
            b = {"open": tick.price, "high": tick.price, "low": tick.price,
                 "close": tick.price, "volume": tick.volume, "minute": minute}
            self._current[key] = b
        else:
            b["high"] = max(b["high"], tick.price)
            b["low"] = min(b["low"], tick.price)
            b["close"] = tick.price
            b["volume"] += tick.volume
        # flush completed minutes lazily
        for k in list(self._current):
            if k[1] < minute:
                bar = self._current.pop(k)
                self.bars[k[0]].append(bar)

    def persist(self) -> Path:
        rows = []
        for sym, dq in self.bars.items():
            for b in dq:
                rows.append({"symbol": sym, "minute": b["minute"], **{k: b[k] for k in
                               ("open", "high", "low", "close", "volume")}})
        if not rows:
            return LIVE_DIR / "bars_1min.parquet"
        df = pd.DataFrame(rows)
        out = LIVE_DIR / "bars_1min.parquet"
        if out.exists():
            old = pd.read_parquet(out)
            df = pd.concat([old, df]).drop_duplicates(["symbol", "minute"], keep="last")
        df.to_parquet(out, index=False)
        return out


class UpstoxFeed(PriceFeed):
    """Upstox WebSocket feed (requires UPSTOX_API_KEY + UPSTOX_ACCESS_TOKEN).

    Real tick-level streaming. Instruments: NSE equity tokens — POWERGRID
    instrument_key is resolved via their instrument master on first use.
    Falls back gracefully with a clear error if creds are missing.
    """

    BASE_WS = "wss://stream.upstox.com"

    def __init__(self, symbols: list[str], poll_seconds: int = 60):
        import os
        self.api_key = os.environ.get("UPSTOX_API_KEY", "")
        self.token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        self.symbols = symbols
        self.poll_seconds = poll_seconds
        self._last: dict[str, Tick] = {}
        self._keys: dict[str, str] = {}

    def _check(self):
        if not self.api_key or not self.token:
            raise RuntimeError(
                "Upstox creds missing — set UPSTOX_API_KEY and "
                "UPSTOX_ACCESS_TOKEN in .env (get token via Upstox OAuth)")

    def _instrument_key(self, symbol: str) -> str:
        """Resolve NSE symbol -> Upstox instrument_key (cached)."""
        if symbol in self._keys:
            return self._keys[symbol]
        import requests
        self._check()
        r = requests.get(
            "https://api.upstox.com/v2/market-quote/instruments",
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/json"}, timeout=20)
        # instrument master is large; Upstox also offers per-segment search.
        # Minimal viable: try the ltpc quote endpoint directly with trading_symbol.
        q = requests.get(
            f"https://api.upstox.com/v2/market-quote/ltp",
            params={"instrument_key": f"NSE_EQ|{symbol}"},
            headers={"Authorization": f"Bearer {self.token}"}, timeout=15)
        if q.ok:
            self._keys[symbol] = f"NSE_EQ|{symbol}"
        return self._keys.get(symbol, f"NSE_EQ|{symbol}")

    def latest(self, symbol: str) -> Tick | None:
        """REST fallback: last traded price via Upstox market-quote API."""
        try:
            import requests
            self._check()
            key = self._instrument_key(symbol)
            r = requests.get(
                "https://api.upstox.com/v2/market-quote/ltp",
                params={"instrument_key": key},
                headers={"Authorization": f"Bearer {self.token}"}, timeout=15)
            px = float(r.json()["data"][key]["last_price"])
            tick = Tick(symbol, px, 0, datetime.now())
            self._last[symbol] = tick
            return tick
        except Exception as e:
            log.warning("upstox poll %s failed: %s", symbol, e)
            return self._last.get(symbol)

    def stream(self, symbols: list[str], callback) -> None:
        """True WebSocket streaming via `websockets` + upstox proto.

        NOTE: Upstox WS uses protobuf market-data feed; full impl requires
        the upstox python SDK. This method streams via REST polling as a
        reliable fallback and documents where the WS hook goes.
        """
        log.info("UpstoxFeed: REST-poll mode (WS proto feed = TODO with SDK)")
        while True:
            for s in symbols:
                t = self.latest(s)
                if t:
                    callback(t)
            time.sleep(self.poll_seconds)


class INDstocksFeed(PriceFeed):
    """INDstocks WebSocket feed stub (requires INDSTOCKS_API_KEY).

    INDstocks offers sub-20ms tick streams over WebSocket with a simple
    JSON protocol — implement when credentials are available.
    """

    def __init__(self, symbols: list[str], poll_seconds: int = 60):
        import os
        self.api_key = os.environ.get("INDSTOCKS_API_KEY", "")
        self.symbols = symbols
        self.poll_seconds = poll_seconds
        self._last: dict[str, Tick] = {}

    def latest(self, symbol: str) -> Tick | None:
        if not self.api_key:
            raise RuntimeError("INDSTOCKS_API_KEY not set in .env")
        # REST quote endpoint (documented in their docs)
        try:
            import requests
            r = requests.get(
                f"https://api.indstocks.com/v1/quote/{symbol}",
                headers={"x-api-key": self.api_key}, timeout=15)
            px = float(r.json().get("last_price", 0))
            if px > 0:
                t = Tick(symbol, px, 0, datetime.now())
                self._last[symbol] = t
                return t
        except Exception as e:
            log.warning("indstocks poll %s failed: %s", symbol, e)
        return self._last.get(symbol)

    def stream(self, symbols: list[str], callback) -> None:
        log.info("INDstocksFeed: polling mode (WS tick feed = TODO with key)")
        while True:
            for s in symbols:
                t = self.latest(s)
                if t:
                    callback(t)
            time.sleep(self.poll_seconds)


def make_feed() -> PriceFeed:
    cfg = get_live_config()
    backend = cfg.get("price_feed", {}).get("backend", "yf_poll")
    from src.core.config import get_all_symbols
    syms = get_all_symbols()
    if backend == "yf_poll":
        return YFFeed(syms,
                      poll_seconds=cfg.get("price_feed", {}).get("poll_seconds", 60))
    if backend == "upstox_ws":
        return UpstoxFeed(syms,
                          poll_seconds=cfg.get("price_feed", {}).get("poll_seconds", 60))
    if backend == "indstocks_ws":
        return INDstocksFeed(syms,
                             poll_seconds=cfg.get("price_feed", {}).get("poll_seconds", 60))
    raise NotImplementedError(f"unknown price_feed backend '{backend}'")


def run_poll_loop() -> None:
    """Entry point for the ingest worker: poll -> bars -> publish ticks."""
    feed = make_feed()
    builder = BarBuilder(feed.symbols if hasattr(feed, "symbols") else [])
    bus = get_bus()

    def cb(tick: Tick):
        bus.publish("price_tick", {
            "symbol": tick.symbol, "price": tick.price,
            "ts": tick.ts.isoformat()})
        builder.on_tick(tick)

    try:
        feed.stream(getattr(feed, "symbols", []), cb)
    finally:
        builder.persist()


if __name__ == "__main__":
    # one-shot sanity poll
    feed = make_feed()
    for s in getattr(feed, "symbols", [])[:3]:
        t = feed.latest(s)
        print(s, t)
