"""Stage 25 — FastAPI backend.

Endpoints:
  GET  /health
  GET  /forecast/latest     current forecast (Redis cache -> disk fallback)
  GET  /forecast/history    last 90 days of daily reports
  GET  /risk/latest         VaR/ES/drawdown from latest forecast
  GET  /drivers/latest      SHAP contributions
  WS   /stream              forecast_update + price_tick events
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import ROOT
from src.core.logger import get_logger
from src.streaming.bus import get_bus

log = get_logger(__name__)
app = FastAPI(title="MultiStock-AI X", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

REPORTS_LIVE = ROOT / "reports" / "live"


def _latest_payload() -> dict | None:
    bus = get_bus()
    cached = bus.cache_get("forecast:latest")
    if cached:
        return cached
    if REPORTS_LIVE.exists():
        files = sorted(f for f in REPORTS_LIVE.glob("*.json")
                       if f.stem[:4].isdigit() and len(f.stem) == 10)
        if files:
            return json.loads(files[-1].read_text())
    return None


@app.get("/health")
def health():
    p = _latest_payload()
    return {"status": "ok", "forecast_available": p is not None,
            "ts": datetime.now().isoformat(timespec="seconds")}


@app.get("/forecast/latest")
def forecast_latest():
    p = _latest_payload()
    if p is None:
        return {"status": "NO_FORECAST", "detail": "run daily inference first"}
    return p


@app.get("/price/history")
def price_history(days: int = 60):
    """Recent daily closes for charting."""
    import pandas as pd
    from src.data.target import load_raw
    from src.core.config import get_symbols
    df = load_raw(get_symbols()["target"])[["Close", "Volume"]].tail(days)
    return [{"date": str(i.date()), "close": float(r["Close"]),
             "volume": int(r["Volume"])} for i, r in df.iterrows()]


@app.get("/backtest/equity")
def backtest_equity(model: str = "arima"):
    """Equity curve of sign-strategy on pooled OOS predictions (net of costs)."""
    import numpy as np
    import pandas as pd
    p = ROOT / "data" / "processed" / "oos_predictions.parquet"
    if not p.exists():
        return {"error": "no oos predictions"}
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if model not in df.columns:
        model = "ensemble"
    y = df["y"].to_numpy(); pr = df[model].to_numpy()
    pos = (pr > 0).astype(float)
    cost = 0.0026
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * y - cost * turnover
    eq = np.cumprod(1 + net)
    bh = np.cumprod(1 + np.nan_to_num(y, nan=0.0))
    return {"model": model, "dates": [str(d.date()) for d in df["date"]],
            "strategy": eq.tolist(), "buy_hold": bh.tolist(),
            "final_strategy": float(eq[-1] - 1),
            "final_buyhold": float(bh[-1] - 1)}


@app.get("/scorecard")
def scorecard():
    """Live model report card — real scored forecasts, not backtests."""
    p = REPORTS_LIVE / "scorecard.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"total_scored": 0, "message": "no scored forecasts yet — "
            "log accumulates daily after market close"}


_refresh_state = {"running": False, "started": None, "done": None, "error": None}


@app.post("/forecast/refresh")
def forecast_refresh():
    """Trigger the full daily pipeline now (data refresh -> inference -> score)."""
    import threading
    import src.streaming.scheduler as sched

    if _refresh_state["running"]:
        return {"status": "already_running"}

    def _run():
        _refresh_state.update(running=True,
                              started=datetime.now().isoformat(timespec="seconds"),
                              error=None)
        try:
            sched.daily_job()
            if sched.LAST_ERROR:
                _refresh_state["error"] = sched.LAST_ERROR
        except Exception as e:  # noqa: BLE001
            _refresh_state["error"] = str(e)
        finally:
            _refresh_state["running"] = False
            _refresh_state["done"] = datetime.now().isoformat(timespec="seconds")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/forecast/refresh/status")
def refresh_status():
    return _refresh_state


@app.get("/forecast/history")
def forecast_history(days: int = 90):
    if not REPORTS_LIVE.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for f in sorted(REPORTS_LIVE.glob("*.json")):
        try:
            p = json.loads(f.read_text())
            if datetime.fromisoformat(p["generated_at"]) >= cutoff:
                out.append({
                    "date": p.get("latest_data_date"),
                    "forecast_date": p.get("forecast_date"),
                    "mu": p.get("forecast", {}).get("mu_log_return"),
                    "sigma": p.get("forecast", {}).get("sigma"),
                    "p_up": p.get("forecast", {}).get("p_up"),
                    "no_trade": p.get("sizing", {}).get("no_trade"),
                    "status": p.get("status"),
                })
        except Exception:
            continue
    return out


@app.get("/risk/latest")
def risk_latest():
    p = _latest_payload() or {}
    return {"risk": p.get("risk", {}), "monte_carlo_21d": p.get("monte_carlo_21d", {}),
            "status": p.get("status")}


@app.get("/drivers/latest")
def drivers_latest():
    p = _latest_payload() or {}
    return {"drivers": p.get("drivers", {}),
            "model_preds": p.get("forecast", {}).get("model_preds", {})}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    bus = get_bus()
    loop = None
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        q = asyncio.Queue()

        def _on_event(payload):
            loop.call_soon_threadsafe(q.put_nowait, payload)

        bus.subscribe("forecast_update", _on_event)
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30)
                await ws.send_json(payload)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping",
                                    "ts": datetime.now().isoformat(timespec="seconds")})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("ws error: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
