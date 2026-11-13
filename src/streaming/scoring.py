"""Live prediction scoring — the 'report card'.

Every daily forecast is appended to data/live/forecast_log.parquet.
score_pending() checks which forecasts now have a realized next-day return
and computes running metrics: directional accuracy, Brier, MAE, coverage.
This gives TRUE live accuracy — not backtest numbers.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ROOT
from src.core.logger import get_logger
from src.data.target import build_target_panel, forecast_date, latest_available_date

log = get_logger(__name__)
LIVE = ROOT / "data" / "live"
LIVE.mkdir(parents=True, exist_ok=True)
LOG_PATH = LIVE / "forecast_log.parquet"
SCORE_PATH = ROOT / "reports" / "live" / "scorecard.json"


def log_forecast(payload: dict) -> None:
    """Append one forecast row (called by inference/scheduler)."""
    f = payload.get("forecast", {})
    row = {
        "logged_at": payload.get("generated_at"),
        "latest_data_date": payload.get("latest_data_date"),
        "forecast_date": payload.get("forecast_date"),
        "mu": f.get("mu_log_return"),
        "sigma": f.get("sigma"),
        "p_up": f.get("p_up"),
        "p10": f.get("p10"), "p50": f.get("p50"), "p90": f.get("p90"),
        "no_trade": payload.get("sizing", {}).get("no_trade"),
        "status": payload.get("status"),
        **{f"pred_{k}": v for k, v in (f.get("model_preds") or {}).items()},
    }
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        old = pd.read_parquet(LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["forecast_date"], keep="last")
    df.to_parquet(LOG_PATH, index=False)
    log.info("forecast logged for %s", row["forecast_date"])


def score_pending() -> dict:
    """Score forecasts whose forecast_date has now passed and actual return exists."""
    if not LOG_PATH.exists():
        return {"scored": 0, "message": "no forecasts logged yet"}
    log_df = pd.read_parquet(LOG_PATH)
    panel = build_target_panel()
    # realized return for forecast_date d = ln(C_d / C_{d-1}) = target_ret at d-1
    px = panel["Close"]

    scored = log_df.copy()
    scored["actual_ret"] = np.nan
    scored["scored"] = False
    for i, r in scored.iterrows():
        fd = pd.Timestamp(r["forecast_date"])
        if fd not in px.index:
            continue  # forecast date hasn't happened yet (or holiday mismatch)
        pos = px.index.get_loc(fd)
        if pos == 0:
            continue
        prev = px.index[pos - 1]
        # the return realized ON forecast_date uses prev trading day's close
        scored.loc[i, "actual_ret"] = float(np.log(px.loc[fd] / px.loc[prev]))
        scored.loc[i, "scored"] = True

    done = scored[scored["scored"]].copy()
    out = {"total_logged": int(len(log_df)),
           "total_scored": int(len(done)),
           "pending": int(len(log_df) - len(done))}
    if len(done) == 0:
        SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCORE_PATH.write_text(json.dumps(out, indent=2))
        return out

    y = done["actual_ret"].to_numpy()
    mu = done["mu"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[mask], mu[mask]

    out.update({
        "dir_acc": float(np.mean(np.sign(mu) == np.sign(y))),
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "mean_pred": float(np.mean(mu)),
        "mean_actual": float(np.mean(y)),
        "brier_p_up": float(np.mean((done.loc[mask, "p_up"].to_numpy() - (y > 0)) ** 2))
        if "p_up" in done else None,
        "no_trade_rate": float(done["no_trade"].mean()) if "no_trade" in done else None,
        "last_scored_date": str(done["forecast_date"].iloc[-1]),
    })
    # per-expert dir-acc
    for c in done.columns:
        if c.startswith("pred_"):
            p = done[c].to_numpy()
            m = np.isfinite(p) & np.isfinite(y)
            if m.sum() > 5:
                out[f"dir_acc_{c[5:]}"] = float(np.mean(np.sign(p[m]) == np.sign(y[m])))

    # recent 20 scored
    recent = done.tail(20)
    if len(recent) >= 5:
        yr = recent["actual_ret"].to_numpy(); mr = recent["mu"].to_numpy()
        m = np.isfinite(yr) & np.isfinite(mr)
        out["dir_acc_20d"] = float(np.mean(np.sign(mr[m]) == np.sign(yr[m])))

    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORE_PATH.write_text(json.dumps(out, indent=2))
    log.info("scorecard: %d scored, dir_acc=%.3f", len(done), out.get("dir_acc", 0))
    return out


if __name__ == "__main__":
    print(json.dumps(score_pending(), indent=2))
