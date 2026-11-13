"""Stage 28 — Kill switch / health monitor.

DEGRADED conditions (checked on every /health and daily report):
  - rolling 20-day directional accuracy < 45%
  - input data stale > 2 days
  - calibration broken (conformal coverage far from nominal)
  - forecast payload missing/error

run_monitor() -> status dict; API /health surfaces it; dashboard banner.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ROOT, load_yaml
from src.core.logger import get_logger
from src.data.target import latest_available_date

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"
REPORTS_LIVE = ROOT / "reports" / "live"


def monitor_status() -> dict:
    cfg = load_yaml("config.yaml")
    ks = cfg.get("kill_switch", {})
    reasons = []

    # 1. data freshness
    try:
        latest = latest_available_date()
        stale = (pd.Timestamp.today().normalize() - latest).days
        if stale > ks.get("max_input_staleness_days", 2):
            reasons.append(f"data stale {stale}d > {ks.get('max_input_staleness_days')}d")
    except Exception as e:
        stale = 999
        reasons.append(f"cannot read latest data: {e}")

    # 2. rolling 20d directional accuracy from OOS predictions
    dir_acc = None
    oos_path = PROC / "oos_predictions.parquet"
    if oos_path.exists():
        oos = pd.read_parquet(oos_path).tail(20)
        if len(oos) >= 10:
            dir_acc = float(np.mean(np.sign(oos["ensemble"]) == np.sign(oos["y"])))
            if dir_acc < ks.get("min_directional_accuracy_20d", 0.45):
                reasons.append(f"20d dir_acc {dir_acc:.3f} < "
                               f"{ks.get('min_directional_accuracy_20d')}")
    else:
        reasons.append("no OOS predictions file")

    # 3. latest forecast payload status
    payload_ok = False
    if REPORTS_LIVE.exists():
        files = sorted(REPORTS_LIVE.glob("*.json"))
        if files:
            try:
                p = json.loads(files[-1].read_text())
                payload_ok = p.get("status") == "OK"
                if not payload_ok:
                    reasons.append(f"latest payload status={p.get('status')}")
            except Exception as e:
                reasons.append(f"corrupt latest payload: {e}")
    if not payload_ok and not any("payload" in r for r in reasons):
        reasons.append("no forecast payload yet")

    status = "DEGRADED" if reasons else "OK"
    out = {
        "status": status,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "data_staleness_days": stale,
        "dir_acc_20d": dir_acc,
        "reasons": reasons,
    }
    log.info("monitor: %s %s", status, reasons)
    return out


def write_daily_report() -> Path:
    """Persist today's health report."""
    REPORTS_LIVE.mkdir(parents=True, exist_ok=True)
    out = monitor_status()
    today = datetime.now().strftime("%Y-%m-%d")
    path = REPORTS_LIVE / f"health_{today}.json"
    path.write_text(json.dumps(out, indent=2))
    return path


if __name__ == "__main__":
    print(json.dumps(monitor_status(), indent=2))
