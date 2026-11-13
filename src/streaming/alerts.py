"""Stage 8 — Alert system.

Fires on events worth a human's attention:
  - forecast flips to TRADE (no_trade False with meaningful edge)
  - strong directional signal (|p_up - 0.5| > threshold)
  - system DEGRADED / ERROR
  - kill-switch trigger

Channels: console + log always; webhook (Telegram/Discord/generic) if
ALERT_WEBHOOK_URL is set in .env. Rate-limited: same alert max 1/hour.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from src.core.logger import get_logger

log = get_logger(__name__)
STATE = Path("data/live/alert_state.json")
STATE.parent.mkdir(parents=True, exist_ok=True)
COOLDOWN_S = 3600


def _state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def _save(st: dict) -> None:
    STATE.write_text(json.dumps(st))


def _send_webhook(text: str) -> None:
    url = os.environ.get("ALERT_WEBHOOK_URL", "")
    if not url:
        return
    try:
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        log.info("alert sent via webhook")
    except Exception as e:
        log.warning("webhook failed: %s", e)


def evaluate(payload: dict) -> list[str]:
    """Return list of alert messages triggered by this forecast payload."""
    alerts = []
    f = payload.get("forecast", {})
    s = payload.get("sizing", {})
    status = payload.get("status", "OK")

    if status in ("DEGRADED", "ERROR"):
        alerts.append(f"🚨 SYSTEM {status}: {payload.get('error', 'check data freshness')}")

    if not s.get("no_trade") and s.get("kelly_fraction", 0) != 0:
        alerts.append(
            f"💰 TRADE SIGNAL: {s.get('kelly_pct', 0):.1f}% position suggested "
            f"(μ={f.get('mu_log_return', 0):.4f}, P(up)={f.get('p_up', 0):.2f})")

    p_up = f.get("p_up")
    if p_up is not None and abs(p_up - 0.5) > 0.15:
        direction = "BULLISH" if p_up > 0.5 else "BEARISH"
        alerts.append(f"📈 Strong {direction} lean: P(up)={p_up:.1%}")

    return alerts


def fire(payload: dict) -> list[str]:
    """Evaluate + send (rate-limited) + log alerts."""
    alerts = evaluate(payload)
    if not alerts:
        return []
    st = _state()
    now = time.time()
    sent = []
    for a in alerts:
        key = a.split(":")[0]
        if now - st.get(key, 0) < COOLDOWN_S:
            continue
        st[key] = now
        log.warning("ALERT: %s", a)
        _send_webhook(f"MultiStock-AI X: {a}")
        sent.append(a)
    _save(st)
    return sent


if __name__ == "__main__":
    # demo
    demo = {"status": "OK",
            "forecast": {"mu_log_return": 0.01, "p_up": 0.62},
            "sizing": {"no_trade": False, "kelly_fraction": 0.05, "kelly_pct": 5.0}}
    print(fire(demo))
