"""Stage 19 — Position sizing (conformal Kelly) + no-trade filter.

Kelly fraction for a bet with edge mu and variance sigma^2:
    f* = mu / sigma^2        (continuous approximation, fraction of capital)
We shrink it: f = clip(0.25 * f*, -f_max, f_max) — quarter-Kelly is the
standard risk-adjusted choice, and we cap at f_max=20%.

Conformal adjustment: widen sigma by the conformal half-width so sizing
accounts for calibration error.

No-trade filter fires when ANY of:
  - |predicted return| < round-trip cost (~0.26%)
  - model disagreement > threshold (std of expert preds)
  - regime unknown / degraded (kill-switch input)
  - recent 20d directional accuracy < 0.45
Output includes no_trade flag + reason — reported honestly.
"""
from __future__ import annotations

import numpy as np

from src.core.config import load_yaml


def kelly_fraction(mu: float, sigma: float, fraction: float = 0.25,
                   f_max: float = 0.20) -> float:
    """Quarter-Kelly fraction, capped. Sign follows mu."""
    if sigma <= 0 or not np.isfinite(mu) or not np.isfinite(sigma):
        return 0.0
    f_star = mu / (sigma ** 2)
    f = fraction * f_star
    return float(np.clip(f, -f_max, f_max))


def no_trade_check(pred_mu: float, pred_sigma: float,
                   disagreement: float = 0.0,
                   dir_acc_20d: float | None = None,
                   data_stale_days: int = 0,
                   conformal_hw: float | None = None) -> dict:
    """Return sizing decision."""
    cfg = load_yaml("config.yaml")
    costs = cfg["model"]["transaction_costs"]
    rt_cost = 2 * (costs["brokerage_per_side"] + costs["statutory_per_side"]
                   + costs["slippage_per_side"])
    ks = cfg.get("kill_switch", {})

    reasons = []
    # 1. edge must exceed costs
    if abs(pred_mu) < rt_cost:
        reasons.append(f"|mu|={abs(pred_mu):.4f} < cost={rt_cost:.4f}")
    # 2. disagreement
    dis_thr = cfg.get("uncertainty", {}).get("disagreement_threshold", 0.6)
    if disagreement > dis_thr * pred_sigma:
        reasons.append(f"disagreement {disagreement:.4f} > {dis_thr}*sigma")
    # 3. recent degradation
    if dir_acc_20d is not None and dir_acc_20d < ks.get("min_directional_accuracy_20d", 0.45):
        reasons.append(f"20d dir_acc {dir_acc_20d:.3f} < threshold")
    # 4. stale data
    if data_stale_days > ks.get("max_input_staleness_days", 2):
        reasons.append(f"data stale {data_stale_days}d")

    no_trade = len(reasons) > 0
    # sigma inflated by conformal half-width if provided
    sig_eff = pred_sigma
    if conformal_hw is not None:
        sig_eff = float(np.sqrt(pred_sigma ** 2 + conformal_hw ** 2))
    f = 0.0 if no_trade else kelly_fraction(pred_mu, sig_eff)

    return {
        "no_trade": no_trade,
        "reasons": reasons,
        "kelly_fraction": f,
        "kelly_pct": f * 100,
        "effective_sigma": sig_eff,
        "round_trip_cost": rt_cost,
    }
