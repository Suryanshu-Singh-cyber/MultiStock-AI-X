"""Stage 20 — Realistic backtest with Indian-market costs.

Costs per side: brokerage 0.03% + statutory ~0.05% + slippage ~0.05% ≈ 0.13%.
Round trip ≈ 0.26% + market impact for size (we model flat 0.02% impact
above ₹10L notional — documented approximation).

Strategy: position from forecast via sizing rule (sign or kelly), daily
rebalance, costs on turnover. Reports Sharpe/Sortino/maxDD/turnover/net.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_yaml
from src.core.logger import get_logger

log = get_logger(__name__)


def cost_model() -> dict:
    c = load_yaml("config.yaml")["model"]["transaction_costs"]
    per_side = c["brokerage_per_side"] + c["statutory_per_side"] + c["slippage_per_side"]
    return {"per_side": per_side, "round_trip": 2 * per_side,
            "market_impact": 0.0002}  # ~2bps for institutional size


def backtest(predictions: pd.DataFrame, return_col: str = "y",
             cost_per_side: float | None = None,
             sizing: str = "sign",
             kelly_fraction: float = 0.25,
             sigma_col: str | None = None) -> dict:
    """predictions: DataFrame indexed by date with columns [pred, y, (sigma)].

    sizing='sign': long 1.0 when pred>threshold else flat.
    sizing='kelly': fraction = clip(kelly_fraction * pred/sigma^2, ±0.2).
    """
    costs = cost_model()
    cps = cost_per_side if cost_per_side is not None else costs["per_side"]

    df = predictions.dropna(subset=[return_col]).copy()
    if "pred" not in df.columns:
        raise ValueError("predictions needs a 'pred' column")
    y = df[return_col].to_numpy()
    p = df["pred"].to_numpy()

    if sizing == "sign":
        pos = (p > 0).astype(float)
    else:
        sig = df[sigma_col].to_numpy() if sigma_col and sigma_col in df else np.full(len(df), np.nanstd(y))
        f = np.clip(kelly_fraction * p / np.maximum(sig, 1e-6) ** 2, -0.2, 0.2)
        pos = np.where(np.isfinite(f), f, 0.0)

    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    # cost on each unit of position change, both sides already in per_side
    cost = cps * turnover + costs["market_impact"] * turnover
    gross = pos * y
    net = gross - cost

    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    dd = 1 - eq / peak

    def sharpe(x):
        x = x[np.isfinite(x)]
        return float(np.sqrt(252) * x.mean() / x.std()) if len(x) > 2 and x.std() > 0 else 0.0

    downside = net[net < 0]
    sortino = (np.sqrt(252) * net.mean() / downside.std()
               if len(downside) > 2 and downside.std() > 0 else 0.0)

    return {
        "n_days": int(len(net)),
        "gross_sharpe": sharpe(gross), "net_sharpe": sharpe(net),
        "sortino": float(sortino),
        "net_total_return": float(eq[-1] - 1),
        "buy_hold_return": float(np.prod(1 + y) - 1),
        "max_drawdown": float(dd.max()),
        "turnover": float(turnover.sum()),
        "cost_drag": float(cost.sum()),
        "avg_daily_cost": float(cost.mean()),
        "exposure": float(np.mean(np.abs(pos))),
    }


def run_on_oos(model_col: str = "ensemble",
               sizing: str = "sign") -> dict:
    df = pd.read_parquet("data/processed/oos_predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    preds = df.set_index("date")[[model_col, "y"]].rename(columns={model_col: "pred"})
    res = backtest(preds, sizing=sizing)
    res["model"] = model_col
    res["sizing"] = sizing
    log.info("backtest %s/%s: net_sharpe=%.3f net_ret=%.2f%% dd=%.2f%%",
             model_col, sizing, res["net_sharpe"], res["net_total_return"] * 100,
             res["max_drawdown"] * 100)
    return res


if __name__ == "__main__":
    for m in ["arima", "ridge", "rf", "ensemble"]:
        for s in ["sign", "kelly"]:
            run_on_oos(m, s)
