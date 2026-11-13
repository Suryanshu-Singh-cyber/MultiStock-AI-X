"""Honest evaluation metrics — regression + trading, all out-of-sample."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       sigma: np.ndarray | float | None = None) -> dict:
    """MAE, RMSE, directional accuracy, Brier (Gaussian), and coverage if sigma given."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_true - y_pred
    out = {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "dir_acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }
    if sigma is not None:
        sig = np.asarray(sigma, float)[mask] if isinstance(sigma, np.ndarray) else float(sigma)
        sig = np.maximum(sig, 1e-6)
        z = err / sig
        # Brier for direction via Gaussian CDF
        from scipy.stats import norm
        p_up = norm.cdf(y_pred / sig)  # P(r>0) under N(pred, sig^2)
        out["brier"] = float(np.mean((p_up - (y_true > 0)) ** 2))
        out["pinball90"] = float(np.mean(np.maximum(0.9 * np.abs(err) - 0.1 * err,
                                                 0.1 * np.abs(err) - 0.9 * err) / 0.9))
        out["cover90"] = float(np.mean(np.abs(err) <= 1.645 * sig))
    return out


def trading_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    cost_per_unit: float = 0.0026,
                    threshold: float = 0.0) -> dict:
    """Simple long/flat strategy: position = 1 if pred > threshold else 0.
    Gross & net daily returns, Sharpe (ann.), max drawdown, turnover, no-trade rate.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    r, p = y_true[mask], y_pred[mask]

    pos = (p > threshold).astype(float)
    # cost charged on position changes (entries+exits), per unit notional
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    gross = pos * r
    net = gross - cost_per_unit * turnover

    def sharpe(x):
        x = x[np.isfinite(x)]
        if len(x) < 2 or x.std() == 0:
            return 0.0
        return float(np.sqrt(252) * x.mean() / x.std())

    eq = np.cumprod(1 + net)
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else 0.0
    return {
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "net_total_ret": float(eq[-1] - 1) if len(eq) else 0.0,
        "max_drawdown": dd,
        "turnover": float(turnover.sum()),
        "no_trade_rate": float(1 - pos.mean()),
        "n_trades_days": int(pos.sum()),
    }


def aggregate_split_metrics(per_split: list[dict]) -> pd.DataFrame:
    """Concatenate per-split metric dicts into one row-per-split frame."""
    rows = []
    for s in per_split:
        row = {"split": s.get("split"), "test_start": s.get("test_start"),
               "test_end": s.get("test_end")}
        row.update({k: v for k, v in s.items() if isinstance(v, (int, float))})
        rows.append(row)
    return pd.DataFrame(rows)
