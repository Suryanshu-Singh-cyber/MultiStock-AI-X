"""Stage 6b — Multi-horizon forecasts: 1d / 5d / 21d log returns.

Direct multi-horizon models (not iterated 1-day) on the selected features.
Quick walk-forward eval with Ridge — reports whether longer horizons carry
more predictable signal (they usually do: signal-to-noise improves).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.core.config import ROOT
from src.core.logger import get_logger
from src.data.target import build_target_panel
from src.features.library import build_feature_library
from src.leakage.scaler import LeakageCheckingImputer, TrainOnlyScaler
from src.leakage.split import walk_forward_splits
from src.models.metrics import regression_metrics, trading_metrics

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"
HORIZONS = {"1d": 1, "5d": 5, "21d": 21}


def horizon_target(close: pd.Series, h: int) -> pd.Series:
    """log(C_{t+h}/C_t) stored at t — same causal convention as 1d target."""
    return np.log(close.shift(-h) / close)


def run_horizons(stride: int = 8, max_splits: int = 12) -> pd.DataFrame:
    panel = build_target_panel()
    Xall = build_feature_library()
    sel = json.loads((PROC / "selected_features.json").read_text())
    cols = sel["selected"][:30]
    close = panel["Close"]
    label_dates = panel.index[panel["target_ret"].notna()]
    splits = walk_forward_splits(label_dates)[::stride][:max_splits]

    rows = []
    for hname, h in HORIZONS.items():
        y_all = horizon_target(close, h)
        oos_y, oos_p = [], []
        for sp in splits:
            cutoff = sp.test_dates.min()
            Xtr = Xall[cols].reindex(sp.train_dates)
            Xte = Xall[cols].reindex(sp.test_dates)
            ytr = y_all.reindex(sp.train_dates)
            ok = ytr.notna().to_numpy()
            # need h days of future for labels — test rows near the end may be NaN
            imp = LeakageCheckingImputer().fit_dated(Xtr[ok], sp.train_dates[ok], cutoff)
            a, b = imp.transform(Xtr[ok]), imp.transform(Xte)
            sc = TrainOnlyScaler().fit_dated(a, sp.train_dates[ok], cutoff)
            a, b = sc.transform(a), sc.transform(b)
            ok2 = ~np.isnan(a).any(axis=1)
            m = Ridge(alpha=1.0).fit(a[ok2], ytr.to_numpy()[ok][ok2])
            p = m.predict(b)
            yt = y_all.reindex(sp.test_dates).to_numpy()
            mask = np.isfinite(yt) & np.isfinite(p)
            oos_y.append(yt[mask]); oos_p.append(p[mask])
        if not oos_y:
            continue
        yy = np.concatenate(oos_y); pp = np.concatenate(oos_p)
        met = regression_metrics(yy, pp)
        # direction accuracy is the headline for multi-horizon
        rows.append({"horizon": hname, "h_days": h, "n": len(yy),
                     "dir_acc": met["dir_acc"], "mae": met["mae"], "rmse": met["rmse"],
                     "mean_abs_pred": float(np.mean(np.abs(pp)))})
        log.info("%s: dir_acc=%.4f rmse=%.5f", hname, met["dir_acc"], met["rmse"])
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "horizon_results.csv", index=False)
    return df


if __name__ == "__main__":
    print(run_horizons().to_string(index=False))
