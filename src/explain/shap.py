"""Stage 23 — Explainability: SHAP on the tree ensemble + component attributions.

SHAP TreeExplainer on a final RF/XGB trained on the selected features.
Produces per-date driver contributions for the dashboard 'why' block.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from src.core.config import ROOT
from src.core.logger import get_logger
from src.data.target import build_target_panel
from src.features.library import build_feature_library
from src.leakage.scaler import LeakageCheckingImputer, TrainOnlyScaler

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"


def train_explainer_model(train_dates: pd.DatetimeIndex,
                          cols: list[str] | None = None):
    panel = build_target_panel()
    Xall = build_feature_library()
    y = panel["target_ret"]
    if cols is None:
        sel = PROC / "selected_features.json"
        cols = json.loads(sel.read_text())["selected"][:30] if sel.exists() \
            else list(Xall.columns[:30])
    X = Xall[cols]
    cutoff = train_dates.max() + pd.Timedelta(days=1)
    Xtr, ytr = X.reindex(train_dates), y.reindex(train_dates)
    ok = ytr.notna().to_numpy()
    imp = LeakageCheckingImputer().fit_dated(Xtr[ok], train_dates[ok], cutoff)
    a = imp.transform(Xtr[ok])
    sc = TrainOnlyScaler().fit_dated(a, train_dates[ok], cutoff)
    a = sc.transform(a)
    model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                  min_samples_leaf=20, n_jobs=-1,
                                  random_state=42).fit(a, ytr[ok])
    return model, cols, sc, imp


def explain_latest(model, cols, sc, imp, Xall, date):
    x = imp.transform(Xall[cols].reindex([date]))
    xs = sc.transform(x)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(xs)
    sv = sv[0] if isinstance(sv, list) else sv[0]
    contrib = pd.Series(sv, index=cols).sort_values(key=abs, ascending=False)
    return contrib


if __name__ == "__main__":
    panel = build_target_panel()
    label_dates = panel.index[panel["target_ret"].notna()]
    train_dates = label_dates[-1000:]  # recent ~4y for explainer
    model, cols, sc, imp = train_explainer_model(train_dates)
    Xall = build_feature_library()
    latest = label_dates[-1]
    contrib = explain_latest(model, cols, sc, imp, Xall, latest)
    print(f"SHAP drivers for {latest.date()}:")
    print(contrib.head(15).round(5).to_string())
