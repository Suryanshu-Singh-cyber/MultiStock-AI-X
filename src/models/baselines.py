"""Stage 4 — Baseline ladder. THE FLOOR every later model must beat.

Models (in order of sophistication):
  naive_zero      predict 0 every day
  hist_mean       predict expanding train mean of target_ret (drift)
  arima           ARIMA(1,0,1) on train target_ret, 1-step forecast
  ridge           Ridge on core features (train-only scaled)
  lasso           Lasso on core features
  rf              RandomForestRegressor
  xgb             XGBoostRegressor
  ensemble        equal-weight average of ridge+rf+xgb OOS predictions

Protocol: walk-forward splits, fit on train ONLY, predict test. All scalers/
imputers fitted via leakage-guarded dated methods. OOS predictions saved to
data/processed/oos_predictions.parquet for pooled metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

from src.core.config import ROOT, load_yaml
from src.core.logger import get_logger
from src.data.target import build_target_panel
from src.features.core import build_core_features
from src.leakage.scaler import LeakageCheckingImputer, TrainOnlyScaler
from src.leakage.split import Split, walk_forward_splits
from src.models.metrics import regression_metrics, trading_metrics

log = get_logger(__name__)

PROC = ROOT / "data" / "processed"
STRIDE = 4  # use every 4th test window -> ~40 eval windows, keeps runtime sane
COST = 0.0026  # round-trip ~0.26% (see config.yaml model.transaction_costs)


@dataclass
class ModelSpec:
    name: str
    kind: str  # zero|mean|arima|linear|tree|xgb|ensemble


MODELS = [
    ModelSpec("naive_zero", "zero"),
    ModelSpec("hist_mean", "mean"),
    ModelSpec("arima", "arima"),
    ModelSpec("ridge", "linear"),
    ModelSpec("lasso", "linear"),
    ModelSpec("rf", "tree"),
    ModelSpec("xgb", "xgb"),
    ModelSpec("ensemble", "ensemble"),
]


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=1.0, n_jobs=-1, random_state=42)


def _prep_xy(feats: pd.DataFrame, target: pd.Series, dates: pd.DatetimeIndex):
    X = feats.reindex(dates)
    y = target.reindex(dates)
    return X, y


def run_baselines(stride: int = STRIDE, max_splits: int | None = None) -> pd.DataFrame:
    panel = build_target_panel()
    feats = build_core_features()
    target = panel["target_ret"]

    label_dates = panel.index[panel["target_ret"].notna()]
    all_splits = walk_forward_splits(label_dates)
    splits = all_splits[::stride]
    if max_splits:
        splits = splits[:max_splits]
    log.info("walk-forward: %d total windows, evaluating %d (stride=%d)",
             len(all_splits), len(splits), stride)

    feat_cols = [c for c in feats.columns]
    records = []          # per-split metric rows
    oos = {m.name: [] for m in MODELS}
    oos_dates, oos_y = [], []

    for si, sp in enumerate(splits):
        tr_dates, te_dates = sp.train_dates, sp.test_dates
        cutoff = te_dates.min()  # first transform/eval timestamp

        Xtr_raw, ytr = _prep_xy(feats[feat_cols], target, tr_dates)
        Xte_raw, yte = _prep_xy(feats[feat_cols], target, te_dates)

        # drop train rows with NaN target (shouldn't happen — label_dates filtered)
        ok_tr = ytr.notna().to_numpy()
        Xtr_raw, ytr_v = Xtr_raw[ok_tr], ytr[ok_tr]

        # impute + scale on TRAIN ONLY (leakage-guarded)
        imp = LeakageCheckingImputer().fit_dated(Xtr_raw, tr_dates[ok_tr], cutoff)
        Xtr_i = imp.transform(Xtr_raw)
        Xte_i = imp.transform(Xte_raw)
        keep = Xtr_i.notna().all(axis=1).to_numpy()
        Xtr_i, ytr_v = Xtr_i[keep], ytr_v[keep]

        sc = TrainOnlyScaler().fit_dated(Xtr_i, tr_dates[ok_tr][keep], cutoff)
        Xtr_s = sc.transform(Xtr_i)
        Xte_s = sc.transform(Xte_i)

        preds: dict[str, np.ndarray] = {}

        # --- simple baselines (no fitting) ---
        preds["naive_zero"] = np.zeros(len(te_dates))
        preds["hist_mean"] = np.full(len(te_dates), float(ytr_v.mean()))

        # --- ARIMA(1,0,1) on train returns ---
        try:
            from statsmodels.tsa.arima.model import ARIMA
            mod = ARIMA(ytr_v.to_numpy(), order=(1, 0, 1), trend="c")
            res = mod.fit()
            fc = res.get_forecast(steps=len(te_dates))
            preds["arima"] = fc.predicted_mean
        except Exception as e:
            log.warning("arima failed split %d: %s", si, e)
            preds["arima"] = preds["hist_mean"].copy()

        # --- linear ---
        ridge = Ridge(alpha=1.0).fit(Xtr_s, ytr_v)
        preds["ridge"] = ridge.predict(Xte_s)
        lasso = Lasso(alpha=0.001, max_iter=5000).fit(Xtr_s, ytr_v)
        preds["lasso"] = lasso.predict(Xte_s)

        # --- trees ---
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                   n_jobs=-1, random_state=42).fit(Xtr_s, ytr_v)
        preds["rf"] = rf.predict(Xte_s)
        xgb = _xgb().fit(Xtr_s, ytr_v)
        preds["xgb"] = xgb.predict(Xte_s)

        # --- ensemble: mean of ridge+rf+xgb ---
        preds["ensemble"] = (preds["ridge"] + preds["rf"] + preds["xgb"]) / 3.0

        yte_v = yte.to_numpy()
        sig = float(ytr_v.std())  # residual-scale sigma proxy per split

        for m in MODELS:
            p = np.asarray(preds[m.name], float)
            met = regression_metrics(yte_v, p, sigma=sig)
            tr = trading_metrics(yte_v, p, cost_per_unit=COST)
            records.append({
                "split": si,
                "test_start": str(te_dates[0].date()),
                "test_end": str(te_dates[-1].date()),
                "model": m.name,
                **{f"reg_{k}": v for k, v in met.items()},
                **{f"tr_{k}": v for k, v in tr.items()},
            })
            oos[m.name].append(pd.Series(p, index=te_dates))
        oos_dates.append(te_dates)
        oos_y.append(pd.Series(yte_v, index=te_dates))

        if si % 10 == 0:
            log.info("split %d/%d done", si, len(splits))

    # ---- pooled OOS predictions ----
    oos_df = pd.DataFrame({"date": np.concatenate([d for d in oos_dates]),
                           "y": np.concatenate([y.to_numpy() for y in oos_y])})
    for m in MODELS:
        oos_df[m.name] = np.concatenate([s.to_numpy() for s in oos[m.name]])
    oos_df.to_parquet(PROC / "oos_predictions.parquet")

    res = pd.DataFrame(records)
    res.to_csv(PROC / "baseline_results.csv", index=False)
    return res


def report(res: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per model, sorted by net_sharpe."""
    agg = res.groupby("model").agg(
        dir_acc=("reg_dir_acc", "mean"),
        mae=("reg_mae", "mean"),
        rmse=("reg_rmse", "mean"),
        brier=("reg_brier", "mean"),
        net_sharpe=("tr_net_sharpe", "mean"),
        gross_sharpe=("tr_gross_sharpe", "mean"),
        max_dd=("tr_max_drawdown", "mean"),
        turnover=("tr_turnover", "mean"),
        no_trade=("tr_no_trade_rate", "mean"),
    ).sort_values("net_sharpe", ascending=False)
    print("\n=== BASELINE LADDER (mean over walk-forward splits) ===")
    print(agg.round(4).to_string())
    return agg


if __name__ == "__main__":
    res = run_baselines()
    report(res)
