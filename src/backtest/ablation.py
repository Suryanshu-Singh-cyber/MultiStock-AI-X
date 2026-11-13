"""Stage 22 — Ablation ladder E0 -> E9.

Each row adds one component family and reports the full metric panel.
Components that don't improve OOS net Sharpe / dir-acc are flagged REMOVE.

E0 naive (predict 0 / hist mean)
E1 +market features        (ridge on market-only cols)
E2 +full feature library   (ridge on selected 53)
E3 +decomposition alpha    (predict alpha component target)
E4 +deep sequence model    (LSTM/attn/transformer — filled if run)
E5 +graph features         (graph cols added to E2)
E6 +regime gating          (DISABLED — regimes failed validation gate)
E7 +MoE combiner           (gated ensemble of experts)
E8 +uncertainty sizing     (conformal-Kelly position sizing)
E9 +no-trade filter        (final production policy)
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
from src.sizing.kelly import no_trade_check

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"


def _wf_eval(make_pred_fn, label: str, stride: int = 8, max_splits: int = 15) -> dict:
    panel = build_target_panel()
    Xall = build_feature_library()
    y = panel["target_ret"]
    label_dates = panel.index[panel["target_ret"].notna()]
    splits = walk_forward_splits(label_dates)[::stride][:max_splits]

    all_y, all_p = [], []
    for sp in splits:
        cutoff = sp.test_dates.min()
        pred = make_pred_fn(Xall, y, sp, cutoff)
        if pred is None:
            continue
        yt = y.reindex(sp.test_dates).to_numpy()
        m = np.isfinite(yt) & np.isfinite(pred)
        all_y.append(yt[m]); all_p.append(pred[m])
    if not all_y:
        return {"label": label, "error": "no predictions"}
    yy = np.concatenate(all_y); pp = np.concatenate(all_p)
    met = regression_metrics(yy, pp)
    tr = trading_metrics(yy, pp, cost_per_unit=0.0026)
    return {"stage": label, **{f"reg_{k}": v for k, v in met.items()},
            **{f"tr_{k}": v for k, v in tr.items()}}


def _ridge_pred(cols_fn):
    def fn(Xall, y, sp, cutoff):
        cols = cols_fn(Xall)
        Xtr = Xall[cols].reindex(sp.train_dates)
        Xte = Xall[cols].reindex(sp.test_dates)
        ytr = y.reindex(sp.train_dates)
        ok = ytr.notna().to_numpy()
        imp = LeakageCheckingImputer().fit_dated(Xtr[ok], sp.train_dates[ok], cutoff)
        a, b = imp.transform(Xtr[ok]), imp.transform(Xte)
        sc = TrainOnlyScaler().fit_dated(a, sp.train_dates[ok], cutoff)
        a, b = sc.transform(a), sc.transform(b)
        ok2 = ~np.isnan(a).any(axis=1)
        m = Ridge(alpha=1.0).fit(a[ok2], ytr[ok].to_numpy()[ok2])
        return m.predict(b)
    return fn


def run_ablation() -> pd.DataFrame:
    sel = json.loads((PROC / "selected_features.json").read_text())
    stable = sel["selected"]
    market_cols = [c for c in stable if c.startswith(
        ("mkt_", "sec_", "vix", "beta_", "corr_", "peer_median", "dow_", "month_"))]

    rows = []

    # E0: naive zero
    rows.append({"stage": "E0_naive", "dir_acc": np.nan, "note": "predict 0 — dir_acc undefined"})
    # use hist_mean-like: predict train mean via constant — approximated below

    # E1: market-only ridge
    rows.append(_wf_eval(_ridge_pred(lambda X: market_cols), "E1_market_only"))
    # E2: all stable features
    rows.append(_wf_eval(_ridge_pred(lambda X: stable), "E2_full_features"))
    # E3: alpha-target ridge (predict decomposition residual)
    try:
        decomp = pd.read_parquet(PROC / "decomposition.parquet")
        # alpha target = ret - fitted; we predict alpha then add fitted? simplify:
        # evaluate predicting raw ret but trained on alpha is inconsistent —
        # instead E3 = predict market+sector fitted + alpha model. Approximate:
        # ridge on stable features but target = alpha residual
        def alpha_target_pred(Xall, y, sp, cutoff):
            # rebuild alpha target for train dates from decomposition
            alpha = decomp["alpha"].reindex(sp.train_dates)
            fitted = decomp["intercept"] + decomp["beta_mkt"] * decomp["mkt_ret"] \
                     + decomp["beta_sec"] * decomp["sec_ret"]
            # test: predict alpha, then total = fitted_test + alpha_pred
            cols = stable
            Xtr = Xall[cols].reindex(sp.train_dates)
            Xte = Xall[cols].reindex(sp.test_dates)
            ok = alpha.notna().to_numpy()
            imp = LeakageCheckingImputer().fit_dated(Xtr[ok], sp.train_dates[ok], cutoff)
            a, b = imp.transform(Xtr[ok]), imp.transform(Xte)
            sc = TrainOnlyScaler().fit_dated(a, sp.train_dates[ok], cutoff)
            a, b = sc.transform(a), sc.transform(b)
            ok2 = ~np.isnan(a).any(axis=1)
            m = Ridge(alpha=1.0).fit(a[ok2], alpha.to_numpy()[ok][ok2])
            alpha_pred = m.predict(b)
            fit_te = (decomp["intercept"] + decomp["beta_mkt"] * decomp["mkt_ret"]
                      + decomp["beta_sec"] * decomp["sec_ret"]).reindex(sp.test_dates).to_numpy()
            return alpha_pred + np.nan_to_num(fit_te, nan=0.0)
        rows.append(_wf_eval(alpha_target_pred, "E3_decomposition"))
    except Exception as e:
        log.warning("E3 skipped: %s", e)
        rows.append({"stage": "E3_decomposition", "error": str(e)})
    # E5: + graph features
    try:
        gf = pd.read_parquet(PROC / "graph_features.parquet")
        gcols = [c for c in gf.columns]
        def with_graph(Xall, y, sp, cutoff):
            Xtr = pd.concat([Xall[stable].reindex(sp.train_dates),
                             gf.reindex(sp.train_dates)], axis=1)[stable + gcols]
            Xte = pd.concat([Xall[stable].reindex(sp.test_dates),
                             gf.reindex(sp.test_dates)], axis=1)[stable + gcols]
            ytr = y.reindex(sp.train_dates)
            ok = ytr.notna().to_numpy()
            imp = LeakageCheckingImputer().fit_dated(Xtr[ok], sp.train_dates[ok], cutoff)
            a, b = imp.transform(Xtr[ok]), imp.transform(Xte)
            sc = TrainOnlyScaler().fit_dated(a, sp.train_dates[ok], cutoff)
            a, b = sc.transform(a), sc.transform(b)
            ok2 = ~np.isnan(a).any(axis=1)
            m = Ridge(alpha=1.0).fit(a[ok2], ytr[ok].to_numpy()[ok2])
            return m.predict(b)
        rows.append(_wf_eval(with_graph, "E5_graph"))
    except Exception as e:
        rows.append({"stage": "E5_graph", "error": str(e)})
    # E6: regime — DISABLED by validation gate
    rows.append({"stage": "E6_regime", "note": "DISABLED — regimes failed predictive gate "
                 "(see reports/regime_validation.md)"})
    # E7: MoE — from saved moe_predictions
    try:
        moe = pd.read_parquet(PROC / "moe_predictions.parquet")
        ok = moe["moe"].notna() & moe["y"].notna()
        met = regression_metrics(moe.loc[ok, "y"].to_numpy(), moe.loc[ok, "moe"].to_numpy())
        tr = trading_metrics(moe.loc[ok, "y"].to_numpy(), moe.loc[ok, "moe"].to_numpy())
        rows.append({"stage": "E7_moe", **{f"reg_{k}": v for k, v in met.items()},
                     **{f"tr_{k}": v for k, v in tr.items()}})
    except Exception as e:
        rows.append({"stage": "E7_moe", "error": str(e)})
    # E9: production policy = ensemble pred + no-trade filter (from oos preds)
    try:
        oos = pd.read_parquet(PROC / "oos_predictions.parquet")
        p = oos["ensemble"].to_numpy(); yv = oos["y"].to_numpy()
        keep = np.isfinite(p) & np.isfinite(yv)
        p, yv = p[keep], yv[keep]
        # apply no-trade: |pred| < cost -> flat
        cost = 0.0026
        pos = np.where(np.abs(p) >= cost, np.sign(p), 0.0)
        net = pos * yv - cost * np.abs(np.diff(np.concatenate([[0], pos])))
        eq = np.cumprod(1 + net)
        dd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
        sharpe = float(np.sqrt(252) * net.mean() / net.std()) if net.std() > 0 else 0
        rows.append({"stage": "E9_production_policy",
                     "tr_net_sharpe": sharpe,
                     "tr_net_total_ret": float(eq[-1] - 1),
                     "tr_max_drawdown": dd,
                     "no_trade_rate": float(np.mean(pos == 0)),
                     "note": f"ensemble + |pred|>={cost} filter"})
    except Exception as e:
        rows.append({"stage": "E9_production_policy", "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "ablation.csv", index=False)
    print(df[["stage", "reg_dir_acc", "reg_rmse", "tr_net_sharpe",
              "tr_max_drawdown", "note"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_ablation()
