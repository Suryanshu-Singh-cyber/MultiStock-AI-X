"""Daily inference pipeline — produces the forecast payload.

Runs after market close: latest close -> features -> model ensemble ->
mu/sigma/disagreement -> risk -> sizing/no-trade -> forecast_update event.

MODEL CHOICE (honest): the SPA test showed NO model significantly beats
naive on squared error. We therefore ship the ENSEMBLE prediction but the
no-trade filter is EXPECTED to fire most days — that is the correct,
honest behavior for a near-efficient daily series. The dashboard shows
P(up), interval, drivers, and the no-trade signal prominently.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.core.config import ROOT, get_symbols, load_yaml
from src.core.logger import get_logger
from src.data.target import (
    build_target_panel,
    forecast_date,
    latest_available_date,
    load_raw,
)
from src.features.library import build_feature_library
from src.leakage.scaler import LeakageCheckingImputer, TrainOnlyScaler
from src.models.metrics import regression_metrics
from src.risk.engine import risk_snapshot
from src.scenarios.engine import monte_carlo
from src.sizing.kelly import no_trade_check

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"
REPORTS_LIVE = ROOT / "reports" / "live"
REPORTS_LIVE.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = 1000  # ~4y for production model
LSTM_CKPT = ROOT / "models_ckpt" / "lstm_prod.pt"
LSTM_SEQ = 60
LSTM_MAX_AGE_DAYS = 7


def _lstm_frame_cols() -> list[str]:
    """Same 40 features the LSTM was evaluated on (deep.load_model_frame)."""
    sel = PROC / "selected_features.json"
    if sel.exists():
        return json.loads(sel.read_text())["selected"][:40]
    return []


def train_lstm_expert(force: bool = False) -> dict | None:
    """Train production LSTM on recent label dates; cache checkpoint weekly."""
    import torch
    from src.models.deep import _LSTMNet, make_sequences, SEQ_LEN, EPOCHS, BATCH, LR, HIDDEN

    if LSTM_CKPT.exists() and not force:
        age = (datetime.now().timestamp() - LSTM_CKPT.stat().st_mtime) / 86400
        if age < LSTM_MAX_AGE_DAYS:
            return torch.load(LSTM_CKPT, weights_only=False)

    cols = _lstm_frame_cols()
    if not cols:
        return None
    panel = build_target_panel()
    Xall = build_feature_library()
    y = panel["target_ret"]
    label_dates = panel.index[panel["target_ret"].notna()]
    train_dates = label_dates[-TRAIN_DAYS:]

    Xv = Xall[cols].reindex(label_dates).to_numpy(dtype=np.float32)
    yv = y.reindex(label_dates).to_numpy(dtype=np.float32)
    dpos = {d: i for i, d in enumerate(label_dates)}
    tr_pos = np.array([dpos[d] for d in train_dates if d in dpos])

    x_mu = np.nanmean(Xv[tr_pos], axis=0)
    x_sd = np.nanstd(Xv[tr_pos], axis=0) + 1e-6
    y_mu = float(np.nanmean(yv[tr_pos])); y_sd = float(np.nanstd(yv[tr_pos]) + 1e-6)
    Xs = np.nan_to_num((Xv - x_mu) / x_sd)
    ys = (yv - y_mu) / y_sd

    Xtr, ytr = make_sequences(Xs, ys, tr_pos, seq_len=SEQ_LEN)
    if len(Xtr) < 200:
        log.warning("lstm: too few sequences (%d)", len(Xtr))
        return None

    torch.manual_seed(42); np.random.seed(42)
    net = _LSTMNet.build(len(cols), kind="lstm")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(xt)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            ii = perm[b:b + BATCH]
            out = net(xt[ii])
            mu_p, ls_p = out[:, 0], out[:, 1]
            var = torch.exp(2 * ls_p).clamp(min=1e-6)
            loss = (0.5 * ((yt[ii] - mu_p) ** 2 / var + 2 * ls_p + np.log(2 * np.pi))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        log.info("lstm epoch %d/%d loss=%.4f", ep + 1, EPOCHS, float(loss))

    ckpt = {"state_dict": net.state_dict(), "n_features": len(cols),
            "cols": cols, "seq_len": SEQ_LEN, "x_mu": x_mu, "x_sd": x_sd,
            "y_mu": y_mu, "y_sd": y_sd, "kind": "lstm",
            "trained_through": str(train_dates.max().date())}
    torch.save(ckpt, LSTM_CKPT)
    log.info("lstm checkpoint saved -> %s", LSTM_CKPT)
    return ckpt


def predict_lstm(ckpt: dict, asof_date: pd.Timestamp) -> float | None:
    """Predict next-day return using last SEQ_LEN feature rows ending at asof."""
    import torch
    from src.models.deep import _LSTMNet

    cols = ckpt["cols"]
    Xall = build_feature_library()
    # need SEQ_LEN rows ending at asof — take tail of full index
    sub = Xall[cols].loc[:asof_date].tail(ckpt["seq_len"])
    if len(sub) < ckpt["seq_len"]:
        return None
    x = sub.to_numpy(dtype=np.float32)
    x = np.nan_to_num((x - ckpt["x_mu"]) / ckpt["x_sd"])
    net = _LSTMNet.build(len(cols), kind=ckpt["kind"])
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    with torch.no_grad():
        out = net(torch.tensor(x[None]))
        mu_norm = float(out[0, 0])
    return mu_norm * ckpt["y_sd"] + ckpt["y_mu"]


def _selected_cols() -> list[str]:
    sel = PROC / "selected_features.json"
    if sel.exists():
        return json.loads(sel.read_text())["selected"][:30]
    return []


def train_production_models():
    """Fit ridge + rf + xgb on the most recent TRAIN_DAYS label-eligible days.
    Returns dict of fitted models + preprocessing + columns + sigma."""
    from xgboost import XGBRegressor

    panel = build_target_panel()
    Xall = build_feature_library()
    cols = _selected_cols() or list(Xall.columns[:30])
    X = Xall[cols]
    y = panel["target_ret"]
    label_dates = panel.index[panel["target_ret"].notna()]
    train_dates = label_dates[-TRAIN_DAYS:]
    cutoff = label_dates.max() + pd.Timedelta(days=1)

    Xtr, ytr = X.reindex(train_dates), y.reindex(train_dates)
    ok = ytr.notna().to_numpy()
    imp = LeakageCheckingImputer().fit_dated(Xtr[ok], train_dates[ok], cutoff)
    a = imp.transform(Xtr[ok])
    sc = TrainOnlyScaler().fit_dated(a, train_dates[ok], cutoff)
    a_sc = sc.transform(a)
    yv = ytr[ok].to_numpy()

    models = {
        "ridge": Ridge(alpha=1.0).fit(a_sc, yv),
        "rf": RandomForestRegressor(n_estimators=200, max_depth=8,
                                    min_samples_leaf=20, n_jobs=-1,
                                    random_state=42).fit(a_sc, yv),
        "xgb": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=1.0, n_jobs=-1, random_state=42).fit(a_sc, yv),
    }
    resid_std = float(np.std(yv - np.mean([m.predict(a_sc) for m in models.values()], axis=0)))

    # multi-horizon ridge experts (5d, 21d) — direct models, same features
    close = panel["Close"]
    horizons = {}
    for hname, h in (("5d", 5), ("21d", 21)):
        yh = np.log(close.shift(-h) / close).reindex(train_dates)
        okh = yh.notna().to_numpy()
        try:
            mh = Ridge(alpha=1.0).fit(a_sc[okh], yh.to_numpy()[okh])
            horizons[hname] = mh
        except Exception as e:
            log.warning("horizon %s fit failed: %s", hname, e)

    return {"models": models, "cols": cols, "imp": imp, "sc": sc,
            "sigma": resid_std, "trained_through": str(train_dates.max().date()),
            "horizons": horizons}


def _decomposition_split(panel: pd.DataFrame, latest: pd.Timestamp,
                         mu: float) -> dict | None:
    """Approximate causal split of today's mu into market/sector/alpha.

    market_comp = beta_mkt * E[r_NIFTY tomorrow]   (trailing ridge, ~2y)
    sector_comp = beta_sec * E[r_energy tomorrow]
    alpha       = mu - market - sector  (stock-specific leftover)
    All inputs strictly <= latest. Returns None on any failure.
    """
    from sklearn.linear_model import Ridge

    tgt_ret = np.log(panel["Close"]).diff()
    out = {"market": 0.0, "sector": 0.0, "alpha": mu}

    for key, fn in (("market", "IDX_NSEI.parquet"),
                    ("sector", "IDX_CNXENERGY.parquet")):
        p = ROOT / "data" / "raw" / fn
        if not p.exists():
            continue
        ir = np.log(pd.read_parquet(p)["Close"]).diff()
        if ir is None:
            continue
        j = pd.concat([tgt_ret.rename("t"), ir.rename("i")], axis=1,
                      join="inner").dropna().iloc[:-1]  # drop inference row
        j = j.loc[:latest]
        if len(j) < 300:
            continue
        # trailing 504d ridge: index ret -> target ret  (beta)
        w = j.iloc[-504:]
        beta = float(np.cov(w["t"], w["i"])[0, 1] / (np.var(w["i"]) + 1e-9))
        # index next-day forecast from its own lags (tiny ridge)
        ii = ir.dropna()
        feats, yv = [], []
        for k in range(21, len(ii) - 1):
            wv = ii.iloc[k - 21:k + 1]
            feats.append([wv.iloc[-1], wv.iloc[-5:].mean(),
                          wv.iloc[-21:].mean(), wv.std()])
            yv.append(ii.iloc[k + 1])
        if len(feats) < 100:
            continue
        m = Ridge(alpha=1.0).fit(np.array(feats), np.array(yv))
        last = ii.iloc[-21:]
        x_now = np.array([[last.iloc[-1], last.iloc[-5:].mean(),
                           last.iloc[-21:].mean(), last.std()]])
        idx_mu = float(m.predict(x_now)[0])
        comp = beta * idx_mu
        out[key] = comp

    out["alpha"] = mu - out["market"] - out["sector"]
    return {k: float(v) for k, v in out.items()}


def run_daily_inference() -> dict:
    """Full pipeline for 'today'. Returns forecast payload dict."""
    latest = latest_available_date()
    fdate = forecast_date()
    staleness = (pd.Timestamp.today().normalize() - latest).days

    bundle = train_production_models()
    Xall = build_feature_library()
    x_today = Xall[bundle["cols"]].reindex([latest])
    if x_today.isna().all().all():
        raise RuntimeError("no features for latest date — data may be stale")
    xi = bundle["imp"].transform(x_today)
    xs = bundle["sc"].transform(xi)

    preds = {name: float(m.predict(xs)[0]) for name, m in bundle["models"].items()}

    # LSTM expert (best model from Stage 8-10 eval)
    try:
        ckpt = train_lstm_expert()
        if ckpt is not None:
            lp = predict_lstm(ckpt, latest)
            if lp is not None and np.isfinite(lp):
                preds["lstm"] = float(lp)
    except Exception as e:
        log.warning("lstm expert failed: %s", e)

    mu = float(np.mean(list(preds.values())))
    disagreement = float(np.std(list(preds.values())))
    sigma = float(np.sqrt(bundle["sigma"] ** 2 + disagreement ** 2))

    # risk + scenarios
    panel = build_target_panel()
    hist_ret = np.log(panel["Close"]).diff().dropna().iloc[-252:].to_numpy()
    risk = risk_snapshot(mu, sigma, hist_ret)
    try:
        from src.risk.garch import vol_forecast
        risk["vol_forecast"] = vol_forecast(
            np.log(panel["Close"]).diff().dropna())
    except Exception as e:
        log.warning("vol forecast failed: %s", e)
    mc = monte_carlo(mu, sigma, n_paths=10_000, horizon=21)

    # 20d directional accuracy from recent OOS (if available)
    dir_acc_20d = None
    oos_path = PROC / "oos_predictions.parquet"
    if oos_path.exists():
        oos = pd.read_parquet(oos_path).tail(20)
        if len(oos) >= 10:
            dir_acc_20d = float(np.mean(np.sign(oos["ensemble"]) == np.sign(oos["y"])))

    sizing = no_trade_check(mu, sigma, disagreement=disagreement,
                            dir_acc_20d=dir_acc_20d,
                            data_stale_days=staleness)

    # drivers: simple SHAP on RF for the 'why' block
    drivers = {}
    try:
        import shap
        explainer = shap.TreeExplainer(bundle["models"]["rf"])
        sv = explainer.shap_values(xs)[0]
        contrib = pd.Series(sv, index=bundle["cols"]).sort_values(
            key=abs, ascending=False)
        drivers = {k: float(v) for k, v in contrib.head(10).items()}
    except Exception as e:
        log.warning("shap failed: %s", e)

    # return decomposition: market / sector / alpha split of mu (causal,
    # trailing-window ridge forecasts of the two indices + rolling betas)
    try:
        payload_decomp = _decomposition_split(panel, latest, mu)
    except Exception as e:
        log.warning("decomposition failed: %s", e)
        payload_decomp = None

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_data_date": str(latest.date()),
        "forecast_date": str(fdate.date()),
        "data_staleness_days": staleness,
        "decomposition": payload_decomp,
        "forecast": {
            "mu_log_return": mu,
            "sigma": sigma,
            "p_up": float(1 / (1 + np.exp(-mu / max(sigma, 1e-6) * 1.7))),  # approx
            "p10": float(mu - 1.2816 * sigma),
            "p50": mu,
            "p90": float(mu + 1.2816 * sigma),
            "disagreement": disagreement,
            "model_preds": preds,
            "horizons": {
                h: float(m.predict(xs)[0])
                for h, m in (bundle.get("horizons") or {}).items()
            },
        },
        "risk": {k: v for k, v in risk.items()
                 if isinstance(v, (int, float)) or k == "vol_forecast"},
        "monte_carlo_21d": mc,
        "sizing": sizing,
        "drivers": drivers,
        "status": "DEGRADED" if staleness > 2 else "OK",
    }

    # persist daily report + log forecast for live scoring
    REPORTS_LIVE.mkdir(exist_ok=True)
    (REPORTS_LIVE / f"{latest.date()}.json").write_text(json.dumps(payload, indent=2))
    try:
        from src.streaming.scoring import log_forecast
        log_forecast(payload)
    except Exception as e:
        log.warning("forecast logging failed: %s", e)
    try:
        from src.streaming.alerts import fire
        fire(payload)
    except Exception as e:
        log.warning("alerts failed: %s", e)
    log.info("inference done: forecast_date=%s mu=%.5f sigma=%.5f no_trade=%s",
             fdate.date(), mu, sigma, sizing["no_trade"])
    return payload


if __name__ == "__main__":
    p = run_daily_inference()
    print(json.dumps({k: p[k] for k in ["latest_data_date", "forecast_date",
                                        "status"]}, indent=2))
    print("forecast:", {k: round(v, 5) if isinstance(v, float) else v
                        for k, v in p["forecast"].items() if k != "model_preds"})
    print("sizing:", p["sizing"]["no_trade"], p["sizing"]["reasons"])
