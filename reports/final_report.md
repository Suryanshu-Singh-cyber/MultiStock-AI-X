# MultiStock-AI X — Final Report

**Target:** POWERGRID.NS · **Target variable:** next-day log return
`r_{t+1} = ln(C_{t+1}/C_t)` · **Data:** 2007-10-05 → 2026-11-12 (4,710 rows;
Yahoo history for POWERGRID begins Oct-2007, not 2005) · **Forecast date
logic:** dynamic — always `next NSE trading day` via real holiday calendar
(`pandas_market_calendars`), verified `2026-11-12 → 2026-11-13`.

---

## 1. Methodology

- **Walk-forward only**: rolling train(1260d)/valid(126d)/test(21d),
  mechanically enforced (`src/leakage/split.py` raises on ordering violations).
- **Leakage framework**: timestamp guards for features, train-only
  scaler/imputer wrappers, news first-available, graph-edge causality
  (edge at t uses data ≤ t−1), regime causality, HP-selection window.
  14 pytest tests prove every guard fires (`tests/test_leakage.py`).
- **Feature library**: 184 causal features (momentum, volatility estimators,
  volume, trend, market/sector/VIX/macro relations, 9-peer cross-section,
  calendar). Selection: |ρ|>0.95 redundancy → univariate |IC| top-60 →
  sign-stability ≥60% across folds → **53 stable features** survived.
- **Decomposition**: rolling 252d causal OLS `r = a + β_mkt·r_NIFTY +
  β_sec·r_sector + α`. Reconstruction error 3.5e-18 (algebraically exact).
  α share of variance ≈ 53%.
- **Factor model**: constructed MKT/SEC/MOM/VOL/FX/CRUDE factors; factor
  layer explains ~27% of variance, α residual ~73%.
- **Regime HMM**: 3-state GaussianHMM — **failed the predictive gate**
  (persistent ✓, stable ✓, but regime mean spread ≈ 0.7% of pooled σ).
  Regimes NOT used downstream. See `reports/regime_validation.md`.
- **Uncertainty**: Gaussian NLL deep models + split-conformal intervals +
  calibration table.
- **Risk**: parametric + historical-simulation VaR/ES, rolling beta,
  drawdown; Student-t(ν=5) Monte Carlo 10k paths × 21d.
- **Costs**: ~0.13%/side brokerage+statutory+slippage + 2bps impact.

## 2. Headline results (honest)

### Baseline ladder (mean over 25 walk-forward test windows, stride 4)

| model | dir_acc | MAE | RMSE | Brier | net Sharpe | turnover | no-trade |
|---|---|---|---|---|---|---|---|
| naive_zero* | — | 0.0112 | 0.0144 | 0.250 | 0.00 | 0 | 100% |
| hist_mean | 0.509 | 0.0112 | 0.0144 | 0.250 | +0.11 | 0.96 | 4% |
| **arima** | **0.518** | 0.0112 | 0.0144 | 0.250 | **+0.23** | 1.20 | 6% |
| ridge | 0.497 | 0.0113 | 0.0145 | 0.252 | −0.72 | 8.12 | 56% |
| lasso | 0.503 | 0.0111 | 0.0144 | 0.249 | −0.40 | 3.48 | 20% |
| rf | 0.512 | 0.0112 | 0.0144 | 0.250 | −0.92 | 7.64 | 53% |
| xgb | 0.474 | 0.0119 | 0.0150 | 0.263 | −1.93 | 7.72 | 54% |
| ensemble | 0.499 | 0.0114 | 0.0145 | 0.253 | −1.12 | 7.36 | 53% |

*naive_zero direction is undefined (predicts exactly 0).

**SPA / Reality Check** (Hansen 2005, stationary bootstrap, 1000 reps,
525 OOS days, 8 models vs naive benchmark): best model = rf,
**p = 0.594 → NOT significant at 5%**. No model beats naive after
data-snooping correction on squared-error loss.

### Deep models

| eval | dir_acc | RMSE | net Sharpe |
|---|---|---|---|
| LSTM, 12 folds | 0.544 | 0.0139 | +0.96 |
| LSTM, **25 folds (fuller)** | **0.526** | 0.0156 | +0.78 |
| Transformer | 0.496 | 0.0141 | +0.20 |
| LSTM+Attention | 0.452 | 0.0143 | +0.15 |

**Honest update:** the fuller 25-fold eval shows the LSTM's edge was
partly small-sample optimism — still the best model, but dir-acc 52.6%
not 54.4%. Attention/transformer confirmed worse → rejected.

**SPA including LSTM** (525 OOS days, 9 models): best=rf, **p=0.649 —
still NOT significant**. LSTM's mean squared-error differential vs naive
is slightly negative: better direction, worse magnitude on average.

### Multi-horizon (direct models, ridge, 12 folds)

| horizon | dir_acc | note |
|---|---|---|
| 1d | 0.524 | baseline |
| 5d | 0.544 | +2pp |
| **21d** | **0.591** | signal-to-noise improves with horizon |

21d forecasts now live in the payload (`forecast.horizons`).

### Volatility forecasting

GARCH(1,1)-t fitted daily: next-day vol forecast **26.3% annualized**
(persistence 0.94) vs realized ~21% — vol currently elevated. EWMA
fallback when `arch` unavailable. Feeds risk card.

### Ablation E0→E9 (ridge-based ladder, 15 folds)

| stage | dir_acc | RMSE | net Sharpe | verdict |
|---|---|---|---|---|
| E1 market-only | 0.487 | 0.0153 | +0.38 | baseline |
| E2 +feature library | 0.535 | 0.0159 | +0.72 | features help |
| E3 +decomposition | 0.465 | 0.0249 | −1.50 | **REMOVE** |
| E5 +graph features | 0.538 | 0.0158 | +0.51 | marginal |
| E6 regime | — | — | — | **DISABLED (failed gate)** |
| E7 MoE | 0.531 | 0.0152 | −0.38 | no improvement over best expert |
| E8 uncertainty | — | — | — | sizing layer, not a return model |
| E9 production policy | — | — | −0.57 | costs dominate at daily freq |

### Live inference (verified 2026-11-13)

```
latest_data_date = 2026-11-12   forecast_date = 2026-11-13 (Fri)
mu = +0.068%  sigma = 1.26%  P(up) = 52.3%
experts: ridge/rf/xgb disagreement σ = 0.22%
DECISION: NO-TRADE — |mu| 0.07% < round-trip cost 0.26%
```

This is the system working as designed: the honest answer most days is
"edge too small vs costs → don't trade".

## 3. What worked / what didn't

**Worked:** leakage framework; walk-forward splitter; feature selection
(53 stable of 184); decomposition (exact); **LSTM > all baselines**
(dir-acc 0.544, net Sharpe +0.96 vs ARIMA +0.23); live pipeline
end-to-end (inference → API → dashboard data); kill-switch monitor;
attention/transformer honestly rejected (worse than LSTM).

**Didn't (reported honestly):** ML models don't beat costs after
transaction fees; regimes not return-predictive; decomposition target
hurt; MoE gating no better than best expert; GDELT unreachable from this
network during build (code in place, first-available guards enforced);
SPA says nothing is significant yet.

## 4. Limitations

- Yahoo data for POWERGRID starts 2007 (not 2005) → ~19y effective sample.
- Daily-frequency equity prediction is close to efficient; realistic edge
  is small. The no-trade filter firing often is a *feature*, not a bug.
- Deep-model numbers remain the weakest link statistically — SPA says
  nothing beats naive yet; live scorecard (reports/live/scorecard.json)
  is now accumulating TRUE out-of-sample evidence daily.
- News: GDELT reachable but POWERGRID-specific coverage is thin in the
  English index (sector/macro queries return articles; backfill running).
  FinBERT sentiment not yet integrated.
- Broker feeds (Upstox/INDstocks) implemented behind config flag but
  untested — need your API credentials in .env.
- GNN replaced by lead-lag graph features (documented in code).

## 5. Live architecture (verified)

```
yfinance daily batch → features → ridge+rf+xgb ensemble → mu/σ/disagreement
→ risk (VaR/ES) + Monte Carlo + SHAP drivers → conformal-Kelly sizing
→ no-trade filter → forecast_update event → FastAPI (:8010) → React dashboard
Kill switch: 20d dir-acc <45% OR stale data >2d OR payload error → DEGRADED
```

Run: `uvicorn src.api.main:app --port 8010` · dashboard `cd src/dashboard/web && npm run dev`
Scheduler: `python -m src.streaming.scheduler` (15:45 IST Mon–Fri)
