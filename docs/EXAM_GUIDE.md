# MultiStock-AI X — Complete Technical Documentation & Exam Guide

> An evidence-driven, leakage-proof forecasting system for POWERGRID.NS (NSE India).
> Built to answer one question honestly: *"Can we predict tomorrow's return well
enough to beat trading costs — and do we know when NOT to trade?"*

---

## 1. THE BIG PICTURE (30-second pitch)

```
Yahoo prices (15 symbols, ~19 yrs)
        │
        ▼
TARGET: r(t+1) = ln(Close[t+1] / Close[t])        ← next-day log return
        │
        ▼
184 causal features → strict no-leakage filter → 53 selected
        │
        ▼
4-model ensemble: Ridge + RandomForest + XGBoost + LSTM
        │
        ▼
μ (expected return), σ (uncertainty), disagreement
        │
        ├──► Return decomposition: market / sector / alpha
        ├──► Risk: VaR, ES, GARCH vol forecast, Monte Carlo (10k paths)
        ├──► Multi-horizon: 1d / 5d / 21d direct models
        └──► No-trade engine: trade ONLY if edge > 0.26% costs
        │
        ▼
React dashboard (3D toy UI) + FastAPI + WebSocket + scheduler
```

**The honest headline:** directional accuracy ~52–54%, SPA test says nothing
statistically beats naive yet — and the system *says so*. Its real value is
risk honesty + knowing when to sit out.

---

## 2. DATA LAYER

### 2.1 What we ingest
- `src/data/historical.py` — yfinance batch download, OHLCV + Adj Close → `data/raw/*.parquet`
- 15 symbols: target POWERGRID.NS + 9 peers (NTPC, TATAPOWER, ADANIPOWER, JSWENERGY, NHPC, SJVN, ONGC, BPCL, RELIANCE) + indices/macro (^NSEI, ^CNXENERGY, ^INDIAVIX, USDINR=X, CL=F)
- Rolling window: config `end: null` → always "today", never frozen

### 2.2 Target construction (`src/data/target.py`)
- `build_target_panel()` → `data/processed/POWERGRID.NS_panel.parquet` (4,710 rows)
- Target: `r_{t+1} = ln(C[t+1]/C[t])` — **indexed at t** (feature time), verified numerically for all rows
- `latest_available_date()` = max date in cleaned data — computed, never hardcoded
- `forecast_date()` = next NSE trading day via `pandas_market_calendars` (calendar "NSE") — skips weekends AND real holidays
- Last row keeps `target_ret = NaN` by design — that's the live inference row

**WHY indexed at t:** if the target were indexed at t+1, every feature at t+1
would "predict" a return that already happened → silent leakage. We verified
`target_ret[t] == ln(C[t+1]/C[t])` mechanically.

**WHY log returns:** additive across time, symmetric-ish, standard in quant.

**Data reality check:** Yahoo only has POWERGRID from **2007-10-05** (not 2005)
— ~19 years effective history. ^CNXENERGY only from 2011. We document this
instead of pretending.

---

## 3. THE LEAKAGE FRAMEWORK (the gate — most important part)

`src/leakage/` + `tests/test_leakage.py` — **14/14 tests pass**.

### 3.1 Rules enforced mechanically
| Rule | Enforcement |
|---|---|
| Feature time < target time | `assert_feature_before_target()` raises `LeakageError` |
| Train window < test window | `assert_train_before_test()` inside the splitter itself |
| Scaler/imputer fit only on train | `TrainOnlyScaler`, `LeakageCheckingImputer` require a cutoff arg and **raise** if fit data reaches it |
| News timestamps | `assert_news_first_available()` — article counts only from publish time onward |
| Graph edges | `assert_graph_edge_causal()` — edge at t may only use data ≤ t−1 |
| Regime labels | `assert_regime_causal()` — HMM state at t uses returns ≤ t |
| Hyperparameters | `assert_hp_selection_window()` — selection only on validation fold |

### 3.2 Walk-forward splitting (`src/leakage/split.py`)
- `walk_forward_splits()` — rolling train/valid/test windows, **ordering asserted inside the splitter**
- A split that violates time-ordering cannot even be constructed
- NO static 80/20 split anywhere — that leaks regime information

### 3.3 WHY this matters (exam answer)
> "Most '98% accurate' stock predictors fail exactly here. A scaler fit on the
> full dataset leaks future mean/variance into the past. We made leakage
> *impossible to ship silently* — violations raise exceptions, and pytest
> proves each guard actually fires."

---

## 4. FEATURE ENGINEERING (`src/features/`)

- `library.py` builds **184 candidate features**, all strictly ≤ t:
  - Price: returns, lags, momentum (5/10/21/42/63/126/252d), volatility, drawdown, distance-from-52w-high, streaks, MACD, RSI-like, volume ratios
  - Peers: correlations, relative strength vs peer basket, lead-lag features
  - Market: NIFTY returns/beta, sector index (^CNXENERGY), VIX level/change, USDINR, crude
  - Graph features: lead-lag correlation network across symbols (replaces GNN — documented why)
- `selection` (`data/processed/selected_features.json`): stability selection across folds → **53 stable features**
- Scalers/imputers fit **per-fold on train only**

**WHY 184→53:** more features ≠ better; unstable features overfit. Selection is
causal (chosen inside each fold's validation window, never on test).

---

## 5. MODEL LADDER (`src/models/`)

Evaluated strictly walk-forward, all metrics out-of-sample:

| Model | dir-acc | RMSE | net Sharpe (after 0.26% cost) |
|---|---|---|---|
| Naive (0) | ~50% | baseline | ~0 |
| ARIMA | 51.8% | — | +0.23 |
| Ridge | ~50% | good | negative |
| Lasso | ~50% | good | negative |
| RandomForest | ~50% | ok | negative |
| XGBoost | ~50% | ok | negative |
| Ensemble | ~52% | ok | mixed |
| **LSTM (Gaussian NLL)** | **52.6%** (25 folds) | 0.0156 | **+0.78** |
| LSTM+Attention | 45.2% | worse | rejected |
| Transformer | 49.6% | ok | rejected |

- LSTM: sequence model, seq_len=60, predicts (μ, log σ) via Gaussian NLL loss — **outputs uncertainty, not just a point**
- Attention/transformer **rejected** — they made it worse; protocol says remove what doesn't help
- Production ensemble = **equal-weight mean** of ridge/RF/XGB/LSTM + std as disagreement

---

## 6. COMPONENTS THAT WERE TESTED AND HONESTLY REJECTED

This is what makes it research-grade:

| Component | Result | Decision |
|---|---|---|
| HMM market regimes | Persistent & stable, but **not return-predictive** | Disabled (`reports/regime_validation.md`) |
| Return decomposition as *target* | Hurt badly (−1.50 Sharpe) | Kept only as *explanation*, not as target |
| MoE gating | No better than best expert | Not in production path |
| GNN | Replaced by lead-lag graph features (simpler, same info) | Documented |
| SPA test (9 models, 525 OOS days) | best=RF, **p=0.649 → NOT significant** | We do NOT claim alpha |

---

## 7. UNCERTAINTY, RISK & SCENARIOS (`src/uncertainty`, `src/risk`, `src/scenarios`, `src/sizing`)

### 7.1 Uncertainty
- σ combines model residual σ + **expert disagreement** (std of the 4 predictions)
- Conformal-style intervals: p10/p50/p90 = μ ∓ 1.2816σ (80% band)
- P(up) ≈ logistic(μ/σ) — calibrated-ish, honestly approximate

### 7.2 Risk snapshot (`risk_snapshot`)
- VaR95/ES95, VaR99/ES99 — Gaussian AND historical (empirical) versions
- Annualized vol, max drawdown over window

### 7.3 GARCH volatility forecast (`src/risk/garch.py`)
- GARCH(1,1)-t on log returns → next-day vol forecast (currently ~26% ann vs ~21% realized → vol elevated)
- EWMA fallback if `arch` unavailable
- **WHY:** vol clusters — tomorrow's risk is better predicted by a vol model than by assuming constant σ

### 7.4 Monte Carlo (`src/scenarios/engine.py`)
- 10,000 paths × 21 days, **Student-t(ν=5)** daily returns (fat tails — Gaussian understates crashes)
- Outputs: p05/p50/p95 final return, prob_loss, max-drawdown distribution, 24-bin histogram

### 7.5 No-trade engine (`src/sizing`)
```
trade_allowed = |μ| > round_trip_cost (0.26%) AND dir_acc_20d healthy
              AND data fresh AND model disagreement low
```
- ¼-Kelly sizing when trading IS justified
- Kill-switch conditions: stale data, degraded accuracy, high disagreement → status DEGRADED

**WHY no-trade is a feature:** at daily frequency, costs dominate. The correct
trade most days IS nothing. The system quantifies this instead of pretending.

---

## 8. MULTI-HORIZON (`horizons` in payload)

Direct models per horizon (not iterated — avoids error compounding):

| Horizon | dir-acc (research) |
|---|---|
| 1d | 52.4% |
| 5d | 54.4% |
| **21d** | **59.1%** |

**WHY it works:** signal-to-noise ratio improves with horizon — daily noise
drowns the signal, but drift persists over weeks. In payload:
`forecast.horizons = {"5d": ..., "21d": ...}` cumulative expected returns.

---

## 9. RETURN DECOMPOSITION (the 3D tower's data)

`src/streaming/inference.py::_decomposition_split()` — causal approximation:

```
market_comp = β_mkt × E[r_NIFTY tomorrow]     (trailing 504d ridge beta + ridge forecast)
sector_comp = β_sec × E[r_CNENERGY tomorrow]
alpha       = μ_ensemble − market − sector     (stock-specific leftover)
```

Example (Nov-13): market +0.022% · sector +0.026% · alpha +0.021% — roughly
even split. All inputs strictly ≤ latest date.

**WHY:** answers "is this bet about the whole market, the sector, or the
stock itself?" — crucial for understanding whether you have genuine edge.

---

## 10. LIVE SYSTEM (`src/streaming`, `src/api`)

### 10.1 Daily pipeline (`run_daily_inference`)
1. `latest_available_date()` → `forecast_date()` (NSE calendar)
2. Build feature library → take row at `latest`
3. Predict with ridge/RF/XGB (+ LSTM if checkpoint fresh <7d)
4. μ = mean(preds), σ = √(resid_σ² + disagreement²)
5. Risk snapshot + GARCH vol + Monte Carlo + horizons + SHAP drivers + decomposition
6. No-trade check → payload → save `reports/live/{latest_data_date}.json` → publish WS event → log forecast for scoring

### 10.2 Live scoring (`src/streaming/scoring.py`)
- Every forecast appended to `data/live/forecast_log.parquet`
- Next day, `score_pending()` compares prediction vs realized return → `reports/live/scorecard.json`
- Dashboard "Report card" shows TRUE live accuracy — earned, not claimed

### 10.3 API (`src/api/main.py`, FastAPI :8010)
`/health` `/forecast/latest` `/forecast/history` `/price/history` `/backtest/equity` `/scorecard` `/risk/latest` `/drivers/latest` `/forecast/refresh` (POST) `/forecast/refresh/status` `/debug/paths` `/ws/stream`

### 10.4 Scheduler (`src/streaming/scheduler.py`)
- APScheduler, 15:45 IST Mon–Fri → `daily_job()`: ingest (240s cap) → inference → score → publish
- `LAST_ERROR` global surfaces failures to the refresh endpoint

### 10.5 Alerts (`src/streaming/alerts.py`)
- Fires on trade signals, strong directional leans (>65%/<35% P(up)), DEGRADED status
- Webhook-ready via `ALERT_WEBHOOK_URL` (Telegram/Discord), rate-limited 1/hr

---

## 11. DEPLOYMENT

| Piece | Where | How |
|---|---|---|
| Frontend | Vercel | `src/dashboard/web`, Vite build, `VITE_API_URL` → backend |
| Backend | Render | `render.yaml` blueprint, `start_render.py` (data bootstrap → uvicorn), `RUN_SCHEDULER=1` |
| Data | git repo | parquets committed (~5MB); yfinance blocked on Render IPs → data refresh is the known weak point |
| Redis | optional | in-memory bus fallback works fine single-process |

**Honest deploy caveats:** free tier = 512MB RAM (tight with torch — LSTM checkpoint committed to avoid retraining), cold starts ~30-60s, yfinance rate-limits datacenter IPs.

---

## 12. DASHBOARD (`src/dashboard/web`)

React 18 + Vite + Recharts + Framer Motion + React Three Fiber 8 + drei 9.

Sections: sticky nav → cinematic hero (canvas market viz + 3D-tilt verdict card) → AI Brain story (4-stage stepper) → Features → **Toy Box** (R3F exploding tower: market/sector/alpha blocks + confidence sphere + rest-day shield + weather metaphor + Pro Quant toggle) → Command Center (verdict, price+MA chart, equity curve, gauge, range bar, horizons, risk grid, SHAP diverging bars, Monte Carlo histogram, voting room, position decision) → Performance (equity vs buy-hold, live report card, honest research table).

Design: midnight navy / electric blue / violet / emerald / coral; SVG icons not emoji; `prefers-reduced-motion` respected; mobile breakpoints; WebGL fallback.

---

## 13. PROBABLE EXAM QUESTIONS — WITH ANSWERS

**Q: How do you prevent lookahead bias?**
> Mechanical guards that raise exceptions: features must be timestamped < target,
> scalers fit per-fold on train only with cutoff enforcement, splits assert
> ordering internally, news/graph/regime have domain guards. 14 pytest tests
> prove violations fire. It's not a convention — it's enforced in code.

**Q: Why walk-forward and not a random split?**
> Random splits leak regime/macro information across time. Markets are
> non-stationary; only forward-chaining evaluation estimates real deployability.

**Q: Why does the model mostly say NO TRADE?**
> Daily POWERGRID returns are near-efficient; predictable edge (~0.05%) is
> smaller than round-trip costs (0.26%). The honest answer most days is nothing.
> We quantify this rather than hide it — net Sharpe is negative for almost
> every model after costs.

**Q: Is the LSTM actually good?**
> Best model tested: 52.6% dir-acc over 25 folds, +0.78 net Sharpe — but the
> fuller eval showed small-sample optimism (was 54.4% on 12 folds). SPA test
> p=0.649 → NOT statistically proven to beat naive. We report this honestly.

**Q: Why not just use the best backtest model?**
> Because backtest ≠ live. We log every forecast and score it next day
> (scorecard) — real evidence accumulates instead of trusting history.

**Q: Why GARCH for volatility?**
> Volatility clusters (large moves follow large moves). GARCH(1,1)-t captures
> this; current forecast shows elevated vol (~26% ann) — directly feeds VaR bands.

**Q: Why Student-t in Monte Carlo?**
> Financial returns have fat tails — Gaussian simulation understates crashes.
> t(ν=5) gives realistic tail risk for the 21-day horizon distribution.

**Q: What's the weakest part of the system?**
> (1) Statistical edge is unproven — SPA says nothing beats naive yet.
> (2) Data refresh on cloud is blocked by yfinance rate limits.
> (3) ~19y history is shorter than ideal; POWERGRID only exists from 2007.
> (4) Live accuracy needs months of scorecard accumulation to trust.

**Q: Why decomposition if it didn't help as a target?**
> As a *target* it hurt (−1.50 Sharpe) — rejected. As an *explanation layer*
> it answers "market vs sector vs stock" which is essential for trusting
> whether alpha is real. Different purpose, kept honest.

**Q: How does the 3D UI relate to the quant output?**
> Pure translation layer — no logic changes. Tower blocks = decomposition
> values; bubble size = σ; shield = no_trade flag; weather = vol forecast.
> Pro Quant Mode reveals raw μ/σ/p-values for verification.

---

## 14. FILE MAP (where to point when asked)

| Concern | File |
|---|---|
| Target & dates | `src/data/target.py` |
| Leakage guards | `src/leakage/guards.py`, `split.py`, `scaler.py`, `domain_guards.py` |
| Leakage tests | `tests/test_leakage.py` (14 tests) |
| Features | `src/features/library.py`, `core.py` |
| Baselines/deep | `src/models/` (baselines.py, deep.py) |
| Decomposition | `src/decomposition/` + `_decomposition_split` in inference |
| Regime (rejected) | `src/regime/` + `reports/regime_validation.md` |
| Uncertainty/risk/scenarios/sizing | `src/uncertainty/`, `src/risk/`, `src/scenarios/`, `src/sizing/` |
| Backtest/SPA/SHAP | `src/backtest/`, `src/explain/` |
| Live inference | `src/streaming/inference.py`, `scheduler.py`, `scoring.py`, `alerts.py` |
| API/WS | `src/api/main.py`, `src/streaming/bus.py` |
| Dashboard | `src/dashboard/web/src/` (App, components/*) |
| Deploy | `render.yaml`, `start_render.py`, `docker/`, `vercel.json` |
| Reports | `reports/final_report.md`, `reports/regime_validation.md` |

---

## 15. CURRENT LIVE FORECAST (as of last run)

```
forecast_date : 2026-11-13
mu            : +0.049%  (tiny positive)
σ             : 1.26% daily
P(up)         : ~52%
P10/P50/P90   : -1.54% / +0.05% / +1.66%
decision      : NO TRADE (edge 0.05% < cost 0.26%)
decomposition : market +0.022% | sector +0.026% | alpha ~0%
GARCH vol     : ~26% annualized (elevated)
```

**The one-line summary for the professor:**
> "We built a leakage-proof, walk-forward, cost-aware forecasting pipeline
> that predicts POWERGRID's next-day return with calibrated uncertainty —
> and it honestly reports that the predictable edge is smaller than trading
> costs, so it recommends sitting out most days. That's not a failure;
> that's what a truthful quant system does."
