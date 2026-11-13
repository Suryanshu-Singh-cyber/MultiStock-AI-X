# MultiStock-AI X — Deep-Dive Exam Notes (Formulas, Models, Workflow, Examples)

> Companion to `EXAM_GUIDE.md`. This file goes one level deeper: every formula,
> every model's objective, the exact evaluation workflow, and worked numerical
> examples using real values from the project.

---

## PART A — THE MATH, FORMULA BY FORMULA

### A.1 Target definition

```
r(t+1) = ln( Close[t+1] / Close[t] )
```

- Natural log → continuously-compounded return; additive over time
- **Indexing:** stored at row `t`, so features at `t` predict `r(t+1)`
- Verified numerically on all 4,709 rows: `target_ret[t] == ln(C[t+1]/C[t])`

**Worked example:** Close[t]=₹259.45 (Nov-12), Close[t+1]=₹261.80
→ r = ln(261.80/259.45) = ln(1.009056) ≈ **+0.00901** (+0.90%)

### A.2 Ensemble prediction

```
μ      = mean( preds_ridge, preds_rf, preds_xgb, preds_lstm )
disagr = std( preds_ridge, preds_rf, preds_xgb, preds_lstm )   ← expert spread
σ      = sqrt( σ_residual² + disagr² )
```

**Real values (Nov-13 run):**
```
ridge=+0.00382  rf=−0.00105  xgb=−0.00073  lstm=+0.00071
μ     = +0.000686  (≈ +0.069%)
disagr= 0.00193
decision inputs all below
```

σ combines *model noise* (residual std on train) + *model disagreement* —
two independent uncertainty sources added in quadrature.

### A.3 Probability of rising

```
P(up) ≈ σ_logistic( μ/σ × 1.7 )  =  1 / (1 + exp(−μ/σ × 1.7))
```

Approximation of a probit; honest — it is NOT claimed to be calibrated.
With μ=0.00069, σ=0.01256 → P(up)=52.3% — barely better than a coin flip.

### A.4 Prediction interval (80% band)

```
p10 = μ − 1.2816·σ      p50 = μ      p90 = μ + 1.2816·σ
```

Real: p10 = −1.54%, p50 = +0.07%, p90 = +1.68% — the band is ~24× wider than μ.
**That ratio IS the finding:** the signal is tiny vs the noise.

### A.5 Ridge regression (one baseline)

```
min_w  Σ_i (y_i − w·x_i)²  +  α·||w||²
```

L2 penalty shrinks coefficients → stable on correlated features (our 53
features are highly collinear: momentum at different windows correlate).
Closed-form solution: `w = (XᵀX + αI)⁻¹Xᵀy`. Fast, deterministic, hard to
overfit — the "boring but honest" baseline.

### A.6 Random Forest / XGBoost (nonlinear baselines)

- **RF:** bagged decision trees, each on bootstrap sample + random feature
  subset; prediction = average. Variance reduction through decorrelated trees.
- **XGBoost:** gradient boosting — adds trees sequentially, each fitting the
  *residuals* of the ensemble: `F_m = F_{m−1} + η·h_m(x)`, with
  regularization `Ω(h) = γT + ½λ||w||²` on leaf weights.
- Both capture nonlinearities (e.g., "vol high AND momentum negative → dip")
  that ridge cannot.

### A.7 LSTM with uncertainty output (the best model)

Architecture: `input seq (60 days × n_features) → LSTM(H=64) → heads → (μ_norm, log σ_norm)`

```
Loss = Gaussian NLL:
  L = ½ · [ (y − μ)² / σ²  +  2·log σ  +  log(2π) ]
```

**Why this loss matters (exam gold):** a plain MSE loss trains the model to
output the *conditional mean* and treats all days equally. NLL forces the
network to ALSO learn σ — days it can't predict get wide σ, confident days
get narrow σ. The model outputs a **distribution**, not a point.

Training config (real): seq_len=60, hidden=64, epochs=15, batch=128, lr=1e-3,
~4y window (TRAIN_DAYS=1000 label-eligible days), Adam optimizer.

### A.8 GARCH(1,1)-t volatility model

```
r_t      = μ + ε_t ,   ε_t = σ_t·z_t ,   z_t ~ Student-t(ν)
σ_t²     = ω + α·ε²_{t−1} + β·σ²_{t−1}
```

- α = reaction to last shock, β = memory of past vol
- **Persistence = α + β ≈ 0.94** (our fitted value) → vol clusters strongly
- Forecast: iterate σ² forward h days; mean-reverts to unconditional variance
- Student-t innovations → fat tails in vol-of-vol too

**Why it beat naive σ:** realized vol isn't constant — after a volatile week
tomorrow is more likely volatile. GARCH captures that; constant-σ doesn't.

### A.9 VaR / Expected Shortfall

```
VaR_α  = quantile_α( return distribution )         e.g. VaR95 = 5th percentile
ES_α   = E[ return | return ≤ VaR_α ]               mean of the tail beyond VaR
```

We report BOTH Gaussian and historical (empirical) versions — honest because
the Gaussian version understates real tails (kurtosis > 3 in stock returns).

### A.10 Monte Carlo (10,000 futures)

```
for path in 1..10000:
    for day in 1..21:
        r ~ μ + σ · t_ν(standardized)     # Student-t ν=5, fat tails
        path *= (1 + r)
report quantiles of final returns + max drawdown per path
```

**Why Student-t not Gaussian:** equity returns have excess kurtosis —
crashes happen more often than normal theory predicts. t(5) has heavier
tails; P(|t|>3) ≈ 2% vs Gaussian ≈ 0.3%.

### A.11 Kelly criterion + our no-trade filter

```
f*  = μ / σ²                    (continuous Kelly — fraction of capital)
f   = clip( 0.25·f*, −0.20, +0.20 )     ← quarter-Kelly, capped
```

Quarter-Kelly is standard: full Kelly maximizes log-growth but has brutal
drawdowns; quarter-Kelly sacrifices ~25% growth for ~75% less variance.

**No-trade fires if ANY:**
```
|μ| < round_trip_cost (0.26%)          ← edge doesn't cover fees
OR disagreement > 0.6·σ                ← experts argue → don't trust
OR 20d dir_acc < 0.45                  ← recent degradation
OR data_stale > 2 days                 ← inputs unreliable
```

Worked example (Nov-13): |μ|=0.00069 < 0.0026 → reason recorded →
`no_trade=True` → sizing = ₹0. **Correct and honest.**

### A.12 Return decomposition

```
market_comp = β_mkt · E[r_NIFTY tomorrow]     β from trailing 504d covariance
sector_comp = β_sec · E[r_CNENERGY tomorrow]
alpha       = μ_ensemble − market − sector
```

β = Cov(r_stock, r_index)/Var(r_index) — rolling OLS beta.
Index forecasts come from tiny ridge models on index lag features.
Everything ≤ latest date → causal.

### A.13 Evaluation metrics (what "good" means)

| Metric | Formula | Our best |
|---|---|---|
| Directional accuracy | mean( sign(pred)==sign(actual) ) | LSTM 52.6% |
| RMSE | sqrt(mean((y−ŷ)²)) | 0.0156 |
| MAE | mean(\|y−ŷ\|) | — |
| Brier score | mean((p_up − 1{y>0})²) | ~0.25 |
| Sharpe (net) | mean(net_ret)/std(net_ret)·√252 | +0.78 LSTM |
| SPA p-value | see below | 0.649 |

### A.14 SPA test (Superior Predictive Ability)

Tests H0: "no model beats the benchmark" against H1: "at least one does",
correcting for data-snooping across k models:

```
For each model k:  d_k,t = loss_benchmark,t − loss_k,t   (loss diff per day)
SPA statistic = max_k [ sqrt(T)·mean(d_k) / ω_k ]         (studentized)
Bootstrap (stationary) the d_k series → p-value
```

Stationary bootstrap preserves serial correlation in returns — critical,
because daily returns are autocorrelated and naive i.i.d. bootstrap would
understate variance → anti-conservative p-values.

**Our result: p=0.649 → fail to reject H0.** Honest conclusion: nothing
proven yet. This is the difference between a research system and a scam.

### A.15 SHAP explainability

TreeSHAP computes each feature's marginal contribution:

```
φ_i = E[f(x) | x_i present] − E[f(x) | x_i absent]   (Shapley value)
Σ φ_i + base_value = prediction
```

For trees it uses the polynomial-time exact algorithm over tree paths.
We take top-10 |φ| → the "drivers" card. φ is in return units →
"tgt_ret_lag1 = −0.081%" literally means that feature pushed the forecast
down by 0.081 percentage points.

---

## PART B — THE EVALUATION WORKFLOW (step by step, with real dates)

### B.1 Walk-forward split mechanics

```
Full history: 2007-10 → 2026-11 (~4,700 trading days)

Fold structure (rolling):
  [====TRAIN====][--VALID--][TEST]
        └─ shift forward ─┘   └─ shift ─┘
  repeated ~25 folds, stride ~21 days
```

**Concrete example of one fold:**
```
Fold k:
  train : 2007-10-05 → 2022-03-31   (~3,600 days)
  valid : 2022-04-01 → 2022-09-30   (for early stopping / feature sanity)
  test  : 2022-10-03 → 2023-03-31   (~120 days, NEVER touched in training)
Fold k+1: everything shifts ~21 days forward
```

Inside each fold:
1. Fit imputer + scaler **on train rows only** (cutoff enforced)
2. Train model on train
3. Predict on test → collect (pred, actual) pairs
4. Next fold — no overlap of test windows

After all folds: concatenate OOS predictions → `oos_predictions.parquet`
(525+ days for baselines, ~1,050 for the 25-fold LSTM) → compute all metrics
on this pooled OOS set → run SPA across model losses.

### B.2 Why this is the correct workflow

- **No random split:** would leak future regime into past
- **Retraining each fold:** simulates real deployment (you'd retrain weekly)
- **Pooled OOS:** every metric is genuinely out-of-sample
- **SPA afterwards:** corrects for trying many models (multiple comparisons)

### B.3 Production inference (daily, live)

```
15:45 IST (after NSE close 15:30):
  1. ingest_all()          — fresh yfinance prices (240s cap; Render IP rate-limited → falls back)
  2. build_target_panel()  — latest_available_date = max(clean data)
  3. forecast_date()       — next NSE trading day (calendar-aware)
  4. build_feature_library() — 184 features at row `latest`
  5. load prod models      — ridge/RF/XGB refit on last 1000 label-days
                             LSTM checkpoint if <7 days old
  6. predict 4 experts → μ, σ, disagreement
  7. risk_snapshot + GARCH + Monte Carlo(10k) + horizons + SHAP + decomposition
  8. no_trade_check → payload → save JSON → WS publish → log for tomorrow's scoring
```

### B.4 Live scoring loop (the trust mechanism)

```
Day T:   forecast logged {date, μ, p_up, model_preds, no_trade, ...}
Day T+1: score_pending() reads realized r(T+1), appends outcome
         → scorecard.json: total_scored, dir_acc, brier, no_trade_rate, per-expert acc
```

After ~20 trading days you get the FIRST honest live accuracy number —
not a backtest claim. Currently accumulating (0 scored at last check).

---

## PART C — CONCEPTUAL Q&A (deeper than EXAM_GUIDE)

**Q: Why is directional accuracy only ~52-54% — is that even useful?**
> Daily equity returns are ~90% noise. 52-54% direction + correct uncertainty
> + cost filter IS the realistic ceiling for daily single-stock prediction.
> The edge isn't in being right often — it's in (a) knowing the uncertainty
> honestly and (b) only acting when edge>costs. Multi-horizon helps: 21d
> reaches 59% because signal/noise improves with horizon.

**Q: Why does XGBoost/RF have ~50% direction but negative Sharpe?**
> They predict magnitude poorly (RMSE worse than ridge) — sign accuracy
> without magnitude edge → position sizing wrong → costs eat the tiny edge.
> That's why ensemble + no-trade matters more than any single model.

**Q: What's the difference between VaR and ES?**
> VaR95 = "the loss threshold not exceeded 95% of days" — a single quantile.
> ES95 = "the AVERAGE loss given you're in the worst 5%" — always worse than
> VaR, captures tail severity. Regulators (Basel) prefer ES because VaR
> says nothing about how bad the tail gets.

**Q: Why does vol forecast matter if we predict returns?**
> Position sizing and VaR depend on σ, not just μ. GARCH tells us tomorrow's
> σ is ~1.65%/day (26% ann) vs unconditional ~1.3% — sizing and risk bands
> widen appropriately. Vol clustering is one of the most robust empirical
> facts in finance.

**Q: Why is the LSTM's σ output important?**
> NLL loss makes it a *distributional* forecaster — it learns per-day
> uncertainty, not just mean. Combined with disagreement σ, our bands are
> honest. A point-prediction model would give fake confidence.

**Q: What would you do with more time/data?**
> (1) Options/derivatives data for implied vol (better σ than GARCH).
> (2) Intraday data → more observations per day.
> (3) Longer LSTM OOS + full SPA across all folds.
> (4) News sentiment actually working (GDELT coverage for POWERGRID is thin).
> (5) Multi-asset portfolio version — diversification is where real Sharpe lives.

**Q: Biggest weakness?**
> Statistical significance. SPA p=0.649 → we cannot yet claim the models beat
> naive. The system is honest about this — but a professor should hear it
> first, not discover it.

---

## PART D — NUMBERS TO MEMORIZE

```
History          : 2007-10-05 → 2026-11-12  (~4,710 trading days, 19y)
Symbols          : 15 (1 target + 9 peers + 5 macro/index)
Features         : 184 candidates → 53 stable selected
Models           : ridge, RF, XGBoost, LSTM(seq60,h64,ep15,lr1e-3)
Cost             : 0.26% round-trip (brokerage+statutory+slippage, both sides)
Current forecast : μ=+0.049%, σ=1.26%, P(up)=52.3%, NO TRADE
Decomposition    : market +0.022% | sector +0.026% | alpha ~0%
GARCH            : persistence 0.94, next-day vol ~26% ann (elevated)
Horizons dir-acc : 1d=52.4%  5d=54.4%  21d=59.1%
Best OOS model   : LSTM 52.6% dir-acc, +0.78 net Sharpe (25 folds)
SPA              : p=0.649 → NOT significant (honest!)
Tests            : 14/14 leakage guards pass
```
