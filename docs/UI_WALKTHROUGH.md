# Command Center — Panel-by-Panel Explanation (with your live values)

> Every widget on the dashboard, what it means, where the number comes from,
> and how to explain it in one sentence. Values below are from the real
> 2026-11-13 forecast.

---

## HEADER

**"Updated 2026-11-13 16:22:55 · data through 2026-11-12"**
- `generated_at` = when the pipeline ran (local time)
- `latest_data_date` = the last trading day we have prices for
- **Why they differ by a day:** the forecast made "today" uses yesterday's
  close to predict the *next* trading day. Data through Nov-12 → forecast for Nov-13.
- `data_stale_days = 1` → healthy. If >2 days, a ⚠ STALE warning appears
  and the no-trade filter auto-fires (stale inputs = unreliable forecast).

**"Update Now" button** → `POST /forecast/refresh` → runs the full daily
pipeline in a background thread (~1-2 min): ingest → features → 4 models →
risk → scoring → publish. Button polls `/forecast/refresh/status` until done.

---

## TOMORROW'S CALL (verdict panel)

```
TOMORROW'S CALL · 2026-11-13
NO TRADE
"Best estimate is +0.05%, but trading costs ~0.3% — bigger than the
 expected edge. Sitting out is the disciplined move."
```

| Chip | Value | Meaning |
|---|---|---|
| Expected | +0.05% | μ = ensemble mean of 4 model predictions (log return) |
| P(up) | 51.7% | logistic(μ/σ) — probability tomorrow's close > today's |
| σ daily | 1.26% | combined model noise + expert disagreement |
| Uncertainty | low | disagreement < 0.3% of σ — the 4 models roughly agree |

**The verdict logic (from `sizing/kelly.py`):**
```
no_trade = |μ| < 0.0026  (round-trip cost)
         OR disagreement > 0.6·σ
         OR 20d dir_acc < 0.45
         OR data stale > 2 days
```
Here |μ|=0.0005 < 0.0026 → **NO TRADE**. This is the system's core honest
feature: it refuses to recommend trades where fees exceed expected gain.

---

## PRICE CHART — last 90 days

- Blue area = daily closing prices (POWERGRID.NS)
- Purple dashed = **20-day simple moving average** = mean of last 20 closes
  → smooths noise, shows the trend direction
- Use: "is the stock in an uptrend, downtrend, or range?" — context for
  whether momentum features are tailwinds or headwinds

---

## CHANCE IT RISES TOMORROW (gauge)

```
52% / 51.7%  →  "Essentially a coin flip — the honest answer."
```

- Semicircle gauge, needle animates to P(up)
- Green arc if >50%, red if <50%
- **Key honesty note shown on screen:** "probability, not certainty" —
  52% does NOT mean "will rise"; it means slightly better than even odds.

---

## PREDICTION RANGE (P10 → P50 → P90)

```
Bad case   -1.6%
Best guess +0.05%
Good case   1.7%
```

- Formula: `p10 = μ − 1.2816σ`, `p90 = μ + 1.2816σ` → central 80% interval
- The needle sits at μ (best guess)
- **The visual point:** the band (−1.6% to +1.7%) is ~30× wider than the
  guess (+0.05%). Tomorrow is genuinely uncertain — the chart makes that
  visible instead of hiding it behind a single number.

---

## BEYOND TOMORROW (multi-horizon)

```
Tomorrow (1d)  +0.05%
5 days         +1.22%
21 days        +1.80%
```

- **Separate direct models** per horizon — NOT "multiply tomorrow by 21"
  (that would compound wrongly and ignore mean-reversion)
- Each is a ridge model trained to predict the cumulative return over
  that horizon directly
- Research finding: directional accuracy improves with horizon
  (1d=52.4% → 5d=54.4% → **21d=59.1%**) — signal-to-noise ratio rises
  because daily noise averages out but drift persists
- Interpretation: "if you held for 3 weeks, expected cumulative gain ~+1.8%"
  — before costs, and it assumes you can hold through the drawdowns shown
  in the Monte Carlo panel.

---

## RISK INTELLIGENCE (8 cells)

| Cell | Value | Formula / source |
|---|---|---|
| VaR 95% | −2.0% | 5th percentile of return dist — "1-in-20 bad day loses ~2%" |
| ES 95% | −2.5% | mean of returns *below* VaR95 — "when bad, it's usually ~−2.5%" |
| VaR 99% | −2.9% | 1st percentile — "1-in-100 bad day" |
| ES 99% | −3.3% | tail average beyond VaR99 |
| Vol (annual) | 21.0% | σ_daily × √252 — yearly swing size |
| Max drawdown | 24.4% | worst peak→trough over the window |
| GARCH vol tmrw | 24.2% | GARCH(1,1)-t forecast for next-day vol (annualized) |
| Persistence | 0.94 | α+β — how strongly vol clusters |

**Why two VaR versions exist (in code):** Gaussian AND historical —
stock returns have fat tails, so the empirical version is the honest one.

**GARCH explained simply:** "volatility breeds volatility" — after a wild
week, tomorrow is more likely wild. α+β=0.94 means shocks decay slowly.
Current forecast 24.2% > unconditional 21% → market is in a slightly
more-volatile-than-usual regime.

---

## WHY THE AI THINKS THIS (SHAP drivers)

```
ret lag1        −0.08%   ← yesterday's move was down → tomorrow dips (momentum echo)
corr sector 63  −0.06%   ← 63-day correlation with power sector weakened → sector drag
vs peers 21     +0.05%   ← POWERGRID beat its 9 rivals over 21d → relative strength
macd hist       −0.04%   ← MACD histogram turned negative → momentum fading
mom 63          −0.04%   ← 63-day momentum negative
logmom 42       −0.02%   ← 42-day log momentum slightly negative
vol mom 63      −0.02%   ← volatility rising
dist 52w high   −0.02%   ← still below 52-week high
close loc       +0.02%   ← closing near top of recent range (small positive)
logmom 5        +0.02%   ← very short-term momentum positive (tiny bounce)
```

- **SHAP (SHapley Additive exPlanations):** each bar = that feature's
  marginal contribution to today's μ, in percentage points
- Sum of all φ + base value = the ensemble prediction
- Read it as: "yesterday fell → drag; but we beat our rivals → lift;
  net ≈ +0.05%"
- Feature names are de-jargoned for display (`tgt_ret_lag1` → "ret lag1")

---

## 10,000 SIMULATED FUTURES (Monte Carlo)

```
Bad outcome (5%)   −8.4%     ← p05 of 21-day returns
Median (50%)       +0.8%     ← p50
Good outcome (95%) +10.6%    ← p95
Chance of losing   44.4%     ← P(final < 0)
Typical dip        4.4%      ← median max drawdown across paths
Bad-luck dip       10.2%     ← 95th pct of max drawdown
Histogram: red bars = losing outcomes, green = winning
```

- 10,000 simulated 21-day paths, daily returns ~ Student-t(ν=5) scaled to σ
- **Fat tails** = crashes happen more than Gaussian predicts
- How to read: "over a month, typical outcome +0.8%, but 44% chance of
  finishing down, and a bad-luck path could dip 10% along the way"
- This is why position sizing matters even when direction looks OK

---

## MODEL VOTING ROOM

```
RIDGE   +0.38%  ▲ bullish
RF      −0.10%  ▼ bearish
XGB     −0.15%  ▼ bearish
LSTM    +0.07%  ◆ neutral
ENSEMBLE μ +0.05%   spread 0.21% · agree ✓
```

- 4 models predict **independently**, then simple average → μ
- "spread" = std of the 4 predictions = disagreement (0.21% here)
- Agreement check: spread < 0.3% of σ → "agree ✓"; if models argued wildly
  the verdict would widen uncertainty and tighten no-trade
- **Honest note on the card:** "equal weight average" — we do NOT pretend
  LSTM is smarter; research showed ensemble > any single model after costs

---

## POSITION DECISION

```
₹0 — DO NOTHING
|mu|=0.0005 < cost=0.0026
```

- When no_trade=True: shows ₹0 + the FIRST failing reason
- When justified: shows `kelly_pct` = quarter-Kelly fraction
  `f = clip(0.25 × μ/σ², −20%, +20%)` — the "safe bet" size
- **UI never overrides this** — the frontend displays the engine's decision,
  it cannot turn NO TRADE into BUY (deliberate honesty constraint)

---

## EVIDENCE / PERFORMANCE SECTION

**Model vs buy-and-hold** — green line = strategy equity (net of 0.26%
costs, ARIMA-based sign strategy), grey dashed = just holding POWERGRID:
- Model final: **+21.6%** vs Buy&hold: **+18.8%** over 525 OOS days
- Label says "net of costs" and "past performance ≠ future results" —
  small edge, honestly presented

**Live report card** — currently "being built": every forecast is logged
and scored against the real close the next day. After ~20 trading days
it shows TRUE live directional accuracy — evidence, not backtest claims.

**Research findings table:**
- LSTM 52.6% dir-acc (25-fold) — best model
- Best baseline 52.4% — barely different
- **SPA p=0.649 → NOT significant** — the honest headline: nothing
  statistically beats naive yet
- 21d horizon 59.1% — longer horizons predict better
- 525 OOS days tested

---

## THE 30-SECOND EXPLANATION FOR YOUR PROFESSOR

> "Every number on this page is real and timestamped. The model predicts
> POWERGRID's next-day return using only past data — no leakage, enforced
> by tests. It says +0.05% expected with a huge uncertainty band, the four
> models roughly agree, and the expected gain is smaller than the 0.26%
> trading cost — so it recommends doing nothing. The Monte Carlo shows
> even a 'good' forecast has a 44% chance of losing over 3 weeks. That's
> the honest answer: the edge is small, the uncertainty is large, and the
> system's job is to tell you when the odds don't justify a trade."
