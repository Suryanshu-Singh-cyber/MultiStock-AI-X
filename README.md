# MultiStock-AI X

Evidence-driven next-day return forecaster for **POWERGRID.NS** (NSE India).
Target: `r_{t+1} = ln(Close_{t+1} / Close_t)`. Rolling data window,
dynamic forecast date (next NSE trading day, real holiday calendar).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt                  # core
pip install -r requirements-ml.txt                # torch/xgb/lgbm/etc (Stage 8+)
```

```bash
# Stage 1-2: download data + build target panel
python -m src.data.historical
python -m src.data.target

# Stage 3: leakage tests (must pass before anything else)
pytest tests/ -v

# Stage 4: baseline ladder
python -m src.models.baselines

# Stage 5-7: features, selection, decomposition
python -m src.features.library
python -m src.features.selection
python -m src.decomposition.market_model

# Stage 8-10: deep models (needs requirements-ml.txt)
python -m src.models.deep

# Live inference (Stage 24d): one-shot forecast for next trading day
python -m src.streaming.inference

# API (Stage 25)
uvicorn src.api.main:app --port 8010

# Scheduler (daily 15:45 IST)
python -m src.streaming.scheduler

# Dashboard (React): cd src/dashboard/web && npm install && npm run dev
```

## Layout

- `configs/` — config.yaml, symbols.yaml, live.yaml (secrets via .env)
- `src/data/` — yfinance loader, target builder, realtime feed, news
- `src/features/` — core + 184-feature library + selection engine
- `src/leakage/` — guards, walk-forward splitter, train-only scalers
- `src/models/` — baselines, deep (LSTM/attn/transformer), factor, graph, SPA
- `src/decomposition/` market+sector+alpha; `src/regime/` validated HMM
- `src/moe/` gating; `src/uncertainty/` conformal; `src/risk/`; `src/scenarios/`
- `src/sizing/` conformal-Kelly + no-trade; `src/backtest/` costs + ablation
- `src/streaming/` inference pipeline, event bus, scheduler, monitor
- `src/api/` FastAPI; `src/dashboard/web/` React+Vite
- `reports/` — regime_validation.md, live/*.json, final_report.md

## Non-negotiables enforced in code

- Walk-forward only (`src/leakage/split.py`) — no random splits anywhere
- Train-only preprocessing (`TrainOnlyScaler`, `LeakageCheckingImputer`)
- Causality guards for news/graph/regime timestamps
- Honest reporting: SPA test, no-trade rate, calibration — see
  `reports/final_report.md` for what actually worked and what didn't.
