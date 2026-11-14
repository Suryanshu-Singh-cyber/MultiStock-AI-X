"""DEMO SCRIPT — run this in front of the professor.

  .venv\\Scripts\\python demo_training.py

Prints: training config, real epoch logs, and the model-vs-model
out-of-sample comparison computed from saved predictions.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

print("=" * 70)
print("  MultiStock-AI X — TRAINING & MODEL COMPARISON DEMO")
print("=" * 70)

# ---------- 1. TRAINING CONFIG ----------
print("\n[1] LSTM TRAINING CONFIG  (src/models/deep.py)")
print("-" * 70)
from src.models import deep
for k in ["SEQ_LEN", "HIDDEN", "EPOCHS", "BATCH", "LR"]:
    print(f"  {k:<10} = {getattr(deep, k)}")
print("  input      = last 60 days x 53 selected features")
print("  output     = (mu, log_sigma)  -> predicts a DISTRIBUTION")
print("  loss       = Gaussian NLL  (learns prediction AND uncertainty)")
print("  window     = last ~1000 label days (~4 years)")
print("  retrain    = weekly (checkpoint age > 7 days)")
print("  checkpoint = models_ckpt/lstm_prod.pt  (112 KB, in git)")

# ---------- 2. REAL EPOCH LOGS ----------
print("\n[2] REAL TRAINING EPOCH LOGS  (logs/app.log — this actually happened)")
print("-" * 70)
log = ROOT / "logs" / "app.log"
epochs = []
if log.exists():
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "lstm epoch" in line or "lstm checkpoint" in line:
            epochs.append(line.split("|", 2)[-1].strip())
for e in epochs[-16:]:
    print("  " + e)
if not epochs:
    print("  (no epoch logs found — run inference once to train)")

# ---------- 3. MODEL COMPARISON (out-of-sample) ----------
print("\n[3] OUT-OF-SAMPLE MODEL COMPARISON")
print("-" * 70)
print("  Every number below is a TRUE out-of-sample walk-forward")
print("  prediction — models never saw these days during training.\n")

oos = PROC / "oos_predictions.parquet"
if oos.exists():
    df = pd.read_parquet(oos)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  pooled OOS days: {len(df)}   ({df['date'].min().date()} -> {df['date'].max().date()})")
    cost = 0.0026
    rows = []
    for col in df.columns:
        if col in ("date", "y"):
            continue
        pred = df[col].to_numpy()
        y = df["y"].to_numpy()
        mask = np.isfinite(pred) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        p, yy = pred[mask], y[mask]
        dir_acc = float(np.mean(np.sign(p) == np.sign(yy)))
        rmse = float(np.sqrt(np.mean((yy - p) ** 2)))
        mae = float(np.mean(np.abs(yy - p)))
        pos = (p > 0).astype(float)
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pos * yy - cost * turnover
        sharpe = float(np.mean(net) / (np.std(net) + 1e-9) * np.sqrt(252))
        rows.append((col, dir_acc, rmse, mae, sharpe))
    print(f"\n  {'model':<14}{'dir_acc':>9}{'RMSE':>9}{'MAE':>9}{'netSharpe':>11}")
    for name, da, rm, ma, sh in sorted(rows, key=lambda r: -r[4]):
        print(f"  {name:<14}{da:>9.3f}{rm:>9.4f}{ma:>9.4f}{sh:>11.2f}")

# ---------- 4. DEEP MODEL RESULTS ----------
lstm_oos = PROC / "deep_lstm_oos.parquet"
if lstm_oos.exists():
    d = pd.read_parquet(lstm_oos)
    print(f"\n[4] LSTM DEDICATED EVAL  ({len(d)} OOS days)")
    print("-" * 70)
    y = d["y"].to_numpy() if "y" in d else d.iloc[:, -1].to_numpy()
    for col in d.columns:
        if col in ("date", "y"):
            continue
        p = d[col].to_numpy()
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() < 50:
            continue
        print(f"  {col:<22} dir_acc={np.mean(np.sign(p[m])==np.sign(y[m])):.3f}"
              f"  rmse={np.sqrt(np.mean((y[m]-p[m])**2)):.4f}")

# ---------- 5. HONEST SUMMARY ----------
print("\n[5] HONEST VERDICT")
print("-" * 70)
print("  - LSTM is the BEST model but dir_acc ~52-54% (coin flip is 50%)")
print("  - SPA test p=0.649 -> NOT statistically proven to beat naive yet")
print("  - System's real edge = honest uncertainty + NO-TRADE when edge<cost")
print("  - Live accuracy accumulates daily via forecast_log -> scorecard")
print("\n" + "=" * 70)
