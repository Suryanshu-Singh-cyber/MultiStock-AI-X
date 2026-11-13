"""SPA test including deep-model OOS predictions.

Merges baseline OOS preds (oos_predictions.parquet) with deep LSTM preds
(deep_lstm_oos.parquet) on common dates, then runs Hansen SPA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import ROOT
from src.models.spa import spa_test

PROC = ROOT / "data" / "processed"


def run() -> dict:
    base = pd.read_parquet(PROC / "oos_predictions.parquet")
    base["date"] = pd.to_datetime(base["date"])
    base = base.set_index("date")

    deep_path = PROC / "deep_lstm_oos.parquet"
    if deep_path.exists():
        deep = pd.read_parquet(deep_path)
        deep["date"] = pd.to_datetime(deep["date"])
        deep = deep.set_index("date")[["lstm", "y"]].rename(columns={"y": "y_deep"})
        df = base.join(deep, how="inner")
        df["y"] = df["y"].fillna(df["y_deep"])
        df = df.drop(columns=["y_deep"])
    else:
        df = base

    model_cols = [c for c in df.columns if c not in ("y",)]
    L = pd.DataFrame({c: (df["y"] - df[c]) ** 2 for c in model_cols})
    L = L.dropna()
    res = spa_test(L, benchmark="naive_zero", n_boot=1000)
    print("SPA with deep models:")
    print({k: v for k, v in res.items() if k != "mean_differentials"})
    print("diffs:", {k: round(v, 8) for k, v in res["mean_differentials"].items()})
    return res


if __name__ == "__main__":
    run()
