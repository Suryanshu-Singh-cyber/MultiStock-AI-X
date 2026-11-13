"""Stage 21 — SPA / Reality Check test (Hansen 2005).

Tests H0: no model beats the benchmark after data-snooping correction.
We use squared-error loss differentials d_k = L_benchmark - L_model
(positive = model better). Studentized test statistic with stationary
bootstrap p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger

log = get_logger(__name__)


def stationary_bootstrap_indices(n: int, avg_block: float, rng: np.random.Generator):
    """Politis-Romano stationary bootstrap indices."""
    p = 1.0 / avg_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def spa_test(losses: pd.DataFrame, benchmark: str, n_boot: int = 1000,
             block_len: float | None = None, seed: int = 42) -> dict:
    """losses: DataFrame (T x k) of per-period losses for each model.

    Returns p-value for H0 'no model beats benchmark'.
    """
    rng = np.random.default_rng(seed)
    L = losses.dropna()
    T, k = L.shape
    if benchmark not in L.columns:
        raise ValueError(f"benchmark {benchmark} not in columns {list(L.columns)}")

    # loss differentials vs benchmark: d_k = L_bench - L_k  (>0 = k better)
    D = L[benchmark].to_numpy()[:, None] - L.to_numpy()  # T x k
    D = np.delete(D, list(L.columns).index(benchmark), axis=1)
    models = [c for c in L.columns if c != benchmark]

    # observed stats: mean d_k / se(d_k)
    means = D.mean(axis=0)
    vars_ = D.var(axis=0, ddof=1)
    se = np.sqrt(vars_ / T)
    se[se == 0] = 1e-12
    t_obs = means / se
    t_max_obs = t_obs.max()

    # studentization under bootstrap
    if block_len is None:
        block_len = max(1.0, T ** (1 / 3))

    boot_max = np.empty(n_boot)
    for b in range(n_boot):
        ii = stationary_bootstrap_indices(T, block_len, rng)
        Db = D[ii]
        mb = Db.mean(axis=0)
        vb = Db.var(axis=0, ddof=1)
        seb = np.sqrt(vb / T)
        seb[seb == 0] = 1e-12
        # centered bootstrap stat (SPA uses recentered)
        tb = (mb - means) / seb
        boot_max[b] = tb.max()

    p_value = float(np.mean(boot_max >= t_max_obs))
    best_k = int(np.argmax(t_obs))
    return {
        "benchmark": benchmark,
        "n_models": k,
        "T": T,
        "best_model": models[best_k],
        "t_max": float(t_max_obs),
        "spa_pvalue": p_value,
        "significant_5pct": bool(p_value < 0.05),
        "mean_differentials": {m: float(means[i]) for i, m in enumerate(models)},
    }


def run_spa(oos_path: str = "data/processed/oos_predictions.parquet") -> dict:
    df = pd.read_parquet(oos_path)
    df["date"] = pd.to_datetime(df["date"])
    # squared-error losses per model per day
    model_cols = [c for c in df.columns if c not in ("date", "y")]
    L = pd.DataFrame({c: (df["y"] - df[c]) ** 2 for c in model_cols})
    L.index = df["date"]
    # benchmark = naive (predict 0 -> loss = y^2); use hist_mean if absent
    bench = "naive_zero" if "naive_zero" in L.columns else L.columns[0]
    res = spa_test(L, benchmark=bench, n_boot=1000)
    log.info("SPA: best=%s p=%.4f significant=%s",
             res["best_model"], res["spa_pvalue"], res["significant_5pct"])
    return res


if __name__ == "__main__":
    import json
    res = run_spa()
    print(json.dumps({k: v for k, v in res.items() if k != "mean_differentials"}, indent=2))
    print("mean differentials:", {k: round(v, 8) for k, v in res["mean_differentials"].items()})
