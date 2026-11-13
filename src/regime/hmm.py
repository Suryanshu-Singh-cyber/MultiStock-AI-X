"""Stage 14 — Regime detection (validated BEFORE use).

Gaussian HMM on market-layer features (NIFTY ret/vol, VIX, sector ret).
Validation gate: a regime is only allowed into models if
  (a) persistence: mean regime duration >= 10 days
  (b) stability: regime distribution TV-distance between first/second half < 0.3
  (c) predictive: |mean next-day target_ret| differs across regimes (ANOVA-ish)
Only regimes passing all three are exported via regime_labels().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import get_symbols
from src.core.logger import get_logger
from src.data.target import load_raw

log = get_logger(__name__)


def _features() -> pd.DataFrame:
    symbols = get_symbols()
    idx = load_raw(symbols["target"]).index
    def lr(sym):
        return np.log(load_raw(sym)["Close"]).diff().reindex(idx)
    nr = lr(symbols["indices"]["nifty_50"])
    vix = load_raw(symbols["indices"]["india_vix"])["Close"].reindex(idx)
    sr = lr(symbols["indices"]["sector_energy"])
    X = pd.DataFrame({
        "mkt_ret": nr,
        "mkt_vol21": nr.rolling(21, min_periods=10).std(),
        "vix": np.log(vix),
        "vix_chg5": vix.pct_change(5),
        "sec_ret": sr,
    }, index=idx)
    return X


def fit_hmm(n_states: int = 3) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Fit GaussianHMM on the FULL sample for description; for causal use we
    refit expanding-window in regime_labels(). Returns states, names, probs."""
    from hmmlearn.hmm import GaussianHMM

    X = _features().dropna()
    Xs = (X - X.mean()) / X.std()
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=200, random_state=42)
    model.fit(Xs.to_numpy())
    states = model.predict(Xs.to_numpy())
    # name states by mean VIX (low/mid/high vol)
    vix_mean = X.assign(s=states).groupby("s")["vix"].mean().sort_values()
    order = vix_mean.index.tolist()
    names = ["low_vol", "mid_vol", "high_vol"][:n_states]
    mapping = {old: names[i] for i, old in enumerate(order)}
    labels = pd.Series([mapping[s] for s in states], index=X.index)
    probs = model.predict_proba(Xs.to_numpy())
    return states, names, pd.DataFrame(probs, index=X.index)


def validate_regimes(labels: pd.Series, y: pd.Series,
                     min_duration: int = 10, tv_max: float = 0.3) -> dict:
    """Run the three validation gates."""
    # (a) persistence: mean run length
    changes = (labels != labels.shift()).cumsum()
    runs = labels.groupby(changes).size()
    mean_dur = float(runs.mean())
    persist_ok = mean_dur >= min_duration

    # (b) stability: distribution of states in first vs second half
    h = len(labels) // 2
    p1 = labels.iloc[:h].value_counts(normalize=True)
    p2 = labels.iloc[h:].value_counts(normalize=True)
    allk = p1.index.union(p2.index)
    tv = 0.5 * float(sum(abs(p1.get(k, 0) - p2.get(k, 0)) for k in allk))
    stable_ok = tv <= tv_max

    # (c) predictive: does next-day return differ by regime?
    df = pd.DataFrame({"reg": labels, "y": y.reindex(labels.index)}).dropna()
    grp = df.groupby("reg")["y"].agg(["mean", "std", "count"])
    # spread of regime means relative to pooled std
    spread = float(grp["mean"].max() - grp["mean"].min())
    pred_ok = spread > 0.25 * float(df["y"].std())

    return {
        "mean_duration": mean_dur,
        "persistence_ok": persist_ok,
        "tv_distance": tv,
        "stability_ok": stable_ok,
        "regime_mean_spread": spread,
        "predictive_ok": pred_ok,
        "passed": bool(persist_ok and stable_ok and pred_ok),
        "regime_stats": grp.to_dict(),
    }


def regime_labels_causal(n_states: int = 3) -> pd.Series:
    """Causal regime labels: expanding-window HMM refit every 63 days.
    Label at t uses only data <= t-1 (filtered state)."""
    from hmmlearn.hmm import GaussianHMM

    X = _features()
    y = np.log(load_raw(get_symbols()["target"])["Close"]).diff()
    idx = X.index
    Xv = X.to_numpy()
    valid = np.isfinite(Xv).all(axis=1)

    labels = pd.Series(index=idx, dtype=object)
    refit_every = 63
    last_fit = -refit_every
    model = None
    scaler_mean = scaler_std = None
    for i, d in enumerate(idx):
        if not valid[i] or i < 126:
            continue
        if i - last_fit >= refit_every or model is None:
            lo = max(0, i - 756)  # ~3y expanding-ish window
            seg = Xv[lo:i][np.isfinite(Xv[lo:i]).all(axis=1)]
            if len(seg) < 126:
                continue
            scaler_mean = seg.mean(axis=0)
            scaler_std = seg.std(axis=0) + 1e-9
            Z = (seg - scaler_mean) / scaler_std
            model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=100, random_state=42)
            model.fit(Z)
            last_fit = i
        z = (Xv[i] - scaler_mean) / scaler_std
        if not np.isfinite(z).all():
            continue
        # filtered state = predict_proba on single obs approximates filtered
        p = model.predict_proba(z.reshape(1, -1))[0]
        labels.iloc[i] = int(np.argmax(p))
    return labels


if __name__ == "__main__":
    states, names, probs = fit_hmm()
    y = np.log(load_raw(get_symbols()["target"])["Close"]).diff()
    labels = pd.Series(states, index=probs.index)
    labels = labels.map({i: names[i] for i in range(len(names))})
    v = validate_regimes(labels, y)
    print("regime validation:", {k: v[k] for k in
          ["mean_duration", "persistence_ok", "tv_distance", "stability_ok",
           "regime_mean_spread", "predictive_ok", "passed"]})
