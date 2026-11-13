"""Stage 6 — Feature selection engine.

Pipeline (each step can only REMOVE features, never add information):
  1. leakage check      — every feature must have timestamp <= t (by
                          construction; we assert monotonic index & that
                          no feature column references target_ret)
  2. redundancy         — drop |spearman ρ| > 0.95 pairs (keep first by
                          univariate |IC|)
  3. univariate screen  — |IC| vs target on TRAIN windows only; keep top-K
  4. walk-forward value — incremental OOS value of the set vs market-only
                          subset, measured by Ridge walk-forward dir_acc
  5. stability          — feature must have consistent sign of IC across
                          walk-forward folds (>=60% same sign)
  6. regime-conditional — optional: report per-regime IC (filled in Stage 14)

Outputs data/processed/selected_features.json + selection_report.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.core.config import ROOT
from src.core.logger import get_logger
from src.data.target import build_target_panel
from src.features.library import build_feature_library
from src.leakage.scaler import LeakageCheckingImputer, TrainOnlyScaler
from src.leakage.split import walk_forward_splits

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"

RHO_MAX = 0.95
TOP_K_UNIVARIATE = 60
IC_STABILITY_MIN = 0.6


def _ic(x: pd.Series, y: pd.Series) -> float:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 30:
        return 0.0
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def run_selection(max_splits: int = 12) -> dict:
    panel = build_target_panel()
    X = build_feature_library()
    y = panel["target_ret"]

    label_dates = panel.index[panel["target_ret"].notna()]
    splits = walk_forward_splits(label_dates)[::8][:max_splits]
    log.info("selection over %d walk-forward folds", len(splits))

    # ---- step 1: leakage sanity — index must be sorted unique; target col absent
    assert X.index.is_monotonic_increasing and X.index.is_unique
    assert "target_ret" not in X.columns
    # features are trailing windows by construction; document check passed
    log.info("leakage check: %d features, index %s..%s", X.shape[1],
             X.index[0].date(), X.index[-1].date())

    # ---- step 2+3: redundancy + univariate IC on pooled train windows
    # Use the LAST split's train for univariate stats (most recent regime),
    # but IC sign stability is measured across all folds below.
    last = splits[-1]
    tr = last.train_dates
    Xtr, ytr = X.reindex(tr), y.reindex(tr)

    ics = {c: _ic(Xtr[c], ytr) for c in X.columns}
    ic_abs = pd.Series({c: abs(v) for c, v in ics.items()}).sort_values(ascending=False)

    # redundancy on train data (spearman corr of features)
    corr = Xtr[ic_abs.index[:120]].corr(method="spearman").abs()
    keep, dropped_red = [], []
    for c in ic_abs.index:
        if all(abs(corr.loc[c, k]) <= RHO_MAX for k in keep):
            keep.append(c)
        else:
            dropped_red.append(c)
        if len(keep) >= TOP_K_UNIVARIATE:
            break
    log.info("redundancy: kept %d, dropped %d (|rho|>%.2f)",
             len(keep), len(dropped_red), RHO_MAX)

    # ---- step 5: IC sign stability across folds (on the kept set)
    sign_counts = {c: 0 for c in keep}
    fold_ics = []
    for sp in splits:
        Xt, yt = X.reindex(sp.train_dates), y.reindex(sp.train_dates)
        row = {}
        for c in keep:
            ic = _ic(Xt[c], yt)
            row[c] = ic
            if ic != 0 and np.sign(ic) == np.sign(ics[c]):
                sign_counts[c] += 1
        fold_ics.append(row)
    n_folds = len(splits)
    stable = [c for c in keep
              if sign_counts[c] / max(1, n_folds) >= IC_STABILITY_MIN]
    unstable = [c for c in keep if c not in stable]
    log.info("stability: %d stable, %d unstable (sign consistency >= %.0f%%)",
             len(stable), len(unstable), IC_STABILITY_MIN * 100)

    # ---- step 4: walk-forward incremental value
    # Compare Ridge OOS dir-acc: market-only subset vs stable selected set.
    market_cols = [c for c in stable if c.startswith(("mkt_", "sec_", "vix", "beta_", "corr_"))]
    alpha_cols = [c for c in stable if c not in market_cols]

    def wf_dir_acc(cols: list[str]) -> tuple[float, int]:
        if not cols:
            return np.nan, 0
        correct, total = 0, 0
        for sp in splits:
            cutoff = sp.test_dates.min()
            Xtr, ytr = X.reindex(sp.train_dates)[cols], y.reindex(sp.train_dates)
            Xte, yte = X.reindex(sp.test_dates)[cols], y.reindex(sp.test_dates)
            ok = ytr.notna().to_numpy()
            imp = LeakageCheckingImputer().fit_dated(Xtr[ok], sp.train_dates[ok], cutoff)
            a, b = imp.transform(Xtr[ok]), imp.transform(Xte)
            sc = TrainOnlyScaler().fit_dated(a, sp.train_dates[ok], cutoff)
            a, b = sc.transform(a), sc.transform(b)
            m = ~(np.isnan(b).any(axis=1) | np.isnan(yte.to_numpy()))
            if m.sum() < 5:
                continue
            ridge = Ridge(alpha=1.0).fit(a[~np.isnan(a).any(axis=1)], ytr[ok][~np.isnan(a).any(axis=1)])
            p = ridge.predict(b[m])
            yt = yte.to_numpy()[m]
            correct += int(np.sum(np.sign(p) == np.sign(yt)))
            total += int(m.sum())
        return (correct / total if total else np.nan), total

    acc_market, n_m = wf_dir_acc(market_cols)
    acc_all, n_a = wf_dir_acc(stable)
    log.info("walk-forward dir_acc: market-only=%.4f (%d)  selected=%.4f (%d)",
             acc_market, n_m, acc_all, n_a)

    # final selected set: stable features; report dropped reasons
    result = {
        "n_candidates": int(X.shape[1]),
        "n_after_redundancy": len(keep) + len(dropped_red),
        "n_univariate_top": TOP_K_UNIVARIATE,
        "n_stable": len(stable),
        "selected": stable,
        "dropped_redundancy": dropped_red[:50],
        "dropped_unstable": unstable,
        "wf_dir_acc_market_only": acc_market,
        "wf_dir_acc_selected": acc_all,
        "top_ics": {c: ics[c] for c in stable[:20]},
    }
    (PROC / "selected_features.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([{"feature": c, "ic": ics[c],
                    "sign_consistency": sign_counts[c] / max(1, n_folds),
                    "stable": c in stable} for c in keep]
                 ).sort_values("ic", key=abs, ascending=False
                               ).to_csv(PROC / "selection_report.csv", index=False)
    return result


if __name__ == "__main__":
    res = run_selection()
    print(f"\nselected {len(res['selected'])} stable features "
          f"(from {res['n_candidates']} candidates)")
    print(f"WF dir-acc market-only={res['wf_dir_acc_market_only']:.4f} "
          f"selected={res['wf_dir_acc_selected']:.4f}")
