"""Stages 8-10 — Deep sequence models: LSTM, LSTM+Attention, Transformer.

All models predict a Gaussian (mu, sigma) for next-day log return.
Walk-forward protocol identical to baselines: fit on train window only,
predict test window. Sequence length L uses features at t-L+1..t to predict
target at t (return t->t+1) — no future data: the last sequence element is
the feature row at t, and y is target_ret[t].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ROOT
from src.core.logger import get_logger

log = get_logger(__name__)
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "models_ckpt"
CKPT.mkdir(exist_ok=True)

SEQ_LEN = 60
HIDDEN = 64
EPOCHS = 15
BATCH = 128
LR = 1e-3
DEVICE = "cpu"


def load_model_frame(max_features: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix + target aligned on label-eligible dates."""
    from src.data.target import build_target_panel
    from src.features.library import build_feature_library

    panel = build_target_panel()
    y = panel["target_ret"]

    sel_path = PROC / "selected_features.json"
    if sel_path.exists():
        cols = json.loads(sel_path.read_text())["selected"][:max_features]
    else:
        Xall = build_feature_library()
        cols = [c for c in Xall.columns if not Xall[c].isna().all()][:max_features]

    X = build_feature_library()[cols]
    # keep rows where target exists; X may have NaN -> filled per-fold later
    df = X.copy()
    df["__y__"] = y
    df = df.dropna(subset=["__y__"])
    return df[cols], df["__y__"]


def make_sequences(X: np.ndarray, y: np.ndarray, idx: np.ndarray,
                   seq_len: int = SEQ_LEN):
    """Build (seq, target) pairs. Sample i uses X[idx[i]-L+1 : idx[i]+1]."""
    Xs, ys = [], []
    for i in idx:
        if i - seq_len + 1 < 0:
            continue
        Xs.append(X[i - seq_len + 1: i + 1])
        ys.append(y[i])
    if not Xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)
    return np.stack(Xs), np.array(ys)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _LSTMNet:
    """Factory returning an nn.Module for the requested architecture."""

    @staticmethod
    def build(n_features: int, kind: str):
        torch, nn = _torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.kind = kind
                self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=1, batch_first=True)
                if kind == "attn":
                    self.attn = nn.MultiheadAttention(HIDDEN, num_heads=4, batch_first=True)
                if kind == "transformer":
                    self.proj = nn.Linear(n_features, HIDDEN)
                    enc = nn.TransformerEncoderLayer(d_model=HIDDEN, nhead=4,
                                                     dim_feedforward=128,
                                                     batch_first=True)
                    self.transformer = nn.TransformerEncoder(enc, num_layers=1)
                self.head = nn.Linear(HIDDEN, 2)

            def forward(self, x):
                if self.kind == "transformer":
                    h = self.transformer(self.proj(x))
                    z = h[:, -1]
                else:
                    out, _ = self.lstm(x)
                    if self.kind == "attn":
                        a, _ = self.attn(out, out, out)
                        z = a[:, -1]
                    else:
                        z = out[:, -1]
                return self.head(z)

        return Net()

    def __init__(self, n_features: int, kind: str = "lstm"):
        raise RuntimeError("use _LSTMNet.build()")


def _train_eval(X_df: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex,
                kind: str, stride: int = 6, max_splits: int = 12):
    """Walk-forward eval for one deep model kind. Returns OOS preds + metrics."""
    import torch
    from src.data.target import build_target_panel
    from src.features.library import build_feature_library
    from src.leakage.split import walk_forward_splits
    from src.models.metrics import regression_metrics, trading_metrics

    torch.manual_seed(42)
    np.random.seed(42)

    label_dates = y.index[y.notna()]
    splits = walk_forward_splits(label_dates)[::stride][:max_splits]

    # NO ffill/bfill — that would leak future values into early rows.
    # NaNs are handled per-fold: after train-only standardization, NaN -> 0
    # (= train mean), which is causal.
    Xv = X_df.reindex(label_dates).to_numpy(dtype=np.float32)
    yv_all = y.reindex(label_dates).to_numpy(dtype=np.float32)
    date_to_pos = {d: i for i, d in enumerate(label_dates)}

    preds, trues, dates_out = [], [], []
    for si, sp in enumerate(splits):
        tr_pos = np.array([date_to_pos[d] for d in sp.train_dates if d in date_to_pos])
        te_pos = np.array([date_to_pos[d] for d in sp.test_dates if d in date_to_pos])
        if len(tr_pos) < SEQ_LEN + 100 or len(te_pos) == 0:
            continue
        # fold-local standardization (fit on train only)
        mu = np.nanmean(Xv[tr_pos], axis=0)
        sd = np.nanstd(Xv[tr_pos], axis=0) + 1e-6
        Xs = (Xv - mu) / sd
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        ymu, ysd = float(np.nanmean(yv_all[tr_pos])), float(np.nanstd(yv_all[tr_pos]) + 1e-6)

        Xtr_s, ytr_s = make_sequences(Xs, (yv_all - ymu) / ysd, tr_pos)
        Xte_s, yte_s = make_sequences(Xs, (yv_all - ymu) / ysd, te_pos)
        if len(Xtr_s) == 0 or len(Xte_s) == 0:
            continue

        net = _LSTMNet.build(X_df.shape[1], kind=kind)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        xt = torch.tensor(Xtr_s)
        yt = torch.tensor(ytr_s, dtype=torch.float32)
        n = len(xt)
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            for b in range(0, n, BATCH):
                ii = perm[b:b + BATCH]
                out = net(xt[ii])
                mu_p, ls_p = out[:, 0], out[:, 1]
                var = torch.exp(2 * ls_p).clamp(min=1e-6)
                nll = 0.5 * ((yt[ii] - mu_p) ** 2 / var + 2 * ls_p + np.log(2 * np.pi))
                loss = nll.mean()
                opt.zero_grad(); loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            p = []
            for b in range(0, len(Xte_s), 512):
                out = net(torch.tensor(Xte_s[b:b + 512]))
                p.append(out[:, 0].numpy())
            pred_norm = np.concatenate(p)
        pred = pred_norm * ysd + ymu
        preds.append(pred)
        trues.append(yv_all[te_pos])
        dates_out.append(sp.test_dates)
        log.info("%s split %d done (train_n=%d test_n=%d)", kind, si, len(Xtr_s), len(Xte_s))

    if not preds:
        return None
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    met = regression_metrics(y_true, y_pred, sigma=float(np.std(y_true)))
    tr = trading_metrics(y_true, y_pred, cost_per_unit=0.0026)
    return {"kind": kind, "dates": np.concatenate(dates_out),
            **{f"reg_{k}": v for k, v in met.items()},
            **{f"tr_{k}": v for k, v in tr.items()},
            "preds": np.concatenate(preds), "trues": y_true}


def run_deep(stride: int = 6, max_splits: int = 12, kinds=None):
    X_df, y = load_model_frame()
    log.info("deep frame: %d features, %d rows", X_df.shape[1], len(X_df))
    results = {}
    for kind in (kinds or ["lstm", "attn", "transformer"]):
        r = _train_eval(X_df, y, None, kind, stride=stride, max_splits=max_splits)
        if r:
            results[kind] = r
            # persist OOS preds for SPA / ablation
            pd.DataFrame({"date": r.get("dates"), "y": r["trues"],
                          kind: r["preds"]}).to_parquet(
                PROC / f"deep_{kind}_oos.parquet")
            print(f"{kind}: dir_acc={r['reg_dir_acc']:.4f} rmse={r['reg_rmse']:.5f} "
                  f"net_sharpe={r['tr_net_sharpe']:.3f}")
    return results


if __name__ == "__main__":
    run_deep()
