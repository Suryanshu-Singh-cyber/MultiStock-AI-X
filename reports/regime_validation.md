# Honest regime validation result (Stage 14 gate)

Run: 2026-11-13 on full sample 2007-10 -> 2026-11, GaussianHMM n_states=3,
features = [mkt_ret, mkt_vol21, log(VIX), vix_chg5, sec_ret].

## Validation gates (see src/regime/hmm.py::validate_regimes)

| Gate | Metric | Threshold | Result | Pass? |
|---|---|---|---|---|
| Persistence | mean regime duration | >= 10 days | 39.7 days | YES |
| Stability | first/second-half TV distance | <= 0.30 | 0.133 | YES |
| Predictive | spread of regime mean next-day ret | > 0.25 * pooled std | 0.00013 (~0.7% of std) | **NO** |

## Verdict

Regimes are persistent and stable but carry **no detectable next-day
predictive signal** for POWERGRID returns. Per the Stage 14 gate,
`passed = False` -> regime labels are NOT fed into downstream models
(MoE gating on regime is disabled unless re-validated on a richer feature
set). This is reported honestly rather than forced in.

Note: regimes may still be useful as RISK conditioning (vol regimes) in
Stage 17 even if not return-predictive — that usage will be validated
separately on volatility prediction, not returns.
