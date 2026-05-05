# V5 Gate Checks

All gates use V5 = `models/` (retrained on 89 features) vs V4 = `models/baseline_v4/` (80 features).

## Gate 1 — Swing AUC regression ≤ 0.005

| | V4 | V5 | Δ | Threshold | Result |
|-|----|----|---|-----------|--------|
| Swing AUC | 0.8747 | 0.8745 | −0.0002 | ≥ V4 − 0.005 | **PASS** |

## Gate 2 — Contact AUC regression ≤ 0.005

| | V4 | V5 | Δ | Threshold | Result |
|-|----|----|---|-----------|--------|
| Contact AUC | 0.7923 | 0.7876 | −0.0047 | ≥ V4 − 0.005 | **PASS** |

Note: Contact AUC declined −0.0047, within the −0.005 tolerance. Worth monitoring in future retrains.

## Gate 3 — Hard contact AUC regression ≤ 0.005

| | V4 | V5 | Δ | Threshold | Result |
|-|----|----|---|-----------|--------|
| Hard contact AUC | 0.7201 | 0.7184 | −0.0017 | ≥ V4 − 0.005 | **PASS** |

## Gate 4 — Demo battery hitter differentiation

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| P(hard) spread (max − min) | 0.164 | ≥ 0.050 | **PASS** |
| Std dev | 0.052 | — | — |
| Min P(hard) | 0.228 (Kiner-Falefa) | — | — |
| Max P(hard) | 0.393 (Ohtani) | — | — |

## Summary

| Gate | Result |
|------|--------|
| G1 — Swing AUC | ✅ PASS |
| G2 — Contact AUC | ✅ PASS (marginal: −0.0047 of −0.005 budget) |
| G3 — Hard contact AUC | ✅ PASS |
| G4 — Differentiation | ✅ PASS |

**Overall: ALL PASS — V5 is production-ready.**

## Notes
- `hitter_zone_rates` (26 static per-zone features) showed 0% importance — superseded by `hitter_context_new` features that resolve zone at pitch-time
- `hitter_context_new` (7 features, Prompt 2) ranked 2nd among all feature groups by avg importance (27.1%)
- `stuff_vs_hitter` (2 features, Prompt 3) shows meaningful importance in contact stage (6.0%)
- xwOBA model retrained on 89 features: val RMSE=0.209
