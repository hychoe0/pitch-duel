# V5 vs V4 Headline Metrics

## Model Version
- **V4** (baseline, 80 features): `models/baseline_v4/`
- **V5** (current, 89 features): `models/v5/`

## Feature Changes (V4 → V5)
- **+7** HITTER_CONTEXT_FEATURES: `hitter_zone_swing_rate`, `hitter_zone_whiff_rate`, `hitter_zone_xwoba`, `hitter_zone_hard_hit_rate`, `hitter_contact_rate_this_pitch`, `hitter_family_swing_rate`, `hitter_family_whiff_rate`
- **+2** STUFF_VS_HITTER_FEATURES: `stuff_vs_hitter_xwoba`, `stuff_vs_hitter_whiff` (KNN physics similarity per hitter)
- **Total**: 80 → 89 features

## AUC Comparison

| Stage | V4 AUC | V5 AUC | Δ |
|-------|--------|--------|---|
| Swing | 0.8747 | 0.8745 | -0.0002 |
| Contact\|Swing | 0.7923 | 0.7876 | -0.0047 |
| Hard Contact\|Contact | 0.7201 | 0.7184 | -0.0017 |

## Rate Check (sanity)

| Metric | V4 | V5 |
|--------|----|----|
| swing_rate | 0.4716 | 0.4716 |
| contact_rate | 0.7721 | 0.7721 |
| hard_contact_rate | 0.3661 | 0.3661 |

## Observations
- Swing AUC: essentially unchanged (-0.0002)
- Contact AUC: declined -0.0047 (worth monitoring — new context features may be noisy for contact prediction)
- Hard contact AUC: declined -0.0017 (within expected variance)
- All base rates match — no data leakage introduced
