# V5 Feature Group Importances

Avg importance (across all 3 stages) per feature group.

| Group | N | Avg | Swing | Contact | Hard Contact |
|-------|---|-----|-------|---------|--------------|
| pitch_quality | 11 | 0.2772 | 0.2469 | 0.3059 | 0.2789 |
| hitter_context_new | 7 | 0.2714 | 0.2463 | 0.2782 | 0.2896 |
| context | 12 | 0.1955 | 0.2751 | 0.1824 | 0.1290 |
| hitter_aggregate | 7 | 0.0811 | 0.0521 | 0.0597 | 0.1314 |
| hitter_contextual | 5 | 0.0586 | 0.1241 | 0.0279 | 0.0239 |
| hitter_pitch_type | 11 | 0.0502 | 0.0299 | 0.0465 | 0.0743 |
| hitter_family_rates | 8 | 0.0392 | 0.0222 | 0.0394 | 0.0559 |
| stuff_vs_hitter | 2 | 0.0268 | 0.0034 | 0.0600 | 0.0170 |
| hitter_zone_rates | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## New features performance

### HITTER_CONTEXT_FEATURES (+7 Prompt 2)
- **hitter_zone_swing_rate**: avg=0.1099  (0.1965 / 0.0091 / 0.1242)
- **hitter_zone_whiff_rate**: avg=0.0799  (0.0308 / 0.1962 / 0.0126)
- **hitter_zone_xwoba**: avg=0.0050  (0.0026 / 0.0036 / 0.0087)
- **hitter_zone_hard_hit_rate**: avg=0.0421  (0.0022 / 0.0039 / 0.1203)
- **hitter_contact_rate_this_pitch**: avg=0.0220  (0.0037 / 0.0557 / 0.0067)
- **hitter_family_swing_rate**: avg=0.0065  (0.0088 / 0.0040 / 0.0067)
- **hitter_family_whiff_rate**: avg=0.0059  (0.0017 / 0.0056 / 0.0104)

### STUFF_VS_HITTER_FEATURES (+2 Prompt 3)
- **stuff_vs_hitter_xwoba**: avg=0.0068  (0.0013 / 0.0085 / 0.0106)
- **stuff_vs_hitter_whiff**: avg=0.0200  (0.0021 / 0.0515 / 0.0064)
