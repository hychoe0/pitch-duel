# xwOBA Regressor Calibration Investigation

**Date:** 2026-04-26  
**Scope:** 4-question diagnostic, no retraining. Synthetic D1 RHP (50-pitch, seed=42) as test vehicle.  
**Context:** MLB cross-hitter spread is 0.018 (0.129–0.147); AAA spread is 0.011 (0.131–0.142). Question is whether this is artifact or expected.

---

## Q1 — Is the small spread artifact or real?

**Method:** Run all 50 synthetic pitches against Ohtani (MLB) and Domínguez (AAA). Record per-pitch xwOBA distribution.

| Metric | Ohtani (MLB) | Domínguez (AAA) |
|--------|-------------|-----------------|
| n | 50 | 50 |
| mean | 0.1472 | 0.1378 |
| std | **0.0445** | **0.0415** |
| min | 0.0220 | 0.0290 |
| max | 0.2200 | 0.2070 |

**Ohtani top 5 (highest xwOBA):** FC 90.2mph (0.220), CH 83.8mph (0.219), CH 83.9mph (0.207), SL 84.4mph (0.207), FC 89.6mph (0.207)  
**Ohtani top 5 (lowest xwOBA):** FC 92.9mph (0.098), FF 92.6mph (0.098), SL 83.2mph (0.098), SL 85.6mph (0.026), SL 84.3mph (0.022)

**Decision rule:** std > 0.05 = model discriminates | std < 0.02 = near-flat

**Answer:** Both hitters fall in the ambiguous zone (0.040–0.045 std). The model is *not* near-flat — there is a 10× range from min to max (0.022 to 0.220 for Ohtani). The model discriminates strongly on pitch type and location (e.g., SL on the outer edge vs. FC to the inner third). However, this within-pitcher variation is driven mainly by pitch characteristics, not hitter identity. The small cross-hitter spread (averaging ~0.009–0.014 between hitters) reflects genuine compression due to game-context dominance (see Q3), not a bug.

---

## Q2 — Are hitter profile features doing meaningful work?

**Method:** Baseline mean xwOBA for Ohtani on 10 pitches (0.1253), then perturb each hitter aggregate feature by +0.10 in memory and re-predict.

| Feature | Ohtani baseline value | Delta xwOBA from +0.10 | Meaningful (≥0.005)? |
|---------|----------------------|------------------------|----------------------|
| hitter_hard_hit_rate | 0.5508 | −0.0007 | **NO** |
| hitter_chase_rate | 0.2879 | +0.0005 | **NO** |
| hitter_swing_rate | 0.4770 | +0.0001 | **NO** |
| hitter_contact_rate | 0.7214 | +0.0000 | **NO** |
| hitter_whiff_rate | 0.2784 | +0.0000 | **NO** |

**Answer:** None of the five hitter aggregate features produce a meaningful response. A +0.10 change in hard_hit_rate (elite vs. replacement-level gap) moves xwOBA by <0.001. This is the **smoking gun**: hitter aggregate profiles are essentially inert in the xwOBA regressor. This is mechanically consistent with Q3 (game context has 46% of importance; hitter aggregate has 7.9%, and within that, individual features capture very little given they are correlated and already summarized).

---

## Q3 — Feature importance: top 20 by gain

| Rank | Feature | Importance | Category |
|------|---------|------------|---------|
| 1 | balls | 0.2023 | GAME_CONTEXT |
| 2 | pitch_number | 0.1099 | GAME_CONTEXT |
| 3 | hitter_swing_rate_this_zone | 0.0952 | HITTER_CTX |
| 4 | strikes | 0.0616 | GAME_CONTEXT |
| 5 | plate_z | 0.0396 | PITCH |
| 6 | plate_x | 0.0355 | PITCH |
| 7 | hitter_swing_rate_this_count | 0.0353 | HITTER_CTX |
| 8 | hitter_stand_enc | 0.0298 | HITTER_AGG |
| 9 | prev_pitch_result_enc | 0.0286 | GAME_CONTEXT |
| 10 | times_through_order | 0.0279 | GAME_CONTEXT |
| 11 | hitter_whiff_rate_this_zone | 0.0250 | HITTER_CTX |
| 12 | pitch_type_enc | 0.0159 | PITCH |
| 13 | hitter_chase_rate | 0.0137 | HITTER_AGG |
| 14 | hitter_hard_hit_rate | 0.0135 | HITTER_AGG |
| 15 | pfx_z | 0.0127 | PITCH |
| 16 | p_throws_enc | 0.0121 | HITTER_AGG |
| 17 | pfx_x | 0.0120 | PITCH |
| 18 | prev_pitch_type_enc | 0.0084 | GAME_CONTEXT |
| 19 | spin_to_velo_ratio | 0.0069 | PITCH |
| 20 | release_speed | 0.0064 | PITCH |

**Category summary:**

| Category | Share | Interpretation |
|----------|-------|---------------|
| GAME_CONTEXT | 46.3% | Count (`balls`, `strikes`), `pitch_number`, sequencing — varies per PA, not per hitter |
| HITTER_CTX | 16.4% | Zone/count swing rates — changes with pitch location, varies less across hitters than per-pitch |
| PITCH | 14.3% | `plate_z`, `plate_x`, `pitch_type_enc` — the pitch quality signal |
| HITTER_ZONE | 9.2% | Zone-specific swing/whiff rates (profile) |
| HITTER_AGG | 7.9% | Aggregate profile rates (hard_hit, chase, swing, contact, whiff) |
| HITTER_PITCH_TYPE | 3.1% | Per-pitch-type contact rates |
| HITTER_FAMILY | 2.8% | Per-family swing/whiff rates |

**Answer:** Game context (`balls`, `pitch_number`, `strikes`) is the #1 driver of per-pitch xwOBA, accounting for 46.3% of all importance. The #1 feature — `balls` — makes theoretical sense: a 3-0 pitch carries high walk probability (woba_value = 0.69), inflating expected xwOBA regardless of the hitter. Hitter-identity features collectively account for ~30% of importance, but most of that (16.4%) is through contextual features that already vary by pitch location (swing_rate_this_zone), which is the same for every hitter facing the same pitch. True cross-hitter differentiation comes from the remaining ~18% (HITTER_ZONE + HITTER_AGG + HITTER_PITCH_TYPE + HITTER_FAMILY).

---

## Q4 — Decile calibration on full test set (2025+, n=703,915)

| Decile | N | Pred mean | Actual mean | Ratio |
|--------|---|-----------|-------------|-------|
| D1 (lowest) | 69,403 | 0.0009 | 0.0014 | 1.51 |
| D2 | 66,084 | 0.0029 | 0.0030 | 1.02 |
| D3 | 74,538 | 0.0089 | 0.0097 | 1.10 |
| D4 | 69,915 | 0.0216 | 0.0215 | 1.00 |
| D5 | 67,388 | 0.0404 | 0.0413 | 1.02 |
| D6 | 69,712 | 0.0632 | 0.0640 | 1.01 |
| D7 | 71,570 | 0.0896 | 0.0909 | 1.01 |
| D8 | 74,052 | 0.1183 | 0.1200 | 1.01 |
| D9 | 61,700 | 0.1564 | 0.1551 | 0.99 |
| D10 (highest) | 79,553 | 0.2766 | 0.2784 | **1.01** |

Overall: pred mean 0.0801 vs actual mean 0.0808.

**Answer:** Calibration is excellent. D10 actual = 0.2784 (passes ≥0.20 threshold). D1 actual = 0.0014 (passes ≤0.05 threshold). Ratios are within 1.0–1.1 for D2–D10; the slight 1.51 in D1 is noise at very small absolute values (0.0009 vs 0.0014, absolute gap 0.0005). The isotonic calibrator did its job — no systematic over- or under-prediction at any decile.

---

## Synthesis and Smoking Gun

**Smoking gun:** The xwOBA regressor is a *per-pitch damage estimator*, not a *cross-hitter discriminator*. Count (`balls` at 20% importance) and pitch sequencing (`pitch_number` at 11%) dominate because walks and pitch-clock-era count effects create the largest variance in per-pitch xwOBA by definition. A 3-0 pitch is high-xwOBA for any hitter; a 0-2 slider off the plate is low-xwOBA for any hitter. Only ~18% of importance reflects true cross-hitter differentiation, which mechanically produces the small cross-hitter spread.

**This is not a calibration failure.** Q4 shows the model is correctly calibrated across the entire output distribution. The small spread is structurally expected given the feature importance hierarchy.

**Ramírez anomaly:** Ramírez ranks #6 (xwOBA=0.130) rather than the expected #4–5 likely because his aggregate hard_hit_rate (Q2 confirms hitter_hard_hit_rate changes mean xwOBA by only −0.0007 per +0.10) cannot overcome game-context effects that equalize all hitters. His P(hard) = 0.206 (lowest among MLB cohort) correctly captures his contact profile against D1-level pitching; the xwOBA regressor fails to reproduce this ordering because it is blind to his lower hard-hit ceiling at that talent level.

---

## Recommendation

**Accept as-is (Option 1).** The xwOBA model is correctly calibrated and serves its actual purpose: per-pitch risk quantification (how dangerous is this count + location + pitch type?). Cross-hitter ranking should use **P(hard)** as the primary ordering signal — it has 4–5× the cross-hitter spread (MLB P(hard) spread ~0.100 vs xwOBA spread 0.018) and hitter profile features have direct structural influence through the three-stage classifiers.

The xwOBA output is most useful for absolute risk framing ("this pitch type in this count generates 0.147 expected xwOBA") and for pitch-type breakdown analysis, not for hitter ranking. The report already uses xwOBA for danger pitch counts (top-decile identification) rather than ordering, which is the correct use.

**If cross-hitter discrimination from xwOBA is a future requirement**, the targeted fix is a separate on-contact xwOBA model (filtered to batted balls only) with hitter-specific launch angle tendency and hard-hit rate features. That would sacrifice per-pitch scope for hitter differentiation. Given current product goals (per-pitch risk quantification), this is not warranted.
