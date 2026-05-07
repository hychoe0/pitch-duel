# Pitch Duel — Project Context

## What It Is
Competitive exposure simulator for pitchers. Core question: "How good is my pitch against specific hitters?"
Named hitter simulation (e.g. Ohtani, KBO cleanup hitters) is the differentiator — tier fallback for thin samples only.
Origin: KBO/international pitchers have no benchmark against elite hitters. Exposed at every WBC.

## Model Architecture
**Three-stage XGBoost pipeline** (models_v2/ is active):
- Stage 1: P(swing)
- Stage 2: P(contact | swing)
- Stage 3: P(hard_contact | contact)

Each stage has its own XGBClassifier + isotonic calibrator. **Pitcher features are defined but disconnected** (`PITCHER_FEATURES` in preprocess.py excluded from ALL_FEATURES).

**Phase 2 (future):** LSTM/Transformer for full at-bat sequences, tunneling, sequencing effects.

### Feature Groups (80 total)
- **Pitch quality (11):** release_speed, spin_to_velo_ratio, pfx_x/z, release_pos_x/z, release_extension, plate_x/z, pitch_type_enc, era_flag_enc
- **Context (12):** balls, strikes, pitch_number, prev_pitch_type/speed/result_enc, on_1b/2b/3b_flag, inning, score_diff, times_through_order
- **Hitter profile — aggregate (7):** swing/chase/contact/hard_hit/whiff rates, hitter_stand_enc, p_throws_enc
- **Hitter profile — zone rates (26):** swing + whiff rate per Statcast zone (1–9 strike, 11–14 ball)
- **Hitter profile — pitch family rates (8):** swing + whiff rate per family (fastball/breaking/offspeed/other)
- **Hitter profile — per-pitch-type contact (11):** contact rate by individual pitch type
- **Hitter contextual (5):** hitter_swing/whiff_rate_this_zone, hitter_swing/whiff_rate_this_family, hitter_swing_rate_this_count

### Data & Splits
- **Training:** Statcast 2015–2025 (excluding 2020), active hitters from 2022+ only
- **Test:** 2026 (fully unseen)
- **Profile cutoff:** 2026-01-01 (profiles built through end of 2025, no leakage)
- **Early stopping:** 2025 season carved from training as val set

### Era Integrity
Raw spin rate distorted across eras — use `spin_to_velo_ratio` instead.
Era flag: pre_crackdown (<Jun 21 2021), post_crackdown (Jun–Dec 2021), ambiguous (2022), pitch_clock (2023+).
Sample weights: 2026=2.0x, 2025=1.8x, 2024=1.5x, 2023=1.2x, 2022=0.6x, post_crackdown=0.7x, pre_crackdown=0.3x, 2020=excluded.

### Hitter Profiles
Built from Statcast with recency weights: 2026=3.0x → 2021 and earlier=0.5x.
Minimum 200 weighted PA before using named profile; below that, blend with league averages.
Zone rates use Statcast's native `zone` column. Pitch families: fastball (FF/SI/FC), breaking (SL/CU/KC/SV/CS), offspeed (CH/FS/FO), other.
Fallback thresholds: 20 weighted pitches per zone, 30 per family.
`stand` field: 'L', 'R', or 'S' (switch). Switch hitters bat opposite pitcher's hand. Detection: ≥10% minority-side appearances qualifies as 'S'.

#### AAA Profiles (Task 2)
- Source: Baseball Savant `sport_id=11` endpoint (2023–2026), ~2.19M pitches
- Raw: `data/raw_minors/` (monthly CSVs + manifest.json)
- Processed: `data/processed/statcast_minors_processed.parquet` (2.18M rows)
- Profiles: `data/processed/profiles/aaa/` — 175 profiles built (47 thin-sample blended)
- MLB preference rule: only skip players with a **full** (non-thin) MLB profile; thin-MLB players are eligible for AAA profiles
- ABS descriptions normalized: `automatic_ball` → `ball`, `automatic_strike` → `called_strike`
- AAA recency weights: 2026=3.0x, 2025=2.5x, 2024=2.0x, 2023=1.5x
- `league="AAA"` in predict.py searches AAA dir first, then MLB (fallback_to_mlb=True)

### Current Performance (models_v2)
- Swing AUC: **0.8807** | Contact AUC: **0.8032** | Hard Contact AUC: **0.7277**

## Project Structure
```
pitch-duel/
├── data/raw/                    # Statcast MLB CSVs (gitignored)
├── data/raw_minors/             # Statcast AAA CSVs + manifest.json (gitignored)
├── data/processed/              # parquet files, profiles/, profiles/aaa/, profiles_demo/
├── data/synthetic/              # synthetic pitcher CSVs, per-league prediction CSVs
├── models_v2/                   # active: swing/contact/hard_contact .json + .pkl + metadata
├── models/                      # v1 archive (single hit-prob classifier)
├── src/data/fetch.py            # Statcast MLB download (monthly, idempotent)
├── src/data/fetch_minors.py     # Statcast AAA download (5-day chunks, idempotent)
├── src/data/preprocess.py       # cleaning, feature engineering, ALL_FEATURES
├── src/data/preprocess_minors.py  # AAA cleaning + ABS normalization → parquet
├── src/model/train.py           # three-stage XGBoost training + calibration
├── src/model/predict.py         # three-stage inference, league-aware profile lookup
├── src/model/predict_combined.py  # blended model + similarity lookup
├── src/model/report.py          # generate_league_report(), render_report() — per-league output
├── src/model/simulate_atbat.py  # at-bat sequence simulation
├── src/model/evaluate.py        # single-player evaluation
├── src/model/evaluate_expanded.py  # 20-player tiered validation
├── src/hitters/profiles.py      # per-hitter profile build/save/load/merge; AAA support
├── src/demo/                    # Flask demo app (server.py + templates/)
├── scripts/generate_synthetic_d1_pitcher.py  # 50-pitch Trackman CSV, seed=42
├── scripts/run_synthetic_demo.py  # two separated per-league reports + CSVs
├── scripts/validate_aaa_profiles.py  # 3-gate AAA profile reasonability check
└── docs/aaa_data_investigation.md  # Phase 1 GO decision + API notes
```

## Key Constraints
- venv at `./venv` — always use it
- Raw data never committed (gitignored)
- Delete `data/processed/profiles/` and rebuild whenever train/test split changes
- Never use 2026+ data in MLB profile construction (AAA profiles use full date range — no leakage risk since model was trained on MLB only)
- Pitcher features disconnected — do not re-add without updating ALL_FEATURES and retraining
- Do not print cross-tier comparisons or combined rankings — league reports are always separate

## Validation Benchmarks (Task 1 — synthetic D1 RHP, seed=42)
P(hard) averaged across 50 pitches (15 FF / 15 FC / 10 SL / 10 CH):

| Hitter | P(hard) | Notes |
|--------|---------|-------|
| Shohei Ohtani | 0.310 | MLB elite ceiling |
| Jackson Merrill | 0.295 | MLB elite |
| Aaron Judge | 0.289 | MLB elite ceiling |
| Francisco Lindor | 0.259 | MLB solid |
| Kevin McGonigle | 0.226 | thin-sample floor |
| Munetaka Murakami | 0.225 | thin-sample floor |
| José Ramírez | 0.206 | MLB solid |

AAA validation hitters (Task 2):

| Hitter | P(hard) |
|--------|---------|
| Domínguez, Jasson | 0.310 |
| Baldwin, Drake | 0.303 |
| Dingler, Dillon | 0.288 |
| Narváez, Carlos | 0.270 |
| Kurtz, Nick | 0.257 |

## AAA Validation Gates (scripts/validate_aaa_profiles.py)
| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Gate 1: ceiling | ≤ 0.320 | Ohtani (0.310) + 0.010 headroom — elite AAA vs D1-level pitching is expected to match MLB elite |
| Gate 2: floor | ≥ 0.215 | thin-MLB floor − 0.010 |
| Gate 3: spread | ≥ 0.020 | hitter features must differentiate |
| ~~Gate 4~~ | removed | raw hard_hit_rate is not a valid ordering proxy for composite P(hard) = P(swing) × P(contact\|swing) × P(hard\|contact) |

## Status & Next Steps
**Done:** data pipeline, preprocessing, three-stage XGBoost, calibration, zone/family/contextual hitter features, switch hitter support, 20-player tiered validation, Flask demo on Render (15 curated hitters, demo mode hides Game Log tab), xwOBA regressor (trained, wired into predict_combined), AAA hitter profile pipeline (Task 2), per-league separated report module (Task 3), V5 retrain (89 features, Prompts 2-4), PVHI derived metric (Prompt 5), PVHI redesign as unified kNN lookup (Prompt 6), avg_zone_run_value profile field, debug tab with PIN gate.

**PVHI (Pitch vs. Hitter Index):** Redesigned as a unified 11-dim kNN lookup (7 stuff + 2 location + 2 count). Returns `pvhi`, `pvhi_interpretation`, `pvhi_n_neighbors`, `pvhi_relaxation_level`, `pvhi_similarity_quality` in every `predict_pitch()` output dict. Scale: 100 = this hitter's average pitch, range 0–200. Architecture: find k=30 nearest neighbors in the hitter's historical pitch index (ALL pitches, not just swings; xwOBA=0.0 for non-contact), compute mean xwOBA, divide by hitter's `overall_xwoba_per_pitch` (dimensionally matched), multiply by 100. Blend toward 100 by relaxation confidence (Level 1=1.0 → Level 5=0.0). 5-level cascade: exact type → family → all pitches → stuff-only (7 dims) → fallback. Indexes: `data/processed/pvhi_indexes/{player_id}.npz`, standardization: `models/pvhi_standardization.json`. Old formula (stuff×0.15 + location×0.65 + count×0.20) replaced due to denominator mismatch: zone_xwoba was on-contact (~0.35–0.80) vs per-pitch denominator (~0.079), inflating location component to 700+.

**avg_zone_run_value:** Added to HitterProfile — mean per-pitch xwOBA for strike-zone pitches (zones 1–9 only). Always > `overall_xwoba_per_pitch` because ball-zone pitches (mostly 0.0) are excluded. League avg: 0.115.

**Debug tab:** Password-gated tab (PIN: 0001) in the tablet frontend shows all 89 feature values, hitter profile snapshot, and full PVHI kNN computation trace (query vector, relaxation level, neighbor stats, raw/blended/final PVHI).

**Next:** PVHI weight optimization via outcome regression, count-specific hitter rates (hard_contact only), pitcher features re-integration, Phase 2 LSTM/Transformer sequencing.
