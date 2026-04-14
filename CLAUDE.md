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

### Current Performance (models_v2)
- Swing AUC: **0.8807** | Contact AUC: **0.8032** | Hard Contact AUC: **0.7277**

## Project Structure
```
pitch-duel/
├── data/raw/                    # Statcast CSVs (gitignored)
├── data/processed/              # parquet files, profiles/, profiles_demo/
├── models_v2/                   # active: swing/contact/hard_contact .json + .pkl + metadata
├── models/                      # v1 archive (single hit-prob classifier)
├── src/data/fetch.py            # Statcast download (monthly, idempotent)
├── src/data/preprocess.py       # cleaning, feature engineering, ALL_FEATURES
├── src/model/train.py           # three-stage XGBoost training + calibration
├── src/model/predict.py         # three-stage inference
├── src/model/simulate_atbat.py  # at-bat sequence simulation
├── src/model/evaluate.py        # single-player evaluation
├── src/model/evaluate_expanded.py  # 20-player tiered validation
├── src/hitters/profiles.py      # per-hitter profile build/save/load/merge
└── src/demo/                    # Flask demo app (server.py + templates/)
```

## Key Constraints
- venv at `./venv` — always use it
- Raw data never committed (gitignored)
- Delete `data/processed/profiles/` and rebuild whenever train/test split changes
- Never use 2026+ data in profile construction
- Pitcher features disconnected — do not re-add without updating ALL_FEATURES and retraining

## Status & Next Steps
**Done:** data pipeline, preprocessing, three-stage XGBoost, calibration, zone/family/contextual hitter features, switch hitter support, 20-player tiered validation, Flask demo on Render (15 curated hitters, demo mode hides Game Log tab).
**Next:** xwOBA target (Phase 1.5), count-specific hitter rates (hard_contact only), pitcher features re-integration, Phase 2 LSTM/Transformer sequencing.
