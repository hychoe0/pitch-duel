# Pitch Duel — Project Context

## What It Is
Competitive exposure simulator for pitchers. Core question: "How good is my pitch against specific hitters?"
Named hitter simulation (e.g. Ohtani, KBO cleanup hitters) is the differentiator — tier fallback for thin samples only.
Origin: KBO/international pitchers have no benchmark against elite hitters. Exposed at every WBC.

## Model Architecture
**Phase 1 (current):** XGBoost classifier predicting per-pitch hit probability (0–1).
**Phase 2 (future):** LSTM/Transformer for full at-bat sequences, tunneling, sequencing effects.

### Feature Groups (74 total)
- **Pitch quality (11):** release_speed, spin_to_velo_ratio, pfx_x/z, release_pos_x/z, release_extension, plate_x/z, pitch_type_enc, era_flag_enc
- **Context (11):** balls, strikes, pitch_number, prev_pitch_type/speed/result_enc, on_1b/2b/3b_flag, inning, score_diff
- **Hitter profile — aggregate (7):** swing/chase/contact/hard_hit/whiff rates, hitter_stand_enc, p_throws_enc
- **Hitter profile — zone rates (28):** swing + whiff rate per Statcast zone (1–9 strike, 11–14 ball)
- **Hitter profile — pitch family rates (8):** swing + whiff rate per family (fastball/breaking/offspeed/other)
- **Hitter profile — per-pitch-type contact (11):** contact rate by individual pitch type

### Data & Splits
- **Training:** Statcast 2015–2024 (excluding 2020 COVID season), ~6.5M pitches
- **Test:** 2025–2026 (fully unseen), ~700K pitches
- **Early stopping:** 2024 season carved from training
- **Profile cutoff:** 2025-01-01 (no leakage into test set)

### Era Integrity
Raw spin rate is distorted across eras — use `spin_to_velo_ratio` instead.
Era flag: pre_crackdown (<Jun 21 2021), post_crackdown (Jun–Dec 2021), ambiguous (2022), pitch_clock (2023+).
Sample weights: 2026=2.0x, 2025=1.8x, 2024=1.5x, 2023=1.2x, 2022=0.6x, post_crackdown=0.7x, pre_crackdown=0.3x, 2020=excluded.

### Hitter Profiles
Built from Statcast with recency weights: 2026=3.0x → 2021 and earlier=0.5x.
Minimum 200 weighted PA before using named profile; below that, blend with league averages.
Zone rates use Statcast's native `zone` column (hitter-specific strike zone calibration).
Pitch families: fastball (FF/SI/FC), breaking (SL/CU/KC/SV/CS), offspeed (CH/FS/FO), other.
Fallback thresholds: 20 weighted pitches per zone, 30 per family.

### Calibration
Isotonic regression layer on 2024 val set corrects scale_pos_weight overconfidence.
Calibration error: 0.33 → 0.003–0.013. Brier: ~0.21 → 0.049–0.065. AUC unchanged ~0.78.

### Current Performance
- ROC-AUC: 0.7805 | Brier: 0.1877 (raw), 0.049–0.065 (calibrated)
- Top features: plate_z (13.8%), plate_x (10.4%), zone rates (24.9%), family rates (5.1%)
- Total hitter profile importance: 43.1% (up from 11.6% before zone/family features)

## Project Structure
```
pitch-duel/
├── data/raw/                    # Statcast CSVs (gitignored)
├── data/processed/              # parquet files, profiles/, eval_2025/
├── models/                      # xgb.json, calibrator.pkl, feature_cols.json, encodings.json, metrics.json
├── src/data/fetch.py            # Statcast download (monthly, idempotent)
├── src/data/preprocess.py       # cleaning, feature engineering, ALL_FEATURES
├── src/model/train.py           # XGBoost training + calibrator
├── src/model/predict.py         # inference (calibrated)
├── src/model/evaluate.py        # single-player evaluation
├── src/model/evaluate_expanded.py  # 20-player tiered validation
├── src/hitters/profiles.py      # per-hitter profile build/save/load/merge
└── CLAUDE.md
```

## Key Constraints
- venv at `./venv` — always use it
- Raw data never committed (gitignored)
- Delete `data/processed/profiles/` and rebuild whenever train/test split changes
- Never use 2025+ data in profile construction
- Phase 1 before Phase 2

## Status & Next Steps
**Done:** data pipeline, preprocessing, XGBoost model, calibration, 20-player tiered validation, zone/family hitter features.
**Next:** contextual zone lookup (matched zone/family features instead of all 28+8 static), count-specific hitter rates, narrative at-bat demo for coach outreach, xwOBA target (Phase 1.5).