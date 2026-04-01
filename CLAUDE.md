# Pitch Duel — Project Context

## What This Is
Pitch Duel is a competitive exposure simulator for pitchers who have never faced 
world-class hitters. The core question it answers:
"How good is my pitch against different hitters at different levels?"

Origin insight: KBO/international pitchers have no honest benchmark against elite 
hitters — exposed at every WBC. Pitch Duel simulates real matchups against specific 
named players across multiple leagues and levels: MLB, MiLB, KBO, NPB, and beyond.

## Primary Differentiator
Named hitter simulation at any level — not a generic tier, but a specific player 
(e.g. Shohei Ohtani, a specific KBO cleanup hitter, a Triple-A prospect).
Tier simulation is the fallback for thin sample players only.

The deeper differentiator: Pitch Duel models full at-bat sequences, not single 
pitches in isolation. Hit probability updates dynamically as the at-bat unfolds — 
accounting for count, pitch sequencing, tunneling effects, and situational pressure.

## What Makes a Pitch "Good"
A pitch is not good or bad in isolation. It is good or bad in context:
- Pitch quality: velocity, spin rate, movement, release point, extension, location
- Sequence context: what pitch came before, velocity/type contrast, tunneling
- Count state: balls/strikes change hitter zone coverage and aggression
- In-at-bat history: has the hitter already chased this pitch type?
- Situational pressure: runners on base, inning, score differential, leverage index
All of these are encoded in Statcast and used as model features.

## Model Architecture
### Phase 1 — Feature-Enriched Single Pitch Model (current target)
XGBoost classifier. Each pitch row includes:
Pitch quality features:
- release_speed, release_spin_rate, pfx_x, pfx_z (movement)
- release_pos_x, release_pos_z, release_extension
- plate_x, plate_z (location)
- pitch_type

Contextual/sequential features:
- balls, strikes, pitch_number_in_pa
- prev_pitch_type, prev_pitch_speed, prev_pitch_result
- on_1b, on_2b, on_3b, inning, score_diff

Hitter profile features (from named hitter Statcast history):
- swing_rate, chase_rate, contact_rate, hard_hit_rate, whiff_rate
- platoon split (batter/pitcher handedness)

Output: hit probability (0–1) for this pitch in this situation against this hitter.

### Phase 2 — Sequential At-Bat Model (future)
LSTM or Transformer that models the full pitch sequence as a series.
Captures tunneling and sequencing effects more naturally.
Build after Phase 1 is validated.

## Tech Stack
- Language: Python
- ML model: XGBoost (Phase 1), LSTM/Transformer (Phase 2)
- Data: MLB Statcast 2015–2026 (most recent available), Triple-A/FSL MiLB
- Key library: pybaseball (pulls from Baseball Savant)
- Hardware input: agnostic — Trackman, Rapsodo, or any tracker

## Data Weights & Era Integrity

### Training Sample Weights (target: current pitch-clock era baseball)
| Period | Weight | Reasoning |
|---|---|---|
| 2026 partial | 2.0x | Most current |
| 2025 | 1.8x | Full pitch-clock season |
| 2024 | 1.5x | Full pitch-clock season |
| 2023 | 1.2x | First pitch-clock season — players still adapting |
| 2022 | 0.6x | Spin ambiguous — pitchers creeping back to sticky stuff |
| Jun 21–Dec 2021 | 0.7x | Cleanest spin data in modern era post-crackdown |
| Pre Jun 21 2021 | 0.3x | Artificially inflated spin — Spider Tack era |
| 2020 | 0.0x | Excluded — COVID season, no fans, abnormal |
| 2017–2019 | 0.3x | Pitch quality features only, not count/sequence features |

### Spin Rate Data Integrity Problem
The same RPM number means different things across eras:
- Pre Jun 2021: artificially inflated via Spider Tack and foreign substances
- Jun–Dec 2021: cleanest spin data in modern history post-crackdown
- 2022: ambiguous — pitchers found workarounds to umpire checks
- 2023–2026: relatively clean with stricter protocols

### Spin Rate Feature Engineering (required in preprocess.py)
Use spin-to-velocity ratio (SVR) instead of raw spin rate as the primary feature:
  df["spin_to_velo_ratio"] = df["release_spin_rate"] / df["release_speed"]

Add an era flag so the model can learn residual era effects:
  - "pre_crackdown"  → before Jun 21, 2021
  - "post_crackdown" → Jun 21–Dec 2021
  - "ambiguous"      → 2022
  - "pitch_clock"    → 2023 onward

## Hitter Profile Recency Weighting
Named hitter profiles are built from Statcast but weighted aggressively toward
recent seasons. A hitter's 2019 chase rate is largely irrelevant if 3+ seasons
of pitch-clock data exist on them.

Recency weights for hitter profiles:
  2026: 3.0x
  2025: 2.5x
  2024: 2.0x
  2023: 1.5x
  2022: 1.0x
  2021 and earlier: 0.5x

Apply as frequency weights to all rate calculations (swing rate, chase rate,
contact rate, hard-hit rate, whiff rate, count behavior).
Minimum sample threshold: 200 weighted plate appearances before using a
named hitter profile. Fall back to tier profile below this threshold.

## Project Structure
pitch-duel/
├── data/
│   ├── raw/            # downloaded Statcast CSVs
│   └── processed/      # cleaned, model-ready data
├── models/             # saved trained models
├── notebooks/          # Jupyter exploration
├── src/
│   ├── data/
│   │   ├── fetch.py        # pulls Statcast data
│   │   └── preprocess.py   # cleans and engineers features
│   ├── model/
│   │   ├── train.py        # trains the model
│   │   └── predict.py      # runs predictions
│   └── hitters/
│       └── profiles.py     # builds individual hitter profiles
└── CLAUDE.md

## Hitter Profile Logic
Each named hitter profile encodes (built from Statcast via pybaseball):
- Swing/chase rates by zone and count
- Contact rate and hard-hit rate by pitch type
- Platoon splits
- Count behavior (aggression by ball-strike count)
- In-at-bat tendencies (first pitch swing rate, 2-strike approach)

<!-- ## Model Validation Approach
Temporal split: train on 2015–2022, test on 2023–2026 unseen data.
Secondary: validate on repeated pitcher-hitter matchups, compare predicted 
hit probability vs actual wOBA in those matchups. -->

## Current Development Stage
Scaffold is set up. Next steps:
1. Confirm data pipeline works (fetch.py pulling Statcast data cleanly)
2. Build preprocess.py to engineer pitch + contextual features
3. Build Phase 1 XGBoost hit probability model in train.py

## Key Constraints
- venv is located at ./venv — always use it
- Raw data files are large — never commit to git (already in .gitignore)
- Prioritize named hitter output; tier fallback is secondary
- Phase 1 before Phase 2 — validate the single-pitch model before 
  building the sequential model