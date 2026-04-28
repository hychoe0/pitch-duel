# xwOBA Regressor (Task 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a per-pitch xwOBA XGBoost regressor, wire it into the prediction pipeline, and validate that Ramírez ranks higher on xwOBA than on P(hard) vs. the synthetic D1 pitcher.

**Architecture:** Standalone `src/model/train_xwoba.py` computes a per-pitch `xwoba_target` column (0.0 for non-damage pitches, woba_value for walks/HBPs, estimated_woba for batted balls), trains XGBRegressor on ALL pitches, and saves to `models/xwoba_model.json` + `models/xwoba_calibrator.pkl`. `predict.py` already loads and returns these when present (as `predicted_xwoba_on_contact`). `report.py` and `predict_combined.py` are already wired — they activate automatically once the model file exists.

**Tech Stack:** XGBoost (XGBRegressor), scikit-learn IsotonicRegression, pandas, numpy; Python venv at `./venv`.

---

## Codebase State (read before touching anything)

**Already wired — DO NOT MODIFY unless a task explicitly says to:**

| File | What it does for xwOBA |
|------|------------------------|
| `src/model/predict.py` | Loads `models/xwoba_model.json` if present; returns `predicted_xwoba_on_contact` in `predict_pitch()` result dict |
| `src/model/predict_combined.py` | Reads `predicted_xwoba_on_contact` from `predict_pitch()`; stores in `MatchupResult.predicted_xwoba_on_contact` |
| `src/model/report.py` | Reads `r.predicted_xwoba_on_contact`; populates `avg_xwoba` in hitter rows; renders xwOBA column (shows N/A when None) |
| `src/model/train.py` | Has `train_xwoba_stage()` (on-contact approach — trains on in-play rows only). This is in the full pipeline but **will be superseded** by `train_xwoba.py` which uses the per-pitch approach. |

**What's missing:**

1. `models/xwoba_model.json` — not trained yet
2. `models/xwoba_calibrator.pkl` — not trained yet
3. `src/model/train_xwoba.py` — standalone per-pitch training script
4. `predict_xwoba()` helper function in `predict.py`
5. Rank comparison output in `scripts/run_synthetic_demo.py`

**Critical data facts:**

- Parquet: `data/processed/statcast_processed.parquet` (7.07M rows, 206 columns)
- Present columns: `estimated_woba_using_speedangle` (25.3% non-null), `woba_value` (25.6% non-null), `events`, `description`
- **No** `target_xwoba_on_contact` or `xwoba_target` column exists yet — compute on the fly
- Feature list: `ALL_FEATURES` from `src/data/preprocess.py` (80 features)
- Model naming: predict.py loads `models/xwoba_model.json` — use this path (not `pitch_duel_xwoba_xgb.json`)

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| **Create** | `src/model/train_xwoba.py` | Per-pitch xwOBA training pipeline |
| **Modify** | `src/model/predict.py` | Add `predict_xwoba()` standalone helper |
| **Modify** | `scripts/run_synthetic_demo.py` | Add rank comparison table to output |
| **Produced** | `models/xwoba_model.json` | Trained XGBRegressor |
| **Produced** | `models/xwoba_calibrator.pkl` | Isotonic calibrator |
| **Produced** | `data/processed/statcast_processed.parquet` | Updated with `xwoba_target` column |

---

## Task 1: Build `src/model/train_xwoba.py` — Target Construction and Validation

**Files:**
- Create: `src/model/train_xwoba.py`

The per-pitch xwOBA target logic:
- Batted ball rows (`description ∈ IN_PLAY_DESCRIPTIONS` AND `estimated_woba_using_speedangle` not NaN): use `estimated_woba_using_speedangle`
- Walk/HBP rows (`events ∈ {'walk','intent_walk','hit_by_pitch'}` AND `woba_value` not NaN): use `woba_value`
- All other pitches: 0.0

- [ ] **Step 1: Create `src/model/train_xwoba.py` with target computation and validation**

```python
"""
train_xwoba.py — Per-pitch xwOBA regressor (parallel to three-stage classifiers).

Target: per-pitch expected xwOBA
  - Batted ball: estimated_woba_using_speedangle
  - Walk / HBP: woba_value
  - All other pitches: 0.0

Most pitches get target=0.0 (non-damage) — the model learns that most pitches
don't generate damage. Do NOT filter to batted balls only.

Saves to: models/xwoba_model.json, models/xwoba_calibrator.pkl
Updates:  data/processed/statcast_processed.parquet (adds xwoba_target column)

Usage:
    python -m src.model.train_xwoba
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.preprocess import ALL_FEATURES, IN_PLAY_DESCRIPTIONS

PROCESSED_PATH = Path("data/processed/statcast_processed.parquet")
MODEL_DIR      = Path("models")

WALK_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS  = {"hit_by_pitch"}

XWOBA_REG_PARAMS = {
    "n_estimators":        500,
    "max_depth":           6,
    "learning_rate":       0.05,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "min_child_weight":    50,
    "objective":           "reg:squarederror",
    "eval_metric":         "rmse",
    "random_state":        42,
    "n_jobs":              -1,
    "tree_method":         "hist",
    "early_stopping_rounds": 30,
}


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def _compute_xwoba_target(df: pd.DataFrame) -> pd.Series:
    """
    Build per-pitch xwOBA target.

    - Batted ball + valid speed-angle xwOBA: use estimated_woba_using_speedangle
    - Walk / HBP: use woba_value
    - Everything else: 0.0
    """
    if "estimated_woba_using_speedangle" not in df.columns:
        raise ValueError(
            "Column 'estimated_woba_using_speedangle' not in parquet. "
            "These come directly from Statcast — do not fabricate."
        )
    if "woba_value" not in df.columns:
        raise ValueError(
            "Column 'woba_value' not in parquet. "
            "These come directly from Statcast — do not fabricate."
        )

    target = pd.Series(0.0, index=df.index, dtype=float)

    batted_mask = (
        df["description"].isin(IN_PLAY_DESCRIPTIONS)
        & df["estimated_woba_using_speedangle"].notna()
    )
    target[batted_mask] = df.loc[batted_mask, "estimated_woba_using_speedangle"]

    walk_hbp_mask = (
        df["events"].isin(WALK_EVENTS | HBP_EVENTS)
        & df["woba_value"].notna()
    )
    target[walk_hbp_mask] = df.loc[walk_hbp_mask, "woba_value"]

    return target


def load_data() -> pd.DataFrame:
    """Load processed parquet and add xwoba_target column."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_PATH} not found. Run preprocess.py first.")
    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {len(df):,} rows from {PROCESSED_PATH}")
    df["xwoba_target"] = _compute_xwoba_target(df)
    return df


# ---------------------------------------------------------------------------
# Target validation — STOP on anomalies
# ---------------------------------------------------------------------------

def validate_target_distribution(df: pd.DataFrame) -> None:
    """Print distribution stats. Raises if anomalous."""
    t = df["xwoba_target"]

    pct_zero      = (t == 0.0).mean() * 100
    pct_0_to_05   = ((t > 0.0) & (t <= 0.5)).mean() * 100
    pct_05_to_1   = ((t > 0.5) & (t <= 1.0)).mean() * 100
    pct_above_1   = (t > 1.0).mean() * 100

    print("\n=== Target Distribution ===")
    print(f"  At 0.0:         {pct_zero:.1f}%")
    print(f"  (0.0, 0.5]:     {pct_0_to_05:.1f}%")
    print(f"  (0.5, 1.0]:     {pct_05_to_1:.1f}%")
    print(f"  > 1.0:          {pct_above_1:.1f}%")
    print(f"  Overall mean:   {t.mean():.4f}")
    print(f"  Non-zero mean:  {t[t > 0].mean():.4f}")

    if pct_zero > 99.0:
        raise ValueError(
            f"STOP: {pct_zero:.1f}% of targets are 0.0. "
            "Expected ~74-75% (roughly swing rate × non-in-play rate). "
            "Check target construction logic."
        )
    if t.mean() > 0.4:
        raise ValueError(
            f"STOP: Mean target {t.mean():.4f} > 0.4. "
            "Per-pitch xwOBA should average ~0.04–0.08. "
            "Check target construction — possible batted-ball-only filtering."
        )

    # By outcome category
    in_play = df["description"].isin(IN_PLAY_DESCRIPTIONS)
    is_walk = df["events"].isin(WALK_EVENTS)
    is_hbp  = df["events"].isin(HBP_EVENTS)
    is_so   = df["events"].isin({"strikeout", "strikeout_double_play"})

    print("\n  Mean target by outcome category:")
    print(f"    Batted ball:    {df.loc[in_play,  'xwoba_target'].mean():.4f}  (n={in_play.sum():,})")
    print(f"    Walk:           {df.loc[is_walk,  'xwoba_target'].mean():.4f}  (n={is_walk.sum():,})")
    print(f"    HBP:            {df.loc[is_hbp,   'xwoba_target'].mean():.4f}  (n={is_hbp.sum():,})")
    print(f"    Strikeout:      {df.loc[is_so,    'xwoba_target'].mean():.4f}  (n={is_so.sum():,})")
    other = ~in_play & ~is_walk & ~is_hbp & ~is_so
    print(f"    Other:          {df.loc[other,    'xwoba_target'].mean():.4f}  (n={other.sum():,})")

    # Year-over-year drift check
    df["_year"] = df["game_date"].dt.year
    yoy = df.groupby("_year")["xwoba_target"].mean()
    df.drop(columns=["_year"], inplace=True)
    print("\n  Year-over-year mean xwoba_target:")
    for yr, val in yoy.items():
        print(f"    {yr}: {val:.4f}")


# ---------------------------------------------------------------------------
# Splits and NaN filling
# ---------------------------------------------------------------------------

def prepare_splits(df: pd.DataFrame) -> tuple:
    """
    Temporal split matching train.py:
      fit:  < 2024-01-01
      val:  2024 (early stopping)
      test: >= 2025-01-01

    Returns (fit_df, val_df, test_df, col_medians).
    """
    train_cutoff = pd.Timestamp("2025-01-01")
    val_cutoff   = pd.Timestamp("2024-01-01")

    full_train = df[df["game_date"] < train_cutoff].copy()
    test_df    = df[df["game_date"] >= train_cutoff].copy()
    fit_df     = full_train[full_train["game_date"] < val_cutoff].copy()
    val_df     = full_train[full_train["game_date"] >= val_cutoff].copy()

    print(f"\nSplits — fit: {len(fit_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    col_medians = fit_df[ALL_FEATURES].median()

    for split, label in [(fit_df, "fit"), (val_df, "val"), (test_df, "test")]:
        nan_cols = split[ALL_FEATURES].columns[split[ALL_FEATURES].isna().any()].tolist()
        if nan_cols:
            print(f"  NaN columns in {label} (filling with fit medians): {nan_cols}")
        split[ALL_FEATURES] = split[ALL_FEATURES].fillna(col_medians)

    return fit_df, val_df, test_df, col_medians


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    col_medians: pd.Series,
) -> tuple:
    """Train XGBRegressor with early stopping on val RMSE."""
    X_fit = fit_df[ALL_FEATURES].values
    y_fit = fit_df["xwoba_target"].values
    w_fit = fit_df["sample_weight"].values if "sample_weight" in fit_df.columns else np.ones(len(fit_df))

    X_val = val_df[ALL_FEATURES].values
    y_val = val_df["xwoba_target"].values

    for name, arr in [("X_fit", X_fit), ("X_val", X_val)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            raise ValueError(f"{n_nan} NaN values remain in {name}.")

    print(f"\nFit rows: {len(X_fit):,}  |  Val rows: {len(X_val):,}")
    print(f"Fit target mean: {y_fit.mean():.4f}  |  Val target mean: {y_val.mean():.4f}")

    model = xgb.XGBRegressor(**XWOBA_REG_PARAMS)
    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration}")

    val_preds = model.predict(X_val)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_preds, y_val)
    cal_preds = calibrator.predict(val_preds)
    print(f"Val calibration — raw mean: {val_preds.mean():.4f}  "
          f"cal mean: {cal_preds.mean():.4f}  actual mean: {y_val.mean():.4f}")

    return model, calibrator


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: xgb.XGBRegressor,
    calibrator: IsotonicRegression,
    test_df: pd.DataFrame,
) -> None:
    """Print RMSE/MAE/R² on full test set, batted balls only, and by outcome."""
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df["xwoba_target"].values

    raw_preds = model.predict(X_test)
    preds = np.clip(calibrator.predict(raw_preds), 0.0, 2.0)

    rmse_full = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae_full  = float(mean_absolute_error(y_test, preds))
    r2_full   = float(r2_score(y_test, preds))

    in_play = test_df["description"].isin(IN_PLAY_DESCRIPTIONS)
    y_bb    = y_test[in_play.values]
    p_bb    = preds[in_play.values]
    rmse_bb = float(np.sqrt(mean_squared_error(y_bb, p_bb)))
    mae_bb  = float(mean_absolute_error(y_bb, p_bb))
    r2_bb   = float(r2_score(y_bb, p_bb))

    print("\n=== Test Set Metrics ===")
    print(f"Full test set:")
    print(f"  RMSE: {rmse_full:.3f}  MAE: {mae_full:.3f}  R²: {r2_full:.3f}  (n={len(y_test):,})")
    print(f"Batted balls only:")
    print(f"  RMSE: {rmse_bb:.3f}  MAE: {mae_bb:.3f}  R²: {r2_bb:.3f}  (n={in_play.sum():,})")

    if rmse_bb > 0.6:
        print(f"\nWARNING: Batted-ball RMSE {rmse_bb:.3f} > 0.6. Expected 0.30–0.45. "
              "Investigate target construction or feature alignment.")

    # By outcome category
    is_walk = test_df["events"].isin(WALK_EVENTS)
    is_hbp  = test_df["events"].isin(HBP_EVENTS)
    is_so   = test_df["events"].isin({"strikeout", "strikeout_double_play"})
    other   = ~in_play & ~is_walk & ~is_hbp & ~is_so

    print("\nBy outcome:")
    for label, mask in [("Batted ball", in_play), ("Walk", is_walk),
                         ("HBP", is_hbp), ("Strikeout", is_so), ("Other", other)]:
        if mask.sum() == 0:
            continue
        mp = preds[mask.values].mean()
        ma = y_test[mask.values].mean()
        print(f"  {label:<12}: mean predicted {mp:.3f} vs mean actual {ma:.3f}  (n={mask.sum():,})")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(model: xgb.XGBRegressor, calibrator: IsotonicRegression) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xwoba_model.json"))
    joblib.dump(calibrator, MODEL_DIR / "xwoba_calibrator.pkl")
    print(f"\nSaved: {MODEL_DIR}/xwoba_model.json, {MODEL_DIR}/xwoba_calibrator.pkl")


def save_parquet_with_target(df: pd.DataFrame) -> None:
    """Persist xwoba_target column back to the parquet."""
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Updated parquet saved with 'xwoba_target' column → {PROCESSED_PATH}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_training() -> None:
    df = load_data()
    validate_target_distribution(df)
    fit_df, val_df, test_df, col_medians = prepare_splits(df)
    model, calibrator = train_model(fit_df, val_df, col_medians)
    evaluate_model(model, calibrator, test_df)
    save_model(model, calibrator)
    save_parquet_with_target(df)


if __name__ == "__main__":
    run_training()
```

- [ ] **Step 2: Verify the file is syntactically valid**

```bash
source venv/bin/activate && python -c "import src.model.train_xwoba; print('Import OK')"
```

Expected: `Import OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/model/train_xwoba.py
git commit -m "feat: add per-pitch xwOBA regressor training script (Task 4)"
```

---

## Task 2: Add `predict_xwoba()` Helper to `predict.py`

**Files:**
- Modify: `src/model/predict.py` (add one function after `predict_pitch()`)

The spec requires a `predict_xwoba(pitch, hitter_name, league)` convenience function. `predict_pitch()` already returns `predicted_xwoba_on_contact` — this is just a thin wrapper.

- [ ] **Step 1: Add `predict_xwoba()` to `predict.py` after `predict_pitch()`**

Open `src/model/predict.py` and add this function immediately after `predict_pitch()` ends (after line 590, before the CLI section):

```python
def predict_xwoba(
    pitch: dict,
    hitter_name: str,
    model_dir: Path = MODEL_DIR,
    league: str = "MLB",
    fallback_to_mlb: bool = True,
) -> float | None:
    """
    Return predicted per-pitch xwOBA for a single pitch against a named hitter.

    Returns None if the xwOBA model has not been trained yet.
    This is a thin wrapper around predict_pitch() for callers who only want xwOBA.
    """
    result = predict_pitch(pitch, hitter_name, model_dir, league=league,
                           fallback_to_mlb=fallback_to_mlb)
    return result.get("predicted_xwoba_on_contact")
```

- [ ] **Step 2: Verify import**

```bash
source venv/bin/activate && python -c "
from src.model.predict import predict_xwoba
print('predict_xwoba imported OK')
import inspect
print('Signature:', inspect.signature(predict_xwoba))
"
```

Expected:
```
predict_xwoba imported OK
Signature: (pitch: dict, hitter_name: str, model_dir: pathlib.Path = PosixPath('models'), league: str = 'MLB', fallback_to_mlb: bool = True) -> float | None
```

- [ ] **Step 3: Verify backward compat — `predict_pitch()` return dict still has all old keys**

```bash
source venv/bin/activate && python -c "
import warnings; warnings.filterwarnings('ignore')
from src.model.predict import predict_pitch
p = {
    'release_speed': 92.0, 'release_spin_rate': 2200, 'pfx_x': -0.4, 'pfx_z': 1.1,
    'release_pos_x': -1.8, 'release_pos_z': 6.1, 'release_extension': 6.5,
    'plate_x': 0.0, 'plate_z': 2.5, 'pitch_type': 'FF',
    'balls': 1, 'strikes': 1, 'pitch_number': 2,
    'prev_pitch_type': 'FF', 'prev_pitch_speed': 91.0, 'prev_pitch_result': 'ball',
    'on_1b': None, 'on_2b': None, 'on_3b': None,
    'inning': 5, 'score_diff': 0, 'p_throws': 'R',
}
r = predict_pitch(p, 'Shohei Ohtani')
required = ['p_swing','p_contact_given_swing','p_hard_given_contact','p_contact','p_hard_contact','pitch_quality','zone','hitter','pitch_type','count','is_thin_sample','used_fallback','predicted_xwoba_on_contact','xwoba_context']
missing = [k for k in required if k not in r]
print('Missing keys:', missing)
print('All old keys present:', not missing)
print('predicted_xwoba_on_contact:', r.get('predicted_xwoba_on_contact'))
"
```

Expected: `All old keys present: True`. `predicted_xwoba_on_contact` will be `None` until the model is trained — that's correct.

- [ ] **Step 4: Commit**

```bash
git add src/model/predict.py
git commit -m "feat: add predict_xwoba() helper to predict.py"
```

---

## Task 3: Add Rank Comparison Output to `run_synthetic_demo.py`

**Files:**
- Modify: `scripts/run_synthetic_demo.py`

The spec asks for a P(hard) rank vs xwOBA rank comparison table printed alongside the demo output.

- [ ] **Step 1: Add `_print_rank_comparison()` and wire it into `run()`**

Open `scripts/run_synthetic_demo.py`. After the existing `run()` function's final `print(render_report(aaa_report))` call and before the closing `print()`, add:

```python
    # ── Rank comparison: P(hard) vs xwOBA ────────────────────────────────────
    _print_rank_comparison("MLB", mlb_report)
    _print_rank_comparison("AAA", aaa_report)
```

And add this helper function before `run()`:

```python
def _print_rank_comparison(league: str, report: dict) -> None:
    """Print P(hard) rank vs xwOBA rank table. Flag hitters who move 2+ positions."""
    hitters = report["hitters"]
    if not hitters:
        return

    # P(hard) rank is already the sort order of hitters list (sorted by avg_p_hard desc)
    phhard_order = [h["name"] for h in hitters]

    xwoba_hitters = [h for h in hitters if h.get("avg_xwoba") is not None]
    if not xwoba_hitters:
        print(f"\n[{league}] xwOBA not available — model not yet trained.")
        return

    xwoba_order = sorted(xwoba_hitters, key=lambda h: -(h["avg_xwoba"] or 0))
    xwoba_names = [h["name"] for h in xwoba_order]

    print(f"\n{'─' * 64}")
    print(f"  [{league}] P(hard) rank vs xwOBA rank comparison")
    print(f"{'─' * 64}")
    print(f"  {'Hitter':<26}  {'P(hard)':<9}  {'P(hard)#':<9}  {'xwOBA':<8}  {'xwOBA#':<7}  {'Δ'}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*4}")

    for h in hitters:
        name   = h["name"]
        ph_val = h["avg_p_hard"]
        ph_rank = phhard_order.index(name) + 1 if name in phhard_order else "?"
        xw_val  = h.get("avg_xwoba")
        if xw_val is not None and name in xwoba_names:
            xw_rank = xwoba_names.index(name) + 1
            delta   = xw_rank - ph_rank
            flag    = "  ← MOVED" if abs(delta) >= 2 else ""
            print(f"  {name:<26}  {ph_val:<9.3f}  #{ph_rank:<8}  {xw_val:<8.3f}  #{xw_rank:<6}  {delta:+d}{flag}")
        else:
            print(f"  {name:<26}  {ph_val:<9.3f}  #{ph_rank:<8}  {'N/A':<8}  {'N/A':<6}  n/a")
    print(f"{'─' * 64}\n")
```

- [ ] **Step 2: Verify the script still imports and is syntactically valid (model not required yet)**

```bash
source venv/bin/activate && python -c "
import ast, pathlib
src = pathlib.Path('scripts/run_synthetic_demo.py').read_text()
ast.parse(src)
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_synthetic_demo.py
git commit -m "feat: add P(hard) vs xwOBA rank comparison table to synthetic demo"
```

---

## Task 4: Train the xwOBA Regressor

**Files:**
- No code changes — runs the training script from Task 1

This step takes time (XGBoost on 5M+ rows). Expected runtime: 5–15 minutes.

- [ ] **Step 1: Run target validation only (fast sanity check before full training)**

```bash
source venv/bin/activate && python -c "
from src.model.train_xwoba import load_data, validate_target_distribution
df = load_data()
validate_target_distribution(df)
"
```

Expected output includes:
- `At 0.0: ~74-76%` (non-damage pitches)
- `(0.0, 0.5]: ~20-22%`
- `Overall mean: 0.04–0.08`
- Batted ball mean: `~0.35–0.45`
- Walk mean: `~0.69–0.71`
- Year-over-year means relatively stable (±0.01 per year)

If you see `At 0.0: > 99%` or `Overall mean > 0.4` — STOP and report. Do not proceed to training.

- [ ] **Step 2: Run full training**

```bash
source venv/bin/activate && python -m src.model.train_xwoba
```

Expected output structure:
```
Loaded 7,073,894 rows from data/processed/statcast_processed.parquet

=== Target Distribution ===
  At 0.0:         ~75.X%
  (0.0, 0.5]:     ~20.X%
  ...

Splits — fit: X,XXX,XXX  val: X,XXX,XXX  test: XXX,XXX

[XGBoost training output with RMSE decreasing every 50 rounds]
Best iteration: ~200-400

=== Test Set Metrics ===
Full test set:
  RMSE: 0.0XX  MAE: 0.0XX  R²: 0.XXX  (n=XXX,XXX)
Batted balls only:
  RMSE: 0.XXX  MAE: 0.XXX  R²: 0.XXX  (n=XXX,XXX)
By outcome:
  Batted ball : mean predicted 0.XXX vs mean actual 0.XXX
  Walk        : mean predicted 0.XXX vs mean actual 0.XXX
  ...

Saved: models/xwoba_model.json, models/xwoba_calibrator.pkl
Updated parquet saved with 'xwoba_target' column
```

**Stop conditions:**
- If `Batted balls only RMSE > 0.6` — STOP and report; something is wrong
- If `mean predicted batted ball < 0.05` — model may have collapsed to zero-predicting; report

- [ ] **Step 3: Verify model files exist**

```bash
ls -lh models/xwoba_model.json models/xwoba_calibrator.pkl
```

Expected: both files present, `xwoba_model.json` should be several MB.

- [ ] **Step 4: Quick inference smoke test**

```bash
source venv/bin/activate && python -c "
import warnings; warnings.filterwarnings('ignore')
from src.model.predict import predict_pitch, predict_xwoba

p = {
    'release_speed': 92.0, 'release_spin_rate': 2200, 'pfx_x': -0.4, 'pfx_z': 1.1,
    'release_pos_x': -1.8, 'release_pos_z': 6.1, 'release_extension': 6.5,
    'plate_x': 0.0, 'plate_z': 2.5, 'pitch_type': 'FF',
    'balls': 1, 'strikes': 1, 'pitch_number': 2,
    'prev_pitch_type': 'FF', 'prev_pitch_speed': 91.0, 'prev_pitch_result': 'ball',
    'on_1b': None, 'on_2b': None, 'on_3b': None,
    'inning': 5, 'score_diff': 0, 'p_throws': 'R',
}

r = predict_pitch(p, 'Shohei Ohtani')
xwoba = predict_xwoba(p, 'Shohei Ohtani')
print(f'predicted_xwoba_on_contact (in predict_pitch): {r[\"predicted_xwoba_on_contact\"]}')
print(f'predict_xwoba():                               {xwoba}')
print(f'Both not None:', r['predicted_xwoba_on_contact'] is not None and xwoba is not None)
print(f'Per-pitch range check (expect 0.02–0.20): {0.02 <= xwoba <= 0.20}')
"
```

Expected: both values not None, in range 0.02–0.20 for a center-cut FF.

- [ ] **Step 5: Commit artifacts note**

The model files are gitignored (binary). Commit the updated parquet:
```bash
git add data/processed/statcast_processed.parquet
git commit -m "data: add xwoba_target column to processed parquet (per-pitch approach)"
```

---

## Task 5: Run Synthetic Demo and Validate

**Files:**
- No code changes — runs existing scripts

- [ ] **Step 1: Run the full synthetic demo**

```bash
source venv/bin/activate && python -m scripts.run_synthetic_demo 2>/dev/null
```

Expected: Both MLB and AAA reports print with populated `avg xwOBA` column (no `N/A`). Rank comparison tables print for both leagues.

**Capture the output and check:**

1. **xwOBA values populated**: `avg xwOBA` column shows numeric values (not N/A) for all hitters
2. **Per-pitch range check**: avg xwOBA across 50 pitches should be in 0.040–0.150 range per hitter
3. **P(hard) ranks preserved**: the P(hard) ordering should be unchanged from before (Ohtani/Merrill/Judge at top for MLB, Domínguez/Baldwin at top for AAA)

- [ ] **Step 2: Ramírez rank check**

In the MLB rank comparison table, find José Ramírez. 

**Expected:** Ramírez should rank HIGHER on xwOBA than on P(hard).

- P(hard) rank: 7th of 7 (0.206 — below thin-sample floor)
- xwOBA rank: ideally 4th–6th (Ramírez is a contact quality hitter; xwOBA captures his zone control and on-base production better than P(hard))

If Ramírez is STILL 7th on xwOBA: record his actual xwOBA value and the delta from his P(hard) rank. If delta = 0 (no movement), this is worth flagging — the per-pitch model may have the same blind spot as P(hard). Report this.

- [ ] **Step 3: Domínguez rank check (AAA)**

In the AAA rank comparison table, Domínguez should be at or near the top on xwOBA.

- P(hard) rank: 1st (0.310)
- xwOBA rank: expected 1st or 2nd

If Domínguez ranks 4th or lower on xwOBA — this is a concerning divergence. Report it.

- [ ] **Step 4: Collect and record validation summary table**

Record the full tables from terminal output:

MLB validation table (fill in from terminal output):
```
Hitter                      P(hard)    P(hard)#   xwOBA      xwOBA#   Δ
──────────────────────────────────────────────────────────────────────────
Shohei Ohtani               0.310      #1         0.XXX      #X       +/-X
Aaron Judge                 0.289      #2         0.XXX      #X       +/-X
Jackson Merrill             0.295      #3         0.XXX      #X       +/-X  (check — Merrill was #2 by P(hard))
Francisco Lindor            0.259      #4         0.XXX      #X       +/-X
Kevin McGonigle             0.226      #5         0.XXX      #X       +/-X
Munetaka Murakami           0.225      #6         0.XXX      #X       +/-X
José Ramírez                0.206      #7         0.XXX      #X       +/-X  ← KEY CHECK
```

AAA validation table (fill in from terminal output):
```
Hitter                      P(hard)    P(hard)#   xwOBA      xwOBA#   Δ
────────────────────────────────────────────────────────────────────────
Domínguez, Jasson           0.310      #1         0.XXX      #X       +/-X  ← KEY CHECK
Baldwin, Drake              0.303      #2         0.XXX      #X       +/-X
Dingler, Dillon             0.288      #3         0.XXX      #X       +/-X
Narváez, Carlos             0.270      #4         0.XXX      #X       +/-X
Kurtz, Nick                 0.257      #5         0.XXX      #X       +/-X
```

- [ ] **Step 5: Report back to user with filled-in tables**

The user's sign-off criteria:
1. All 12 hitters have populated xwOBA values
2. Ramírez ranks higher on xwOBA than P(hard), OR a clear explanation of why he doesn't
3. Domínguez is #1 or #2 on xwOBA (AAA)
4. Per-pitch xwOBA averages are in the 0.040–0.150 range

Only declare done after reviewing the actual terminal output and filling in the tables above.

---

## Self-Review Checklist

**Spec coverage:**
- [x] Part 1.1: per-pitch xwOBA target (batted ball + walk/HBP + 0.0) — Task 1
- [x] Part 1.2: validate target distribution (print, STOP on anomalies) — Task 1
- [x] Part 2.1: XGBRegressor, same features, same split, same hyperparameters — Task 1
- [x] Part 2.2: `src/model/train_xwoba.py` standalone file — Task 1
- [x] Part 2.3: RMSE/MAE/R² full + batted-balls, by outcome, WARNING on RMSE > 0.6 — Task 1
- [x] Part 3.1: `predict_xwoba()` function — Task 2; existing `predicted_xwoba_on_contact` field already satisfies "additive xwoba field"
- [x] Part 3.2: `avg_xwoba` populated in report — already wired in report.py (activates when model present)
- [x] Part 3.3: backward compat — existing callers unchanged — verified in Task 2 Step 3
- [x] Part 4.1: re-run synthetic demo with populated xwOBA — Task 5
- [x] Part 4.2: Ramírez rank comparison table — Task 3 + 5
- [x] Part 4.3: Domínguez AAA rank check — Task 5
- [x] Part 4.4: sanity check per-pitch xwOBA range — Task 4 Step 4

**Constraints:**
- [x] Not modifying three-stage classifier pipeline
- [x] Not modifying hitter profile building
- [x] Not adding new features
- [x] Not deleting `models/pitch_duel_xgb.json`
- [x] Fixed seed=42 in XWOBA_REG_PARAMS
- [x] Saving to `models/xwoba_model.json` (what predict.py expects, not `pitch_duel_xwoba_xgb.json`)

**Type consistency check:**
- `_compute_xwoba_target()` returns `pd.Series` → used as `df["xwoba_target"]` ✅
- `prepare_splits()` returns `(fit_df, val_df, test_df, col_medians: pd.Series)` — `train_model()` accepts same ✅
- `train_model()` returns `(xgb.XGBRegressor, IsotonicRegression)` — `evaluate_model()` accepts same ✅
- `predict_xwoba()` returns `float | None` — callers handle None ✅

**Naming consistency:**
- Model file: `models/xwoba_model.json` — matches `predict.py` line: `xwoba_path = model_dir / "xwoba_model.json"` ✅
- Calibrator file: `models/xwoba_calibrator.pkl` — matches `predict.py` line: `xwoba_cal_path = model_dir / "xwoba_calibrator.pkl"` ✅
- Target column: `xwoba_target` — train_xwoba.py internal + parquet column ✅
- Return field in predict_pitch: `predicted_xwoba_on_contact` — unchanged from existing ✅

**Placeholder scan:** None found. All code steps are complete and executable.
