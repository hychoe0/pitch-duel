"""
train_xwoba.py — Per-pitch xwOBA regressor (parallel to three-stage classifiers).

Target: per-pitch expected xwOBA
  - Batted ball: estimated_woba_using_speedangle
  - Walk / HBP: woba_value
  - All other pitches: 0.0

Most pitches get target=0.0 (non-damage) — the model learns that most pitches
don't generate damage. Do NOT filter to batted balls only.

Temporal split rationale: This regressor uses a later cutoff (test >= 2025) than
the classifier (test >= 2023). Per-pitch xwOBA distributions are most relevant
when calibrated against the current pitching environment, and the additional 2
years of training data (2023-2024) materially improves coverage of pitch-clock-era
pitch profiles. The tradeoff is that head-to-head test-set comparisons against the
classifier are not strictly apples-to-apples — the classifier and regressor evaluate
on overlapping but distinct test sets.

Saves to: models/xwoba_model.json, models/xwoba_calibrator.pkl
Updates:  data/processed/statcast_processed.parquet (adds xwoba_target column)

Usage:
    python -m src.model.train_xwoba
"""

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
    "n_estimators":          500,
    "max_depth":             6,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "min_child_weight":      50,
    "objective":             "reg:squarederror",
    "eval_metric":           "rmse",
    "random_state":          42,
    "n_jobs":                -1,
    "tree_method":           "hist",
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

    pct_zero    = (t == 0.0).mean() * 100
    pct_0_to_05 = ((t > 0.0) & (t <= 0.5)).mean() * 100
    pct_05_to_1 = ((t > 0.5) & (t <= 1.0)).mean() * 100
    pct_above_1 = (t > 1.0).mean() * 100

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
            "Per-pitch xwOBA should average ~0.04-0.08. "
            "Check target construction — possible batted-ball-only filtering."
        )

    # By outcome category
    in_play = df["description"].isin(IN_PLAY_DESCRIPTIONS)
    is_walk = df["events"].isin(WALK_EVENTS)
    is_hbp  = df["events"].isin(HBP_EVENTS)
    is_so   = df["events"].isin({"strikeout", "strikeout_double_play"})

    print("\n  Mean target by outcome category:")
    print(f"    Batted ball:    {df.loc[in_play, 'xwoba_target'].mean():.4f}  (n={in_play.sum():,})")
    print(f"    Walk:           {df.loc[is_walk, 'xwoba_target'].mean():.4f}  (n={is_walk.sum():,})")
    print(f"    HBP:            {df.loc[is_hbp,  'xwoba_target'].mean():.4f}  (n={is_hbp.sum():,})")
    print(f"    Strikeout:      {df.loc[is_so,   'xwoba_target'].mean():.4f}  (n={is_so.sum():,})")
    other = ~in_play & ~is_walk & ~is_hbp & ~is_so
    print(f"    Other:          {df.loc[other,   'xwoba_target'].mean():.4f}  (n={other.sum():,})")

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
    Temporal split:
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
    w_fit = (
        fit_df["sample_weight"].values
        if "sample_weight" in fit_df.columns
        else np.ones(len(fit_df))
    )

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
    print(
        f"Val calibration — raw mean: {val_preds.mean():.4f}  "
        f"cal mean: {cal_preds.mean():.4f}  actual mean: {y_val.mean():.4f}"
    )

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
    print("Full test set:")
    print(f"  RMSE: {rmse_full:.3f}  MAE: {mae_full:.3f}  R2: {r2_full:.3f}  (n={len(y_test):,})")
    print("Batted balls only:")
    print(f"  RMSE: {rmse_bb:.3f}  MAE: {mae_bb:.3f}  R2: {r2_bb:.3f}  (n={in_play.sum():,})")

    if rmse_bb > 0.6:
        print(
            f"\nWARNING: Batted-ball RMSE {rmse_bb:.3f} > 0.6. Expected 0.30-0.45. "
            "Investigate target construction or feature alignment."
        )

    # By outcome category
    is_walk = test_df["events"].isin(WALK_EVENTS)
    is_hbp  = test_df["events"].isin(HBP_EVENTS)
    is_so   = test_df["events"].isin({"strikeout", "strikeout_double_play"})
    other   = ~in_play & ~is_walk & ~is_hbp & ~is_so

    print("\nBy outcome:")
    for label, mask in [
        ("Batted ball", in_play),
        ("Walk",        is_walk),
        ("HBP",         is_hbp),
        ("Strikeout",   is_so),
        ("Other",       other),
    ]:
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
    print(f"Updated parquet saved with 'xwoba_target' column -> {PROCESSED_PATH}")


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
