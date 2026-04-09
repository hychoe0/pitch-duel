"""
retrain_v2.py — Retrain with 2015-2025 data for forward-looking 2026 predictions.

Creates a parallel model in models_v2/ without touching the original models/.

Changes from the v1 pipeline:
  - Train/test split: train < 2026 (all data), no held-out test set
  - Val (early stopping): 2025 data (was 2024)
  - Fit: 2015-2024 data (was 2015-2023)
  - Profile cutoff: 2026-01-01 (includes 2025 season in hitter profiles)
  - Profiles saved to data/processed/profiles_v2/

Usage:
    python -m src.model.retrain_v2
    python -m src.model.retrain_v2 --skip-preprocess   # reuse existing processed parquet
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.data.preprocess import (
    ALL_FEATURES, PITCH_TYPE_MAP, PREV_RESULT_MAP,
    SWING_DESCRIPTIONS, CONTACT_DESCRIPTIONS, IN_PLAY_DESCRIPTIONS,
    clean, make_prev_pitch_features, make_times_through_order,
    impute_pitcher_medians, make_target, make_score_diff,
    make_three_stage_targets, encode_pitch_types, encode_prev_result,
    encode_handedness, make_runner_flags, make_era_flag,
    make_spin_to_velo_ratio, make_sample_weights,
    add_contextual_hitter_features,
)

PROCESSED_DIR = Path("data/processed")
MODEL_V2_DIR = Path("models_v2")
PROFILE_V2_DIR = PROCESSED_DIR / "profiles_v2"

# Same hyperparams as v1
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

# Cutoffs for v2
TRAIN_CUTOFF = "2026-01-01"   # train on everything before 2026
VAL_CUTOFF = "2025-01-01"     # early stopping on 2025
PROFILE_CUTOFF = "2026-01-01" # profiles include 2025 season


# ---------------------------------------------------------------------------
# Preprocessing (reuses all functions from preprocess.py)
# ---------------------------------------------------------------------------

def preprocess_v2() -> pd.DataFrame:
    """Run full preprocessing with v2 profile cutoff. Returns processed DataFrame."""
    from src.data.fetch import load_raw

    print("=" * 60)
    print("RETRAIN V2 — Preprocessing")
    print("=" * 60)

    raw = load_raw()
    df = clean(raw)
    df = make_prev_pitch_features(df)
    df = make_times_through_order(df)
    df = impute_pitcher_medians(df)

    df["hit"] = make_target(df)
    df["xwoba"] = df["estimated_woba_using_speedangle"].fillna(0.0)
    df["score_diff"] = make_score_diff(df)
    df = make_three_stage_targets(df)
    df = encode_pitch_types(df)
    df = encode_prev_result(df)
    df = encode_handedness(df)
    df = make_runner_flags(df)
    df["era_flag_enc"] = make_era_flag(df)
    df["spin_to_velo_ratio"] = make_spin_to_velo_ratio(df)
    df["sample_weight"] = make_sample_weights(df)

    # Build profiles with v2 cutoff (includes 2025 data)
    print(f"\nBuilding hitter profiles with cutoff={PROFILE_CUTOFF}...")
    PROFILE_V2_DIR.mkdir(parents=True, exist_ok=True)
    from src.hitters.profiles import merge_profiles_into_df
    df = merge_profiles_into_df(
        df,
        profile_dir=PROFILE_V2_DIR,
        date_cutoff=PROFILE_CUTOFF,
    )

    print("Resolving contextual hitter features...")
    df = add_contextual_hitter_features(df)

    # Save v2 processed data
    out_path = PROCESSED_DIR / "statcast_processed_v2.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} ({len(df):,} rows)")

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _fill_nan(df: pd.DataFrame, col_medians: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(col_medians)
    return df


def train_one_stage(
    stage_name: str,
    target_col: str,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    col_medians: pd.Series,
) -> dict:
    """Train one stage, calibrate on val, save to models_v2/."""
    print(f"\n{'=' * 60}")
    print(f"Stage {stage_name.upper()} — P({target_col.replace('target_', '')})")
    print(f"{'=' * 60}")

    fit_df = _fill_nan(fit_df, col_medians)
    val_df = _fill_nan(val_df, col_medians)

    X_fit = fit_df[ALL_FEATURES].values
    y_fit = fit_df[target_col].values
    w_fit = fit_df["sample_weight"].values if "sample_weight" in fit_df.columns else np.ones(len(fit_df))
    X_val = val_df[ALL_FEATURES].values
    y_val = val_df[target_col].values

    for name, arr in [("X_fit", X_fit), ("X_val", X_val)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            raise ValueError(f"{n_nan} NaN values remain in {name}.")

    pos = y_fit.sum()
    neg = len(y_fit) - pos
    rate = pos / len(y_fit)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"  Fit rows:  {len(fit_df):,}")
    print(f"  Val rows:  {len(val_df):,}")
    print(f"  Pos rate:  {rate:.3f}  (scale_pos_weight={scale_pos_weight:.2f})")

    params = DEFAULT_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # Val AUC (no test set in v2)
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    print(f"  Val AUC: {auc:.4f}")

    # Calibration on val set
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_proba, y_val)
    cal_val = calibrator.predict(val_proba)
    print(f"  Calibration — val mean: raw={val_proba.mean():.4f}  "
          f"cal={cal_val.mean():.4f}  actual={y_val.mean():.4f}")

    # Top 5 features
    importances = model.feature_importances_
    top5 = sorted(zip(ALL_FEATURES, importances), key=lambda x: -x[1])[:5]
    print(f"  Top 5: {', '.join(f'{n}({v:.4f})' for n, v in top5)}")

    # Save
    MODEL_V2_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_V2_DIR / f"{stage_name}_model.json"))
    joblib.dump(calibrator, MODEL_V2_DIR / f"{stage_name}_calibrator.pkl")

    feat_imp = {n: float(v) for n, v in zip(ALL_FEATURES, importances)}
    with open(MODEL_V2_DIR / f"{stage_name}_feature_importances.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    print(f"  Saved: {MODEL_V2_DIR}/{stage_name}_model.json")
    return {"model": model, "calibrator": calibrator, "auc": auc, "rate": rate}


def run_training_v2(df: pd.DataFrame) -> dict:
    """Train all three stages with v2 cutoffs."""
    print("\n" + "=" * 60)
    print("RETRAIN V2 — Training")
    print("=" * 60)

    train_cutoff = pd.Timestamp(TRAIN_CUTOFF)
    val_cutoff = pd.Timestamp(VAL_CUTOFF)

    train_df = df[df["game_date"] < train_cutoff].copy()
    print(f"Total training pool: {len(train_df):,} rows")

    fit_df = train_df[train_df["game_date"] < val_cutoff].copy()
    val_df = train_df[train_df["game_date"] >= val_cutoff].copy()
    print(f"Fit (pre-2025): {len(fit_df):,}  |  Val (2025): {len(val_df):,}")

    # Shared column medians from fit set
    col_medians = fit_df[ALL_FEATURES].median()

    # Stage 1: P(swing) — all pitches
    s1 = train_one_stage("swing", "target_swing", fit_df, val_df, col_medians)

    # Stage 2: P(contact | swing) — swing subset
    fit_swing = fit_df[fit_df["target_swing"] == 1]
    val_swing = val_df[val_df["target_swing"] == 1]
    print(f"\n  [contact] Swing subsets — fit: {len(fit_swing):,}  val: {len(val_swing):,}")
    s2 = train_one_stage("contact", "target_contact", fit_swing, val_swing, col_medians)

    # Stage 3: P(hard_contact | contact) — in-play subset
    fit_contact = fit_df[fit_df["description"].isin(IN_PLAY_DESCRIPTIONS)]
    val_contact = val_df[val_df["description"].isin(IN_PLAY_DESCRIPTIONS)]
    print(f"\n  [hard_contact] In-play subsets — fit: {len(fit_contact):,}  val: {len(val_contact):,}")
    s3 = train_one_stage("hard_contact", "target_hard_contact", fit_contact, val_contact, col_medians)

    # Shared artifacts
    with open(MODEL_V2_DIR / "feature_cols.json", "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)
    with open(MODEL_V2_DIR / "encodings.json", "w") as f:
        json.dump({"PITCH_TYPE_MAP": PITCH_TYPE_MAP, "PREV_RESULT_MAP": PREV_RESULT_MAP}, f, indent=2)

    metrics = {
        "swing_auc":         round(s1["auc"], 4),
        "swing_rate":        round(s1["rate"], 4),
        "contact_auc":       round(s2["auc"], 4),
        "contact_rate":      round(s2["rate"], 4),
        "hard_contact_auc":  round(s3["auc"], 4),
        "hard_contact_rate": round(s3["rate"], 4),
        "train_cutoff":      TRAIN_CUTOFF,
        "val_cutoff":        VAL_CUTOFF,
        "profile_cutoff":    PROFILE_CUTOFF,
    }
    with open(MODEL_V2_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("V2 TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Train: 2015-2025 | Fit: pre-2025 | Val: 2025")
    print(f"  Profiles: include 2025 season (cutoff {PROFILE_CUTOFF})")
    print(f"  Stage 1 — P(swing)             Val AUC: {s1['auc']:.4f}")
    print(f"  Stage 2 — P(contact|swing)     Val AUC: {s2['auc']:.4f}")
    print(f"  Stage 3 — P(hard_contact|con.) Val AUC: {s3['auc']:.4f}")
    print(f"\n  Artifacts saved to {MODEL_V2_DIR}/")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain v2: full 2015-2025 pipeline")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Reuse existing statcast_processed_v2.parquet")
    args = parser.parse_args()

    if args.skip_preprocess:
        v2_path = PROCESSED_DIR / "statcast_processed_v2.parquet"
        if not v2_path.exists():
            print(f"{v2_path} not found — running preprocessing anyway.")
            df = preprocess_v2()
        else:
            print(f"Loading existing {v2_path}...")
            df = pd.read_parquet(v2_path)
            print(f"Loaded {len(df):,} rows")
    else:
        df = preprocess_v2()

    run_training_v2(df)
