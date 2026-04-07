"""
train.py — Three-stage hitter behavior XGBoost classifiers.

Stage 1: P(swing)              — all pitches
Stage 2: P(contact | swing)    — swings only
Stage 3: P(hard_contact | contact) — in-play contact only

Each stage is an XGBClassifier with isotonic calibration on the 2024 val set.

Usage:
    python -m src.model.train
"""

import json
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
)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data() -> tuple:
    """Load the profile-merged processed parquet and split temporally."""
    path = PROCESSED_DIR / "statcast_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py then profiles.py first."
        )
    df = pd.read_parquet(path)

    cutoff = pd.Timestamp("2025-01-01")
    train = df[df["game_date"] < cutoff].copy()
    test = df[df["game_date"] >= cutoff].copy()
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test


def _check_features(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {label}: {missing}")


def _fill_nan(df: pd.DataFrame, col_medians: pd.Series = None) -> tuple:
    """Fill residual NaNs with column medians. Returns (clean_df, col_medians)."""
    nan_cols = df[ALL_FEATURES].columns[df[ALL_FEATURES].isna().any()].tolist()
    if nan_cols:
        print(f"  NaN columns (filling median): {nan_cols}")
    if col_medians is None:
        col_medians = df[ALL_FEATURES].median()
    df = df.copy()
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(col_medians)
    return df, col_medians


# ---------------------------------------------------------------------------
# Single-stage training
# ---------------------------------------------------------------------------

def train_one_stage(
    stage_name: str,
    target_col: str,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col_medians: pd.Series,
) -> dict:
    """
    Train one XGBClassifier stage, calibrate it, save artifacts, and print summary.

    Returns dict with model, calibrator, and auc.
    """
    print(f"\n{'─' * 60}")
    print(f"Stage {stage_name.upper()} — P({target_col.replace('target_', '')})")
    print(f"{'─' * 60}")

    # Validate target exists
    for df, label in [(fit_df, "fit"), (val_df, "val"), (test_df, "test")]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in {label} data. "
                             "Re-run preprocess.py to add three-stage targets.")

    # Fill NaNs using pre-computed medians
    fit_df, _ = _fill_nan(fit_df, col_medians)
    val_df, _ = _fill_nan(val_df, col_medians)
    test_df, _ = _fill_nan(test_df, col_medians)

    X_fit  = fit_df[ALL_FEATURES].values
    y_fit  = fit_df[target_col].values
    w_fit  = fit_df["sample_weight"].values if "sample_weight" in fit_df.columns else np.ones(len(fit_df))
    X_val  = val_df[ALL_FEATURES].values
    y_val  = val_df[target_col].values
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df[target_col].values

    # Verify no residual NaNs
    for name, arr in [("X_fit", X_fit), ("X_val", X_val), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            raise ValueError(f"{n_nan} NaN values remain in {name}.")

    pos = y_fit.sum()
    neg = len(y_fit) - pos
    rate = pos / len(y_fit)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"  Training rows: {len(fit_df):,}")
    print(f"  Positive rate: {rate:.3f}  (scale_pos_weight={scale_pos_weight:.2f})")

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

    # Test AUC
    test_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    print(f"  Test AUC: {auc:.4f}")

    # Calibration on val set
    val_proba = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_proba, y_val)
    cal_val = calibrator.predict(val_proba)
    print(f"  Calibration — val mean: raw={val_proba.mean():.4f}  "
          f"cal={cal_val.mean():.4f}  actual={y_val.mean():.4f}")

    # Top 5 features by importance
    importances = model.feature_importances_
    top5 = sorted(zip(ALL_FEATURES, importances), key=lambda x: -x[1])[:5]
    print(f"  Top 5 features: {', '.join(f'{n}({v:.4f})' for n, v in top5)}")

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / f"{stage_name}_model.json"))
    joblib.dump(calibrator, MODEL_DIR / f"{stage_name}_calibrator.pkl")

    # Save per-stage feature importances
    feat_imp = {n: float(v) for n, v in zip(ALL_FEATURES, importances)}
    with open(MODEL_DIR / f"{stage_name}_feature_importances.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    print(f"  Saved: {stage_name}_model.json, {stage_name}_calibrator.pkl")

    return {"model": model, "calibrator": calibrator, "auc": auc, "rate": rate}


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training() -> dict:
    """Train all three stages and return artifacts dict."""
    path = PROCESSED_DIR / "statcast_processed.parquet"
    if not path.exists():
        print("Statcast data not preprocessed. Running preprocess.py...")
        from src.data.preprocess import run_preprocessing
        from src.data.fetch import load_raw
        raw = load_raw()
        run_preprocessing(raw)
        print()

    train_df, test_df = load_training_data()

    # 2024 as early-stopping val; pre-2024 as fit
    val_cutoff = pd.Timestamp("2024-01-01")
    val_df = train_df[train_df["game_date"] >= val_cutoff].copy()
    fit_df = train_df[train_df["game_date"] < val_cutoff].copy()
    print(f"Fit: {len(fit_df):,}  |  Val (2024): {len(val_df):,}")

    # Validate features
    for df, label in [(fit_df, "fit"), (val_df, "val"), (test_df, "test")]:
        _check_features(df, label)

    # Compute shared column medians from fit set (used for NaN imputation across stages)
    nan_cols_global = fit_df[ALL_FEATURES].columns[fit_df[ALL_FEATURES].isna().any()].tolist()
    col_medians = fit_df[ALL_FEATURES].median()

    # ── Stage 1: P(swing) ────────────────────────────────────────────────────
    s1 = train_one_stage("swing", "target_swing", fit_df, val_df, test_df, col_medians)

    # ── Stage 2: P(contact | swing) ─────────────────────────────────────────
    # Train only on pitches where the hitter swung
    fit_swing  = fit_df[fit_df["target_swing"] == 1]
    val_swing  = val_df[val_df["target_swing"] == 1]
    test_swing = test_df[test_df["target_swing"] == 1]
    print(f"\n  [contact] Swing subsets — fit: {len(fit_swing):,}  val: {len(val_swing):,}  test: {len(test_swing):,}")
    s2 = train_one_stage("contact", "target_contact", fit_swing, val_swing, test_swing, col_medians)

    # ── Stage 3: P(hard_contact | contact) ──────────────────────────────────
    # Train only on pitches that were put in play (contact + in-play description)
    # Fouls don't have launch_speed so we restrict to in-play descriptions
    in_play_mask = fit_df["description"].isin(IN_PLAY_DESCRIPTIONS)
    fit_contact  = fit_df[in_play_mask]
    in_play_mask_val = val_df["description"].isin(IN_PLAY_DESCRIPTIONS)
    val_contact  = val_df[in_play_mask_val]
    in_play_mask_test = test_df["description"].isin(IN_PLAY_DESCRIPTIONS)
    test_contact = test_df[in_play_mask_test]
    print(f"\n  [hard_contact] In-play subsets — fit: {len(fit_contact):,}  val: {len(val_contact):,}  test: {len(test_contact):,}")
    s3 = train_one_stage("hard_contact", "target_hard_contact", fit_contact, val_contact, test_contact, col_medians)

    # ── Shared artifacts ─────────────────────────────────────────────────────
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)
    with open(MODEL_DIR / "encodings.json", "w") as f:
        json.dump({"PITCH_TYPE_MAP": PITCH_TYPE_MAP, "PREV_RESULT_MAP": PREV_RESULT_MAP}, f, indent=2)

    metrics = {
        "swing_auc":        round(s1["auc"], 4),
        "swing_rate":       round(s1["rate"], 4),
        "contact_auc":      round(s2["auc"], 4),
        "contact_rate":     round(s2["rate"], 4),
        "hard_contact_auc": round(s3["auc"], 4),
        "hard_contact_rate": round(s3["rate"], 4),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("TRAINING SUMMARY")
    print("═" * 60)
    print(f"  Stage 1 — P(swing)             AUC: {s1['auc']:.4f}  rate: {s1['rate']:.3f}")
    print(f"  Stage 2 — P(contact|swing)     AUC: {s2['auc']:.4f}  rate: {s2['rate']:.3f}")
    print(f"  Stage 3 — P(hard_contact|con.) AUC: {s3['auc']:.4f}  rate: {s3['rate']:.3f}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")

    return {
        "swing": s1["model"], "swing_cal": s1["calibrator"],
        "contact": s2["model"], "contact_cal": s2["calibrator"],
        "hard_contact": s3["model"], "hard_contact_cal": s3["calibrator"],
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_training()
