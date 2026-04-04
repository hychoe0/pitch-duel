"""
train.py — Train the Phase 1 XGBoost xwOBA regressor.

Temporal split: train on 2015-2024, evaluate on 2025-2026.
Uses 2024 season as early-stopping validation set within training.

Target: estimated_woba_using_speedangle (xwOBA) — measures contact danger
based on exit velocity and launch angle, independent of defense/luck.
Non-contact outcomes (strikeouts, walks, etc.) are assigned xwOBA = 0.0.

Usage:
    python -m src.model.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from src.data.preprocess import ALL_FEATURES, PITCH_TYPE_MAP, PREV_RESULT_MAP

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "eval_metric": "rmse",
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


def prepare_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Extract feature matrices, xwOBA targets, and sample weights."""
    missing = [c for c in ALL_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in training data: {missing}")

    X_train = train_df[ALL_FEATURES].values
    y_train = train_df["xwoba"].values
    w_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else np.ones(len(train_df))
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df["xwoba"].values

    # Fill any residual NaNs with column median
    nan_cols = train_df[ALL_FEATURES].columns[train_df[ALL_FEATURES].isna().any()].tolist()
    if nan_cols:
        print(f"Residual NaN columns (filling with column median): {nan_cols}")
        col_medians = train_df[ALL_FEATURES].median()
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[ALL_FEATURES] = train_df[ALL_FEATURES].fillna(col_medians)
        test_df[ALL_FEATURES] = test_df[ALL_FEATURES].fillna(col_medians)
        X_train = train_df[ALL_FEATURES].values
        X_test = test_df[ALL_FEATURES].values

    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            raise ValueError(f"{n_nan} NaN values remain in {name} after imputation.")

    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"Mean xwOBA — train: {y_train.mean():.4f}  |  test: {y_test.mean():.4f}")
    return X_train, y_train, w_train, X_test, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
) -> xgb.XGBRegressor:
    """
    Train XGBRegressor on xwOBA target.
    Era-based sample_weight passed to fit() for per-row weighting.
    Early stopping monitors RMSE on the provided validation set.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_hit: np.ndarray = None,
    feature_cols: list = None,
) -> dict:
    """
    Compute regression metrics plus binarized AUC for comparison with
    the previous binary classifier.
    """
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, None)  # xwOBA can't be negative

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r, _ = pearsonr(y_test, preds)

    metrics = {"mae": mae, "rmse": rmse, "pearson_r": r}

    print("\n--- Test Set Metrics (xwOBA regression) ---")
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  Pearson r:        {r:.4f}")

    # Binarized AUC: compare against previous binary classifier baseline
    if y_test_hit is not None:
        predicted_hit = (preds > 0.15).astype(int)
        actual_hit    = (y_test_hit > 0.0).astype(int)
        if actual_hit.sum() > 0 and actual_hit.sum() < len(actual_hit):
            auc = roc_auc_score(actual_hit, preds)
            metrics["auc_binarized"] = auc
            print(f"  AUC (binarized):  {auc:.4f}  ← compare to previous 0.7805")

    if feature_cols:
        importances = model.feature_importances_
        top = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:40]
        print("\n--- Top 40 Feature Importances ---")
        for name, imp in top:
            print(f"  {name:<35} {imp:.4f}")
        metrics["feature_importances"] = {n: float(i) for n, i in zip(feature_cols, importances)}

    return metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibrator(
    model: xgb.XGBRegressor,
    X_val: np.ndarray,
    y_val_xwoba: np.ndarray,
    model_dir: Path = MODEL_DIR,
) -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator on the val set (2024 season).
    Corrects systematic bias in the regressor's xwOBA predictions.
    Saves calibrator to models/calibrator.pkl.
    """
    raw_preds = model.predict(X_val)
    raw_preds = np.clip(raw_preds, 0.0, None)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_preds, y_val_xwoba)

    cal_preds = calibrator.predict(raw_preds)
    print("\n--- Calibration (on 2024 val set) ---")
    print(f"  Before — mean predicted xwOBA: {raw_preds.mean():.4f}  |  actual: {y_val_xwoba.mean():.4f}")
    print(f"  After  — mean predicted xwOBA: {cal_preds.mean():.4f}  |  actual: {y_val_xwoba.mean():.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, model_dir / "calibrator.pkl")
    print(f"  Calibrator saved to {model_dir}/calibrator.pkl")

    return calibrator


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: xgb.XGBRegressor,
    feature_cols: list,
    metrics: dict,
    model_dir: Path = MODEL_DIR,
) -> Path:
    """Save model (XGBoost native JSON), feature list, metrics, and encodings."""
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "pitch_duel_xgb.json"
    model.save_model(str(model_path))

    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, dict):
            serializable_metrics[k] = {kk: float(vv) for kk, vv in v.items()}
        else:
            serializable_metrics[k] = v
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    with open(model_dir / "encodings.json", "w") as f:
        json.dump({"PITCH_TYPE_MAP": PITCH_TYPE_MAP, "PREV_RESULT_MAP": PREV_RESULT_MAP}, f, indent=2)

    print(f"\nModel saved to {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training() -> xgb.XGBRegressor:
    train_df, test_df = load_training_data()

    # Use 2024 season as early-stopping validation set within training
    val_cutoff = pd.Timestamp("2024-01-01")
    val_df  = train_df[train_df["game_date"] >= val_cutoff].copy()
    fit_df  = train_df[train_df["game_date"] < val_cutoff].copy()
    print(f"Fit: {len(fit_df):,}  |  Val (2024): {len(val_df):,}")

    X_fit, y_fit, w_fit, _, _ = prepare_matrices(fit_df, val_df)
    X_val   = val_df[ALL_FEATURES].values
    y_val   = val_df["xwoba"].values
    X_test  = test_df[ALL_FEATURES].values
    y_test  = test_df["xwoba"].values
    y_test_hit = test_df["hit"].values

    model = train_model(X_fit, y_fit, w_fit, X_val, y_val)
    fit_calibrator(model, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test, y_test_hit=y_test_hit, feature_cols=ALL_FEATURES)
    save_model(model, ALL_FEATURES, metrics)
    return model


if __name__ == "__main__":
    run_training()
