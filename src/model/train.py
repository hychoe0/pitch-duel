"""
train.py — Train the Phase 1 XGBoost hit-probability classifier.

Temporal split: train on 2015-2022, evaluate on 2023-2026.
Uses 2022 season as early-stopping validation set within training.

Usage:
    python -m src.model.train
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

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

    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["game_date"] < cutoff].copy()
    test = df[df["game_date"] >= cutoff].copy()
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test


def prepare_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Extract feature matrices, targets, and sample weights. Fails loudly on any NaN."""
    missing = [c for c in ALL_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in training data: {missing}")

    X_train = train_df[ALL_FEATURES].values
    y_train = train_df["hit"].values
    w_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else np.ones(len(train_df))
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df["hit"].values

    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            raise ValueError(f"{n_nan} NaN values found in {name}. Fix preprocessing first.")

    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"Hit rate — train: {y_train.mean():.4f}  |  test: {y_test.mean():.4f}")
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
) -> xgb.XGBClassifier:
    """
    Train XGBClassifier.
    scale_pos_weight computed from class balance in y_train.
    Era-based sample_weight passed to fit() for per-row weighting.
    Early stopping monitors the provided validation set.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    params["scale_pos_weight"] = n_neg / n_pos
    print(f"scale_pos_weight: {params['scale_pos_weight']:.2f}  ({n_neg:,} non-hit / {n_pos:,} hit)")

    model = xgb.XGBClassifier(**params)
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
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list = None,
) -> dict:
    """Compute AUC, log loss, Brier score, and feature importances."""
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    metrics = {"roc_auc": auc, "log_loss": ll, "brier_score": brier}

    print("\n--- Test Set Metrics ---")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Log Loss   : {ll:.4f}")
    print(f"  Brier Score: {brier:.4f}")

    if feature_cols:
        importances = model.feature_importances_
        top = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:20]
        print("\n--- Top 20 Feature Importances ---")
        for name, imp in top:
            print(f"  {name:<35} {imp:.4f}")
        metrics["feature_importances"] = {n: float(i) for n, i in zip(feature_cols, importances)}

    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: xgb.XGBClassifier,
    feature_cols: list,
    metrics: dict,
    model_dir: Path = MODEL_DIR,
) -> Path:
    """
    Save model (XGBoost native JSON), feature list, metrics, and encodings.
    Returns path to model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "pitch_duel_xgb.json"
    model.save_model(str(model_path))

    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Convert numpy floats for JSON serialization
    serializable_metrics = {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in metrics.items()}
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    with open(model_dir / "encodings.json", "w") as f:
        json.dump({"PITCH_TYPE_MAP": PITCH_TYPE_MAP, "PREV_RESULT_MAP": PREV_RESULT_MAP}, f, indent=2)

    print(f"\nModel saved to {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training() -> xgb.XGBClassifier:
    train_df, test_df = load_training_data()

    # Use 2022 season as early-stopping validation set within training
    val_cutoff = pd.Timestamp("2022-01-01")
    val_df = train_df[train_df["game_date"] >= val_cutoff].copy()
    fit_df = train_df[train_df["game_date"] < val_cutoff].copy()
    print(f"Fit: {len(fit_df):,}  |  Val (2022): {len(val_df):,}")

    X_fit, y_fit, w_fit, _, _ = prepare_matrices(fit_df, val_df)
    X_val = val_df[ALL_FEATURES].values
    y_val = val_df["hit"].values
    X_test, y_test = test_df[ALL_FEATURES].values, test_df["hit"].values

    model = train_model(X_fit, y_fit, w_fit, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test, feature_cols=ALL_FEATURES)
    save_model(model, ALL_FEATURES, metrics)
    return model


if __name__ == "__main__":
    run_training()
