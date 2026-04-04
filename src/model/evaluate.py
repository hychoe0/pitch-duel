"""
evaluate.py — Per-hitter accuracy evaluation on 2025 Statcast data.

Fetches 2025 pitches for 4 target players, runs model predictions,
and reports calibration quality broken down by pitch type and count.

Usage:
    python -m src.model.evaluate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.data.preprocess import (
    HIT_EVENTS,
    encode_pitch_types,
    encode_prev_result,
    encode_handedness,
    impute_pitcher_medians,
    make_era_flag,
    make_prev_pitch_features,
    make_runner_flags,
    make_score_diff,
    make_spin_to_velo_ratio,
)
from src.model.predict import load_model

PROFILE_DIR  = Path("data/processed/profiles")
CACHE_DIR    = Path("data/raw/eval_cache")
MODEL_DIR    = Path("models")

SEASON_START = "2025-03-01"
SEASON_END   = "2025-11-01"

TARGET_PLAYERS = [
    {"name": "Freddie Freeman",  "player_id": 518692},
    {"name": "Yordan Alvarez",   "player_id": 670541},
    {"name": "Bobby Witt Jr.",   "player_id": 677951},
    {"name": "Jackson Merrill",  "player_id": 701538},
]


# ---------------------------------------------------------------------------
# Step 1 — Fetch 2025 data
# ---------------------------------------------------------------------------

def fetch_2025(player_id: int, player_name: str) -> pd.DataFrame:
    """Pull 2025 Statcast batter data, caching locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{player_id}_2025.csv"

    if cache_path.exists():
        print(f"  {player_name}: loading from cache")
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  {player_name}: fetching from Baseball Savant...")
    df = pybaseball.statcast_batter(SEASON_START, SEASON_END, player_id=player_id)
    if df is not None and len(df) > 0:
        df.to_csv(cache_path, index=False)
        print(f"    → {len(df):,} pitches cached")
    else:
        print(f"    → no data returned")
        df = pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# Step 2 — Preprocess to match model input
# ---------------------------------------------------------------------------

def preprocess_2025(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    if len(df) == 0:
        return df

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Drop rows we can't use
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "UN")]
    df = df[df["release_speed"].notna()]
    df = df[df["plate_x"].notna() & df["plate_z"].notna()]
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    df = make_prev_pitch_features(df)
    df = impute_pitcher_medians(df)
    df = encode_pitch_types(df)
    df = encode_prev_result(df)
    df = encode_handedness(df)
    df = make_runner_flags(df)

    df["era_flag_enc"]       = make_era_flag(df)
    df["spin_to_velo_ratio"] = make_spin_to_velo_ratio(df)
    df["score_diff"]         = make_score_diff(df)

    # Target
    df["hit"]   = df["events"].isin(HIT_EVENTS).astype(int)
    df["xwoba"] = df["estimated_woba_using_speedangle"].fillna(0.0)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Merge profile features and run predictions
# ---------------------------------------------------------------------------

def load_hitter_profile_features(player_id: int) -> dict:
    """Load profile and return the full feature dict plus metadata."""
    from src.hitters.profiles import load_profile, profile_to_feature_dict
    profile = load_profile(player_id, PROFILE_DIR)
    d = profile_to_feature_dict(profile)
    d["sample_size"]    = profile.sample_size
    d["is_thin_sample"] = profile.is_thin_sample
    return d


def predict_batch(df: pd.DataFrame, profile_features: dict,
                  feature_cols: list, model, calibrator=None) -> tuple:
    """
    Assemble feature matrix and return (raw_probs, calibrated_probs).
    If no calibrator, calibrated_probs == raw_probs.
    """
    from src.data.preprocess import PITCHER_FEATURES
    for col, val in profile_features.items():
        if col in feature_cols:
            df[col] = val

    # Merge pitcher identity features per row
    if any(f in feature_cols for f in PITCHER_FEATURES) and "pitcher" in df.columns:
        from src.pitchers.features import load_pitcher_features
        pitcher_df = load_pitcher_features()
        df = df.drop(columns=[c for c in PITCHER_FEATURES if c in df.columns])
        df = df.merge(pitcher_df, on="pitcher", how="left")
        for col in PITCHER_FEATURES:
            if col in feature_cols:
                df[col] = df[col].fillna(pitcher_df[col].median())

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].values.astype(float)

    # Impute any residual NaNs with column median
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_medians, inds[1])

    raw_preds = np.clip(model.predict(X), 0.0, None)
    cal_preds = np.clip(calibrator.predict(raw_preds), 0.0, None) if calibrator is not None else raw_preds
    return raw_preds, cal_preds


# ---------------------------------------------------------------------------
# Step 4 — Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_xwoba: np.ndarray, y_hit: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_xwoba) == 0:
        return {"mae": float("nan"), "pearson_r": float("nan"), "auc": float("nan"),
                "mean_xwoba": float("nan"), "mean_pred": float("nan")}
    mae = float(mean_absolute_error(y_xwoba, y_pred))
    r = float(pearsonr(y_xwoba, y_pred)[0]) if y_xwoba.std() > 0 else float("nan")
    auc = float(roc_auc_score(y_hit, y_pred)) if y_hit.sum() > 0 and y_hit.sum() < len(y_hit) else float("nan")
    return {
        "mae":        mae,
        "pearson_r":  r,
        "auc":        auc,
        "mean_xwoba": float(y_xwoba.mean()),
        "mean_pred":  float(y_pred.mean()),
    }


def print_breakdown(df: pd.DataFrame, preds: np.ndarray, group_col: str,
                    label: str, min_pitches: int = 50) -> None:
    df = df.copy()
    df["_pred"]  = preds
    df["_xwoba"] = df["xwoba"].values

    rows = []
    for name, grp in df.groupby(group_col):
        if len(grp) < min_pitches:
            continue
        y = grp["_xwoba"].values
        p = grp["_pred"].values
        rows.append({
            "group":     str(name),
            "n":         len(grp),
            "actual":    y.mean(),
            "predicted": p.mean(),
            "mae":       abs(y.mean() - p.mean()),
        })

    if not rows:
        return

    rows.sort(key=lambda r: -r["n"])
    print(f"\n  Breakdown by {label}:")
    print(f"  {'':8} | {'Actual xwOBA':>12} | {'Mean Pred':>10} | {'MAE':>8} | {'n':>6}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")
    for r in rows:
        print(f"  {r['group']:<8} | {r['actual']:>12.4f} | {r['predicted']:>10.4f} | {r['mae']:>8.4f} | {r['n']:>6,}")


def print_player_report(player_name: str, profile_sample: int,
                        df: pd.DataFrame,
                        raw_preds: np.ndarray, cal_preds: np.ndarray) -> dict:
    m     = compute_metrics(df["xwoba"].values, df["hit"].values, cal_preds)
    m_raw = compute_metrics(df["xwoba"].values, df["hit"].values, raw_preds)

    print(f"\n{'='*65}")
    print(f"  Player: {player_name}  |  Profile pitches: {profile_sample:,}  |  2025 pitches: {len(df):,}")
    print(f"  {'-'*61}")
    print(f"  Mean actual xwOBA:                   {m['mean_xwoba']:.4f}")
    print(f"  Mean predicted xwOBA (raw):          {m_raw['mean_pred']:.4f}")
    print(f"  Mean predicted xwOBA (calibrated):   {m['mean_pred']:.4f}")
    print(f"  MAE (calibrated):                    {m['mae']:.4f}")
    print(f"  Pearson r (calibrated):              {m['pearson_r']:.3f}")
    print(f"  AUC (binarized, calibrated):         {m['auc']:.3f}")

    print_breakdown(df, cal_preds, "pitch_type", "pitch type", min_pitches=50)
    print_breakdown(df, cal_preds, "count_str",  "count",      min_pitches=30)

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation():
    print("\nLoading model...")
    model, feature_cols, _, calibrator = load_model(MODEL_DIR)

    summary_rows = []

    for player in TARGET_PLAYERS:
        pid   = player["player_id"]
        name  = player["name"]

        print(f"\n[{name}]")

        # Fetch + preprocess
        raw = fetch_2025(pid, name)
        if len(raw) == 0:
            print(f"  No 2025 data — skipping.")
            continue
        df = preprocess_2025(raw)
        if len(df) == 0:
            print(f"  No usable pitches after preprocessing — skipping.")
            continue

        # Add count string column for breakdown
        df["count_str"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

        # Load profile
        pf = load_hitter_profile_features(pid)

        # Predict
        raw_preds, cal_preds = predict_batch(df.copy(), pf, feature_cols, model, calibrator)

        # Report
        m = print_player_report(name, pf["sample_size"], df, raw_preds, cal_preds)

        summary_rows.append({
            "Player":          name,
            "Profile pitches": pf["sample_size"],
            "2025 pitches":    len(df),
            "Mean xwOBA":      m["mean_xwoba"],
            "Mean pred":       m["mean_pred"],
            "MAE":             m["mae"],
            "Pearson r":       m["pearson_r"],
            "AUC":             m["auc"],
        })

    # Final summary table
    if not summary_rows:
        print("\nNo results to summarize.")
        return

    print(f"\n\n{'='*95}")
    print("  SUMMARY")
    print(f"  {'-'*93}")
    hdr = (f"  {'Player':<18} | {'Profile pitches':>15} | {'2025 pitches':>12} | "
           f"{'Mean xwOBA':>10} | {'Mean pred':>9} | {'MAE':>7} | {'Pearson r':>9} | {'AUC':>6}")
    print(hdr)
    print(f"  {'-'*93}")
    for r in summary_rows:
        auc_s = f"{r['AUC']:.3f}"      if not np.isnan(r['AUC'])      else "  n/a"
        r_s   = f"{r['Pearson r']:.3f}" if not np.isnan(r['Pearson r']) else "    n/a"
        print(f"  {r['Player']:<18} | {r['Profile pitches']:>15,} | {r['2025 pitches']:>12,} | "
              f"{r['Mean xwOBA']:>10.4f} | {r['Mean pred']:>9.4f} | {r['MAE']:>7.4f} | "
              f"{r_s:>9} | {auc_s:>6}")
    print(f"  {'='*95}\n")


if __name__ == "__main__":
    run_evaluation()
