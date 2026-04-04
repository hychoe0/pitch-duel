"""
evaluate_expanded.py — Expanded 20-player validation across 4 profile-depth tiers.

Tests whether profile sample size drives model accuracy by randomly sampling
5 players per tier and evaluating on their 2025 Statcast data.

Usage:
    python -m src.model.evaluate_expanded
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.data.preprocess import (
    HIT_EVENTS,
    encode_handedness,
    encode_pitch_types,
    encode_prev_result,
    impute_pitcher_medians,
    make_era_flag,
    make_prev_pitch_features,
    make_runner_flags,
    make_score_diff,
    make_spin_to_velo_ratio,
)
from src.hitters.profiles import (
    _build_name_cache,
    _NAME_CACHE,
    build_profile,
    save_profile,
    load_profile,
    profile_to_feature_dict,
    PROFILE_DIR,
)
from src.model.predict import load_model

PROCESSED_PATH = Path("data/processed/statcast_processed.parquet")
CACHE_DIR      = Path("data/processed/eval_2025")
RESULTS_PATH   = Path("data/processed/validation_results.csv")
MODEL_DIR      = Path("models")

SEASON_START = "2025-03-01"
SEASON_END   = "2025-11-01"
RANDOM_SEED  = 42

TIERS = {
    "A": (15000, None),
    "B": (8000,  15000),
    "C": (3000,  8000),
    "D": (0,     3000),
}


# ---------------------------------------------------------------------------
# Step 2 — Select 20 players
# ---------------------------------------------------------------------------

def select_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count pitches per batter in 2015-2024 training window.
    Restrict pool to batters who appear in 2025+ data.
    Tier by pitch count, randomly select 5 per tier (seed=42).
    Returns DataFrame with columns: player_id, tier, profile_pitch_count.
    """
    cutoff = pd.Timestamp("2025-01-01")
    train = df[df["game_date"] < cutoff]
    counts = train.groupby("batter").size().rename("profile_pitches").reset_index()
    counts.columns = ["player_id", "profile_pitches"]

    # Only keep players who appear in the 2025 portion of the dataset
    active_2025 = set(df[df["game_date"] >= cutoff]["batter"].unique())
    counts = counts[counts["player_id"].isin(active_2025)]
    print(f"  Pool restricted to {len(counts):,} players with 2025 data (from {len(active_2025):,} active batters)")

    rng = np.random.default_rng(RANDOM_SEED)
    selected = []

    for tier, (lo, hi) in TIERS.items():
        mask = counts["profile_pitches"] >= lo
        if hi is not None:
            mask &= counts["profile_pitches"] < hi
        pool = counts[mask].copy()
        n = min(5, len(pool))
        chosen = pool.sample(n=n, random_state=RANDOM_SEED)
        chosen = chosen.copy()
        chosen["tier"] = tier
        selected.append(chosen)

    return pd.concat(selected, ignore_index=True).sort_values(
        ["tier", "profile_pitches"], ascending=[True, False]
    )


# ---------------------------------------------------------------------------
# Step 3 — Build profiles for selected players
# ---------------------------------------------------------------------------

def build_selected_profiles(selected: pd.DataFrame, df: pd.DataFrame) -> None:
    pids = selected["player_id"].tolist()
    print(f"\nResolving player names for {len(pids)} players...")
    _build_name_cache(pids)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        pid = int(row["player_id"])
        profile = build_profile(pid, df, date_cutoff="2025-01-01")
        save_profile(profile)


# ---------------------------------------------------------------------------
# Step 4 — Profile summary
# ---------------------------------------------------------------------------

def print_profile_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    print(f"\n{'='*95}")
    print("  PROFILE SUMMARY")
    print(f"  {'-'*93}")
    hdr = f"  {'Player':<22} | {'Tier':>4} | {'Profile pitches':>15} | {'Swing':>6} | {'Chase':>6} | {'Contact':>7} | {'HardHit':>7} | {'Whiff':>6}"
    print(hdr)
    print(f"  {'-'*93}")

    for _, row in selected.iterrows():
        pid = int(row["player_id"])
        path = PROFILE_DIR / f"{pid}.json"
        with open(path) as f:
            p = json.load(f)

        name = p["player_name"]
        tier = row["tier"]
        n    = p["sample_size"]

        if n == 0:
            print(f"  *** {name} has 0 profile pitches — FLAGGED, will skip evaluation ***")

        print(f"  {name:<22} | {tier:>4} | {n:>15,} | {p['swing_rate']:>6.3f} | "
              f"{p['chase_rate']:>6.3f} | {p['contact_rate']:>7.3f} | "
              f"{p['hard_hit_rate']:>7.3f} | {p['whiff_rate']:>6.3f}")

        rows.append({
            "player_id":       pid,
            "player_name":     name,
            "tier":            tier,
            "profile_pitches": n,
            "swing_rate":      p["swing_rate"],
            "chase_rate":      p["chase_rate"],
            "contact_rate":    p["contact_rate"],
            "hard_hit_rate":   p["hard_hit_rate"],
            "whiff_rate":      p["whiff_rate"],
        })

    print(f"  {'='*93}\n")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5 — Fetch 2025 data
# ---------------------------------------------------------------------------

def fetch_2025(player_id: int, player_name: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{player_id}_2025.parquet"

    if cache_path.exists():
        print(f"  {player_name}: loading from cache")
        return pd.read_parquet(cache_path)

    print(f"  {player_name}: fetching from Baseball Savant...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pybaseball.statcast_batter(SEASON_START, SEASON_END, player_id=player_id)

    if df is not None and len(df) > 0:
        df.to_parquet(cache_path, index=False)
        print(f"    → {len(df):,} pitches cached")
    else:
        print(f"    → no 2025 data")
        df = pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# Step 6 — Preprocess + predict
# ---------------------------------------------------------------------------

def preprocess_2025(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
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
    df["hit"]                = df["events"].isin(HIT_EVENTS).astype(int)
    df["xwoba"]              = df["estimated_woba_using_speedangle"].fillna(0.0)
    df["count_str"]          = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    return df.reset_index(drop=True)


def load_profile_features(player_id: int) -> dict:
    profile = load_profile(player_id, PROFILE_DIR)
    return profile_to_feature_dict(profile)


_pitcher_df_eval = None

def _get_pitcher_df():
    global _pitcher_df_eval
    if _pitcher_df_eval is None:
        from src.pitchers.features import load_pitcher_features
        _pitcher_df_eval = load_pitcher_features()
    return _pitcher_df_eval


def predict_batch(df: pd.DataFrame, pf: dict, feature_cols: list,
                  model, calibrator) -> tuple:
    from src.data.preprocess import PITCHER_FEATURES
    df = df.copy()
    for col, val in pf.items():
        if col in feature_cols:
            df[col] = val

    # Merge pitcher identity features per row using the pitcher column
    if any(f in feature_cols for f in PITCHER_FEATURES) and "pitcher" in df.columns:
        pitcher_df = _get_pitcher_df()
        df = df.drop(columns=[c for c in PITCHER_FEATURES if c in df.columns])
        df = df.merge(pitcher_df, on="pitcher", how="left")
        for col in PITCHER_FEATURES:
            if col in feature_cols:
                df[col] = df[col].fillna(pitcher_df[col].median())

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[feature_cols].values.astype(float)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_medians, inds[1])

    raw = model.predict(X)
    raw = np.clip(raw, 0.0, None)
    cal = calibrator.predict(raw) if calibrator is not None else raw
    cal = np.clip(cal, 0.0, None)
    return raw, cal


def compute_metrics(y_xwoba: np.ndarray, y_hit: np.ndarray, preds: np.ndarray) -> dict:
    n = len(y_xwoba)
    if n == 0:
        return {"mae": np.nan, "pearson_r": np.nan, "auc": np.nan,
                "mean_xwoba": np.nan, "mean_pred": np.nan, "n": n}
    mae = float(mean_absolute_error(y_xwoba, preds))
    r, _ = pearsonr(y_xwoba, preds) if y_xwoba.std() > 0 else (np.nan, 1.0)
    auc = np.nan
    if y_hit.sum() > 0 and y_hit.sum() < n:
        auc = float(roc_auc_score(y_hit, preds))
    return {
        "mae":        mae,
        "pearson_r":  float(r),
        "auc":        auc,
        "mean_xwoba": float(y_xwoba.mean()),
        "mean_pred":  float(preds.mean()),
        "n":          n,
    }


# ---------------------------------------------------------------------------
# Step 7 — Print results
# ---------------------------------------------------------------------------

def print_full_table(results: pd.DataFrame) -> None:
    r = results.sort_values("profile_pitches", ascending=False)
    print(f"\n{'='*100}")
    print("  FULL RESULTS — sorted by profile sample size")
    print(f"  {'-'*98}")
    hdr = (f"  {'Player':<22} | {'Tier':>4} | {'Profile pitches':>15} | "
           f"{'2025 pitches':>12} | {'MAE':>7} | {'Pearson r':>9} | {'AUC':>6}")
    print(hdr)
    print(f"  {'-'*98}")
    for _, row in r.iterrows():
        mae_s = f"{row['mae']:.4f}" if not np.isnan(row['mae']) else "   n/a"
        r_s   = f"{row['pearson_r']:.3f}" if not np.isnan(row['pearson_r']) else "    n/a"
        auc_s = f"{row['auc']:.3f}" if not np.isnan(row['auc']) else "  n/a"
        print(f"  {row['player_name']:<22} | {row['tier']:>4} | {int(row['profile_pitches']):>15,} | "
              f"{int(row['n_2025']):>12,} | {mae_s:>7} | {r_s:>9} | {auc_s:>6}")
    print(f"  {'='*100}")


def print_tier_averages(results: pd.DataFrame) -> None:
    tier_labels = {"A": "15,000+", "B": "8,000–15,000", "C": "3,000–8,000", "D": "under 3,000"}
    print(f"\n{'='*85}")
    print("  TIER AVERAGES")
    print(f"  {'-'*83}")
    hdr = (f"  {'Tier':>4} | {'Profile range':<14} | {'Avg profile':>11} | "
           f"{'Avg MAE':>8} | {'Avg Pearson r':>13} | {'Avg AUC':>8}")
    print(hdr)
    print(f"  {'-'*83}")
    for tier in ["A", "B", "C", "D"]:
        sub = results[results["tier"] == tier].dropna(subset=["mae"])
        if len(sub) == 0:
            continue
        print(f"  {tier:>4} | {tier_labels[tier]:<14} | "
              f"{sub['profile_pitches'].mean():>11,.0f} | "
              f"{sub['mae'].mean():>8.4f} | "
              f"{sub['pearson_r'].mean():>13.3f} | "
              f"{sub['auc'].mean():>8.3f}")
    print(f"  {'='*85}")


def print_correlations(results: pd.DataFrame) -> None:
    valid = results.dropna(subset=["mae", "pearson_r"])
    if len(valid) < 3:
        print("\n  Not enough valid results for correlation.")
        return
    r_mae, p_mae = stats.pearsonr(valid["profile_pitches"], valid["mae"])
    r_r,   p_r   = stats.pearsonr(valid["profile_pitches"], valid["pearson_r"])
    print(f"\n  Pearson r (profile pitches vs MAE):             {r_mae:+.3f}  (p={p_mae:.3f})")
    print(f"  Pearson r (profile pitches vs Pearson r):       {r_r:+.3f}  (p={p_r:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("\nLoading model and processed data...")
    model, feature_cols, _, calibrator = load_model(MODEL_DIR)
    df = pd.read_parquet(PROCESSED_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Step 2 — Select players
    print("\nSelecting 20 players across 4 tiers (seed=42)...")
    selected = select_players(df)
    print(f"\n  {'Player ID':>10} | {'Tier':>4} | {'Profile pitches':>15}")
    print(f"  {'-'*35}")
    for _, row in selected.iterrows():
        print(f"  {int(row['player_id']):>10} | {row['tier']:>4} | {int(row['profile_pitches']):>15,}")

    # Step 3 — Build profiles
    print("\nBuilding profiles...")
    build_selected_profiles(selected, df)

    # Step 4 — Profile summary
    profile_df = print_profile_summary(selected)

    # Check for zero-sample profiles
    zero = profile_df[profile_df["profile_pitches"] == 0]
    if len(zero) > 0:
        print(f"\n  WARNING: {len(zero)} player(s) with 0 profile pitches — skipping in evaluation:")
        for _, r in zero.iterrows():
            print(f"    {r['player_name']} (id={r['player_id']})")

    # Step 5+6 — Fetch, preprocess, predict
    results = []
    for _, prow in profile_df.iterrows():
        pid  = int(prow["player_id"])
        name = prow["player_name"]
        tier = prow["tier"]
        prof_n = prow["profile_pitches"]

        if prof_n == 0:
            continue

        print(f"\n[{name}  |  Tier {tier}]")
        raw_2025 = fetch_2025(pid, name)
        if len(raw_2025) == 0:
            print(f"  No 2025 data — skipping.")
            continue

        df_2025 = preprocess_2025(raw_2025)
        if len(df_2025) < 50:
            print(f"  Only {len(df_2025)} usable pitches — skipping.")
            continue

        pf = load_profile_features(pid)
        _, cal_preds = predict_batch(df_2025, pf, feature_cols, model, calibrator)
        m = compute_metrics(df_2025["xwoba"].values, df_2025["hit"].values, cal_preds)

        print(f"  Pitches: {m['n']:,}  |  MAE: {m['mae']:.4f}  |  "
              f"Pearson r: {m['pearson_r']:.3f}  |  AUC: {m['auc']:.3f}")

        results.append({
            "player_id":       pid,
            "player_name":     name,
            "tier":            tier,
            "profile_pitches": prof_n,
            "n_2025":          m["n"],
            "mean_xwoba":      m["mean_xwoba"],
            "mean_pred":       m["mean_pred"],
            "mae":             m["mae"],
            "pearson_r":       m["pearson_r"],
            "auc":             m["auc"],
        })

    if not results:
        print("\nNo results to report.")
        return

    results_df = pd.DataFrame(results)

    # Step 7 — Print
    print_full_table(results_df)
    print_tier_averages(results_df)
    print_correlations(results_df)

    # Save
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
