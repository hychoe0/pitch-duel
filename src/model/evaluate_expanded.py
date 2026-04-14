"""
evaluate_expanded.py — Expanded 20-player validation across 4 profile-depth tiers.

Tests whether profile sample size drives model accuracy by randomly sampling
5 players per tier and evaluating on their held-out 2025 Statcast data.

Primary metric: composite P(hard) = P(swing) × P(contact|swing) × P(hard|contact)
All three stages use stage-specific calibrators from models_v2/.

Usage:
    python -m src.model.evaluate_expanded
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.data.preprocess import (
    HIT_EVENTS,
    add_contextual_hitter_features,
    encode_handedness,
    encode_pitch_types,
    encode_prev_result,
    impute_pitcher_medians,
    make_era_flag,
    make_prev_pitch_features,
    make_runner_flags,
    make_score_diff,
    make_spin_to_velo_ratio,
    make_times_through_order,
)
from src.hitters.profiles import (
    _build_name_cache,
    _NAME_CACHE,
    build_profile,
    load_profile,
    profile_to_feature_dict,
    save_profile,
    PROFILE_DIR,
)

warnings.filterwarnings("ignore")

PROCESSED_PATH = Path("data/processed/statcast_processed_v2.parquet")
CACHE_DIR      = Path("data/processed/eval_2025")
RESULTS_PATH   = Path("data/processed/validation_results.csv")
TIER_PATH      = Path("data/processed/validation_tier_summary.csv")
MODEL_DIR      = Path("models_v2")

SEASON_START = "2025-03-01"
SEASON_END   = "2025-10-31"
RANDOM_SEED  = 42
PROFILE_CUTOFF = "2025-01-01"

TIERS = {
    "A": (15_000, None),
    "B": (8_000,  15_000),
    "C": (3_000,  8_000),
    "D": (0,      3_000),
}
TIER_LABELS = {
    "A": "15,000+",
    "B": "8,000–15,000",
    "C": "3,000–8,000",
    "D": "under 3,000",
}


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}


def _load_pipeline() -> dict:
    if _pipeline_cache:
        return _pipeline_cache

    for stage in ["swing", "contact", "hard_contact"]:
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_DIR / f"{stage}_model.json"))
        cal_path = MODEL_DIR / f"{stage}_calibrator.pkl"
        cal = joblib.load(cal_path) if cal_path.exists() else None
        _pipeline_cache[stage] = (m, cal)

    with open(MODEL_DIR / "feature_cols.json") as f:
        _pipeline_cache["feature_cols"] = json.load(f)
    with open(MODEL_DIR / "encodings.json") as f:
        _pipeline_cache["encodings"] = json.load(f)

    return _pipeline_cache


# ---------------------------------------------------------------------------
# Step 1 — Player selection
# ---------------------------------------------------------------------------

def select_players(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(PROFILE_CUTOFF)
    train  = df[df["game_date"] < cutoff]
    counts = train.groupby("batter").size().rename("profile_pitches").reset_index()
    counts.columns = ["player_id", "profile_pitches"]

    active_2025 = set(df[df["game_date"] >= cutoff]["batter"].unique())
    counts = counts[counts["player_id"].isin(active_2025)]
    print(f"  Pool: {len(counts):,} players with both training data and 2025 appearances.")

    selected = []
    for tier, (lo, hi) in TIERS.items():
        mask = counts["profile_pitches"] >= lo
        if hi is not None:
            mask &= counts["profile_pitches"] < hi
        pool   = counts[mask].copy()
        chosen = pool.sample(n=min(5, len(pool)), random_state=RANDOM_SEED)
        chosen = chosen.copy()
        chosen["tier"] = tier
        selected.append(chosen)

    return pd.concat(selected, ignore_index=True).sort_values(
        ["tier", "profile_pitches"], ascending=[True, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — Build / verify profiles
# ---------------------------------------------------------------------------

def build_selected_profiles(selected: pd.DataFrame, df: pd.DataFrame) -> None:
    pids = selected["player_id"].tolist()
    print(f"  Resolving names for {len(pids)} players...")
    _build_name_cache(pids)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        pid = int(row["player_id"])
        profile_path = PROFILE_DIR / f"{pid}.json"
        if profile_path.exists():
            with open(profile_path) as f:
                existing = json.load(f)
            # Rebuild if profile predates the 2022+ era or has suspiciously low count
            if existing.get("sample_size", 0) == 0:
                print(f"    Rebuilding stale/empty profile for {pid}...")
            else:
                continue  # profile looks fine
        profile = build_profile(pid, df, date_cutoff=PROFILE_CUTOFF)
        save_profile(profile, PROFILE_DIR)


def print_profile_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sep = "─" * 100
    print(f"\n{sep}")
    print("  PROFILE SUMMARY")
    print(sep)
    hdr = (f"  {'Player':<24} | {'Tier':>4} | {'Profile N':>9} | "
           f"{'Swing':>6} | {'Chase':>6} | {'Contact':>7} | {'HardHit':>7} | {'Thin?':>5}")
    print(hdr)
    print(sep)

    for _, row in selected.iterrows():
        pid  = int(row["player_id"])
        path = PROFILE_DIR / f"{pid}.json"
        if not path.exists():
            print(f"  *** profile missing for {pid} — will skip ***")
            continue
        with open(path) as f:
            p = json.load(f)

        name = _NAME_CACHE.get(pid, p.get("player_name", f"Player {pid}"))
        thin = "Yes" if p.get("is_thin_sample") else "No"
        print(f"  {name:<24} | {row['tier']:>4} | {p['sample_size']:>9,} | "
              f"{p['swing_rate']:>6.3f} | {p['chase_rate']:>6.3f} | "
              f"{p['contact_rate']:>7.3f} | {p['hard_hit_rate']:>7.3f} | {thin:>5}")

        rows.append({
            "player_id":       pid,
            "player_name":     name,
            "tier":            row["tier"],
            "profile_pitches": p["sample_size"],
            "swing_rate":      p["swing_rate"],
            "chase_rate":      p["chase_rate"],
            "contact_rate":    p["contact_rate"],
            "hard_hit_rate":   p["hard_hit_rate"],
            "whiff_rate":      p["whiff_rate"],
            "is_thin":         p.get("is_thin_sample", False),
        })

    print(sep + "\n")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — Fetch 2025 data
# ---------------------------------------------------------------------------

def fetch_2025(player_id: int, player_name: str) -> pd.DataFrame:
    import pybaseball

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{player_id}_2025.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  {player_name}: {len(df):,} pitches (cached)")
        return df

    print(f"  {player_name}: fetching from Baseball Savant...", end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pybaseball.statcast_batter(SEASON_START, SEASON_END, player_id=player_id)

    if df is not None and len(df) > 0:
        df.to_parquet(cache_path, index=False)
        print(f"{len(df):,} pitches fetched and cached.")
    else:
        print("no 2025 data.")
        df = pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# Step 4 — Preprocess 2025 data
# ---------------------------------------------------------------------------

def preprocess_2025(df: pd.DataFrame, hitter_features: dict) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "UN")]
    df = df[df["release_speed"].notna()]
    df = df[df["plate_x"].notna() & df["plate_z"].notna()]
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    # Standard feature engineering
    df = make_prev_pitch_features(df)
    df = impute_pitcher_medians(df)
    df = encode_pitch_types(df)
    df = encode_prev_result(df)
    df = encode_handedness(df)
    df = make_runner_flags(df)
    df = make_times_through_order(df)
    df["era_flag_enc"]       = make_era_flag(df)
    df["spin_to_velo_ratio"] = make_spin_to_velo_ratio(df)
    df["score_diff"]         = make_score_diff(df)

    # Inject hitter profile features as constants (per-hitter evaluation)
    for col, val in hitter_features.items():
        df[col] = val

    # Contextual features: must come after hitter profile columns are present
    df = add_contextual_hitter_features(df)

    # Binary hit outcome for AUC
    df["hit"] = df["events"].isin(HIT_EVENTS).astype(int)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Three-stage batch prediction
# ---------------------------------------------------------------------------

def predict_three_stage(df: pd.DataFrame, pipeline: dict) -> np.ndarray:
    """
    Returns composite P(hard) = P(swing) × P(contact|swing) × P(hard|contact)
    for each row in df. Missing features filled with column median.
    """
    feature_cols = pipeline["feature_cols"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].values.astype(float)
    # Fill any NaNs with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        inds = np.where(nan_mask)
        X[inds] = np.take(col_medians, inds[1])

    def _stage(key: str) -> np.ndarray:
        model, cal = pipeline[key]
        raw = model.predict_proba(X)[:, 1]
        if cal is not None:
            cal_vals = cal.predict(raw)
        else:
            cal_vals = raw
        return np.clip(cal_vals, 0.0, 0.97)

    p_swing   = _stage("swing")
    p_contact = _stage("contact")
    p_hard    = _stage("hard_contact")

    return p_swing * p_contact * p_hard


# ---------------------------------------------------------------------------
# Step 4 — Compute per-player metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_hit: np.ndarray, p_composite: np.ndarray) -> dict:
    n = len(y_hit)
    if n == 0:
        return {"auc": np.nan, "brier": np.nan, "cal_error": np.nan,
                "mean_pred": np.nan, "actual_hit_rate": np.nan, "n": 0}

    actual_rate = float(y_hit.mean())
    mean_pred   = float(p_composite.mean())
    brier       = float(np.mean((p_composite - y_hit) ** 2))

    # AUC: composite P(hard) vs binary hit_into_play
    if y_hit.sum() > 0 and y_hit.sum() < n:
        auc = float(roc_auc_score(y_hit, p_composite))
    else:
        auc = np.nan

    # Calibration error: bin predictions, compare bin mean pred to bin actual rate
    n_bins = 10
    bin_edges = np.linspace(p_composite.min(), p_composite.max() + 1e-9, n_bins + 1)
    bin_idx   = np.digitize(p_composite, bin_edges) - 1
    bin_idx   = np.clip(bin_idx, 0, n_bins - 1)
    cal_errs  = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        cal_errs.append(abs(p_composite[mask].mean() - y_hit[mask].mean()))
    cal_error = float(np.mean(cal_errs)) if cal_errs else np.nan

    return {
        "auc":             auc,
        "brier":           brier,
        "cal_error":       cal_error,
        "mean_pred":       mean_pred,
        "actual_hit_rate": actual_rate,
        "n":               n,
    }


# ---------------------------------------------------------------------------
# Step 5 — Print results
# ---------------------------------------------------------------------------

def print_full_table(results: pd.DataFrame) -> None:
    r   = results.sort_values("profile_pitches", ascending=False)
    sep = "─" * 106
    print(f"\n{sep}")
    print("  FULL RESULTS — sorted by profile sample size")
    print(sep)
    hdr = (f"  {'Player':<24} | {'Tier':>4} | {'Profile N':>9} | {'2025 N':>6} | "
           f"{'AUC':>6} | {'Brier':>7} | {'Cal Err':>7} | {'Mean P(hard)':>12} | {'Hit Rate':>8}")
    print(hdr)
    print(sep)
    for _, row in r.iterrows():
        auc_s = f"{row['auc']:.3f}"   if not np.isnan(row['auc'])       else "  n/a"
        br_s  = f"{row['brier']:.4f}" if not np.isnan(row['brier'])     else "   n/a"
        ce_s  = f"{row['cal_error']:.4f}" if not np.isnan(row['cal_error']) else "   n/a"
        print(f"  {row['player_name']:<24} | {row['tier']:>4} | {int(row['profile_pitches']):>9,} | "
              f"{int(row['n']):>6,} | {auc_s:>6} | {br_s:>7} | {ce_s:>7} | "
              f"{row['mean_pred']:>12.4f} | {row['actual_hit_rate']:>8.4f}")
    print(sep)


def print_tier_averages(results: pd.DataFrame) -> pd.DataFrame:
    sep = "─" * 90
    print(f"\n{sep}")
    print("  TIER AVERAGES")
    print(sep)
    hdr = (f"  {'Tier':>4} | {'Profile range':<14} | {'Avg profile N':>13} | "
           f"{'Avg AUC':>8} | {'Avg Brier':>9} | {'Avg Cal Err':>11}")
    print(hdr)
    print(sep)

    tier_rows = []
    for tier in ["A", "B", "C", "D"]:
        sub = results[results["tier"] == tier].dropna(subset=["auc"])
        if len(sub) == 0:
            continue
        avg_prof = sub["profile_pitches"].mean()
        avg_auc  = sub["auc"].mean()
        avg_br   = sub["brier"].mean()
        avg_ce   = sub["cal_error"].mean()
        print(f"  {tier:>4} | {TIER_LABELS[tier]:<14} | {avg_prof:>13,.0f} | "
              f"{avg_auc:>8.3f} | {avg_br:>9.4f} | {avg_ce:>11.4f}")
        tier_rows.append({"tier": tier, "label": TIER_LABELS[tier],
                          "avg_profile_pitches": avg_prof, "avg_auc": avg_auc,
                          "avg_brier": avg_br, "avg_cal_error": avg_ce,
                          "n_players": len(sub)})
    print(sep)
    return pd.DataFrame(tier_rows)


def print_correlations(results: pd.DataFrame) -> dict:
    valid = results.dropna(subset=["auc", "cal_error"])
    if len(valid) < 3:
        print("\n  Not enough valid results for correlation.")
        return {}

    r_auc, p_auc = stats.pearsonr(valid["profile_pitches"], valid["auc"])
    r_ce,  p_ce  = stats.pearsonr(valid["profile_pitches"], valid["cal_error"])

    print(f"\n  Pearson r (profile N vs AUC):            {r_auc:+.3f}  (p={p_auc:.3f})")
    print(f"  Pearson r (profile N vs calibration err): {r_ce:+.3f}  (p={p_ce:.3f})")
    print(f"  Minimum AUC across all players:           {valid['auc'].min():.3f}  ({valid.loc[valid['auc'].idxmin(), 'player_name']})")

    # Flag any Tier D beating a Tier A player on AUC
    tier_a = results[results["tier"] == "A"]["auc"].dropna()
    tier_d = results[results["tier"] == "D"]["auc"].dropna()
    if len(tier_a) > 0 and len(tier_d) > 0:
        crossovers = [(results.loc[results["auc"] == da, "player_name"].values[0], da)
                      for da in tier_d if da > tier_a.min()]
        if crossovers:
            print(f"\n  *** Tier D players beating a Tier A player on AUC:")
            for cname, cauc in crossovers:
                print(f"      {cname}: AUC={cauc:.3f} (Tier A min={tier_a.min():.3f})")
        else:
            print(f"\n  No Tier D player beats any Tier A player on AUC.")

    return {"r_auc": r_auc, "r_cal": r_ce}


# ---------------------------------------------------------------------------
# Step 6 — Verdicts
# ---------------------------------------------------------------------------

def print_verdicts(
    results: pd.DataFrame,
    tier_df: pd.DataFrame,
    corr: dict,
    hit_into_play_phards: list,
    swinging_strike_phards: list,
) -> None:
    print(f"\n{'=' * 60}")
    print("  STEP 6 — PASS/FAIL VERDICTS")
    print("=" * 60)
    print("\n  Note: composite P(hard) is a damage index, not a hit probability.")
    print("  It should not be compared directly to hit rates.")
    passes = 0

    # VERDICT 1 — AUC floor
    valid_auc = results["auc"].dropna()
    v1 = bool(valid_auc.min() > 0.68) if len(valid_auc) > 0 else False
    passes += v1
    print(f"\nVERDICT 1 — AUC floor (all > 0.68): {'PASS' if v1 else 'FAIL'}")
    print(f"  Min AUC = {valid_auc.min():.3f}  Max AUC = {valid_auc.max():.3f}")
    if not v1:
        below = results[results["auc"] <= 0.68][["player_name", "tier", "auc"]]
        for _, r in below.iterrows():
            print(f"  BELOW THRESHOLD: {r['player_name']} ({r['tier']}) AUC={r['auc']:.3f}")

    # VERDICT 2 — Tier gradient
    a_auc = tier_df[tier_df["tier"] == "A"]["avg_auc"].values
    d_auc = tier_df[tier_df["tier"] == "D"]["avg_auc"].values
    if len(a_auc) > 0 and len(d_auc) > 0:
        v2 = bool(a_auc[0] > d_auc[0])
        print(f"\nVERDICT 2 — Tier gradient (A > D): {'PASS' if v2 else 'FAIL'}")
        print(f"  Tier A avg AUC = {a_auc[0]:.3f}  Tier D avg AUC = {d_auc[0]:.3f}")
    else:
        v2 = False
        print(f"\nVERDICT 2 — Tier gradient: SKIP (missing tier data)")
    passes += v2

    # VERDICT 3 — Contact quality signal
    # P(hard) on hit_into_play should exceed P(hard) on swinging_strike by >= 0.020
    # across all 20 players combined.
    if hit_into_play_phards and swinging_strike_phards:
        mean_hip = float(np.mean(hit_into_play_phards))
        mean_ws  = float(np.mean(swinging_strike_phards))
        gap      = mean_hip - mean_ws
        v3 = gap >= 0.020
    else:
        mean_hip = mean_ws = gap = float("nan")
        v3 = False
    passes += v3
    print(f"\nVERDICT 3 — Contact quality signal (hit_into_play P(hard) > swinging_strike + 0.020): "
          f"{'PASS' if v3 else 'FAIL'}")
    print(f"  Mean P(hard) on hit_into_play:   {mean_hip:.4f}  (n={len(hit_into_play_phards):,})")
    print(f"  Mean P(hard) on swinging_strike: {mean_ws:.4f}  (n={len(swinging_strike_phards):,})")
    print(f"  Gap = {gap:.4f}  (threshold: ≥ 0.020)")

    # VERDICT 4 — Profile signal (r_auc)
    r_auc = corr.get("r_auc", np.nan)
    v4 = bool(r_auc > 0.20) if not np.isnan(r_auc) else False
    passes += v4
    print(f"\nVERDICT 4 — Profile signal (Pearson r > 0.20): {'PASS' if v4 else 'FAIL'}")
    print(f"  Pearson r (profile N vs AUC) = {r_auc:+.3f}")

    # Final verdict
    print(f"\n{'=' * 60}")
    print(f"  {passes}/4 verdicts passed.")
    if passes == 4:
        print("  Model is validated across player types. Ready for broad demo.")
    else:
        failed = []
        if not v1: failed.append("V1 AUC floor")
        if not v2: failed.append("V2 tier gradient")
        if not v3: failed.append("V3 contact quality signal")
        if not v4: failed.append("V4 profile signal")
        print(f"  Failed: {', '.join(failed)}")
        if not v1:
            print("  → AUC floor failure: some player types not reliably ranked.")
        if not v2:
            print("  → No tier gradient: profile depth may not drive accuracy.")
        if not v3:
            print("  → Contact quality signal weak: composite P(hard) not tracking damage.")
        if not v4:
            print("  → Weak profile signal: consider stronger hitter features.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("\n" + "=" * 70)
    print("PITCH DUEL — Expanded 20-Player Validation")
    print(f"Model: {MODEL_DIR}  |  Metric: composite P(hard) [three-stage]")
    print("=" * 70)

    print("\nLoading models...")
    pipeline = _load_pipeline()
    print("  done.")

    print("\nLoading processed parquet...")
    df = pd.read_parquet(PROCESSED_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  {len(df):,} rows loaded.")

    # Step 1 — Select players
    print("\nSTEP 1 — Selecting 20 players across 4 tiers (seed=42)...")
    selected = select_players(df)

    print(f"\n  {'Player ID':>10} | {'Tier':>4} | {'Profile pitches':>15}")
    print(f"  {'─'*35}")
    for _, row in selected.iterrows():
        print(f"  {int(row['player_id']):>10} | {row['tier']:>4} | {int(row['profile_pitches']):>15,}")

    # Step 2 — Build profiles
    print("\nSTEP 2 — Building/verifying profiles...")
    build_selected_profiles(selected, df)
    profile_df = print_profile_summary(selected)

    # Step 3 — Fetch 2025 data
    print("\nSTEP 3 — Fetching 2025 evaluation data...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 4 — Run predictions
    print("\nSTEP 4 — Running three-stage predictions on 2025 data...")
    results = []
    # Accumulated across all players for V3
    all_hit_into_play_phards:    list = []
    all_swinging_strike_phards:  list = []

    for _, prow in profile_df.iterrows():
        pid       = int(prow["player_id"])
        name      = prow["player_name"]
        tier      = prow["tier"]
        prof_n    = prow["profile_pitches"]

        print(f"\n  [{name}  |  Tier {tier}  |  Profile N={prof_n:,}]")

        raw_2025 = fetch_2025(pid, name)
        if len(raw_2025) < 50:
            print(f"  → Only {len(raw_2025)} pitches — skipping.")
            continue

        hitter_features = profile_to_feature_dict(load_profile(pid, PROFILE_DIR))
        df_2025 = preprocess_2025(raw_2025, hitter_features)

        if len(df_2025) < 50:
            print(f"  → Only {len(df_2025)} usable pitches after preprocessing — skipping.")
            continue

        try:
            p_composite = predict_three_stage(df_2025, pipeline)
        except ValueError as e:
            print(f"  → Prediction failed: {e}")
            continue

        m = compute_metrics(df_2025["hit"].values, p_composite)
        print(f"  → Pitches: {m['n']:,}  AUC: {m['auc']:.3f}  "
              f"Brier: {m['brier']:.4f}  Actual hit rate: {m['actual_hit_rate']:.4f}")

        # Accumulate outcome-grouped P(hard) for V3
        desc = df_2025["description"] if "description" in df_2025.columns else pd.Series([], dtype=str)
        hip_mask = desc.isin({"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"})
        ws_mask  = desc.isin({"swinging_strike", "swinging_strike_blocked"})
        all_hit_into_play_phards.extend(p_composite[hip_mask.values].tolist())
        all_swinging_strike_phards.extend(p_composite[ws_mask.values].tolist())

        results.append({
            "player_id":       pid,
            "player_name":     name,
            "tier":            tier,
            "profile_pitches": prof_n,
            "n":               m["n"],
            "auc":             m["auc"],
            "brier":           m["brier"],
            "cal_error":       m["cal_error"],
            "mean_pred":       m["mean_pred"],
            "actual_hit_rate": m["actual_hit_rate"],
            "is_thin":         prow["is_thin"],
        })

    if not results:
        print("\nNo results to report.")
        return

    results_df = pd.DataFrame(results)

    # Step 5 — Print results
    print("\n" + "=" * 70)
    print("STEP 5 — RESULTS")
    print_full_table(results_df)
    tier_df = print_tier_averages(results_df)
    corr    = print_correlations(results_df)

    # Step 6 — Verdicts
    print_verdicts(results_df, tier_df, corr, all_hit_into_play_phards, all_swinging_strike_phards)

    # Step 7 — Save
    results_df.to_csv(RESULTS_PATH, index=False)
    tier_df.to_csv(TIER_PATH, index=False)
    print(f"\n  Saved: {RESULTS_PATH}")
    print(f"  Saved: {TIER_PATH}")
    print()


if __name__ == "__main__":
    run()
