"""
validate_hitters.py — Structured validation of three-stage model on 7 named hitters.

Processes Baseball Savant CSV files from data/eval/.
Player IDs are resolved ground-truth from the CSV batter column — no hardcoded IDs.
Produces a terminal report with per-hitter metrics and 5 explicit PASS/FAIL checks.

Usage:
    python -m src.model.validate_hitters
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR   = Path("models_v2")
PROFILE_DIR = Path("data/processed/profiles")
CSV_DIR     = Path("data/eval")
PARQUET_PATH = Path("data/processed/statcast_processed_v2.parquet")

# Outcome groupings from Statcast description column
OUTCOME_GROUPS = {
    "swinging_strike":         "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "called_strike":           "called_strike",
    "ball":                    "ball",
    "blocked_ball":            "ball",
    "foul":                    "foul",
    "foul_tip":                "foul",
    "foul_bunt":               "foul",
    "hit_into_play":           "hit_into_play",
    "hit_into_play_no_out":    "hit_into_play",
    "hit_into_play_score":     "hit_into_play",
}
GRADIENT_ORDER = ["swinging_strike", "called_strike", "foul", "hit_into_play"]

BREAKING_TYPES = {"SL", "CU", "KC", "SV", "CS", "ST"}

_parquet_cache: dict = {}


# ---------------------------------------------------------------------------
# Step 1 — Resolve player IDs + names from CSVs (ground-truth)
# ---------------------------------------------------------------------------

def _resolve_csv_hitters(csv_dir: Path) -> list:
    """
    For each CSV in csv_dir, read batter ID and player name directly.
    Returns list of (csv_path, player_id, player_name) sorted by player_name.
    """
    resolved = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, usecols=lambda c: c in ("batter", "player_name"), nrows=200)
        if "batter" not in df.columns:
            print(f"  WARN: {csv_path.name} has no 'batter' column — skipping.")
            continue
        player_id = int(df["batter"].dropna().iloc[0])
        if "player_name" in df.columns:
            raw_name = df["player_name"].dropna().iloc[0]
            # Savant exports "Last, First" — reformat to "First Last"
            parts = [p.strip() for p in str(raw_name).split(",")]
            player_name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else raw_name
        else:
            player_name = f"Player {player_id}"
        print(f"  Resolved {player_name} → ID {player_id}  ({csv_path.name})")
        resolved.append((csv_path, player_id, player_name))
    return resolved


# ---------------------------------------------------------------------------
# Step 2 — Profile loading with parquet build fallback
# ---------------------------------------------------------------------------

def _get_parquet() -> pd.DataFrame:
    """Load statcast parquet once and cache it."""
    if "df" not in _parquet_cache:
        if not PARQUET_PATH.exists():
            _parquet_cache["df"] = None
        else:
            print(f"  Loading parquet {PARQUET_PATH} (one-time)...", end=" ", flush=True)
            _parquet_cache["df"] = pd.read_parquet(PARQUET_PATH)
            print("done.")
    return _parquet_cache["df"]


def _load_or_build_profile(player_id: int, player_name: str):
    """
    Returns (profile, used_fallback, no_mlb_data, source_label).

    Priority:
      1. Load from PROFILE_DIR/{player_id}.json          → source = "json"
      2. Build from parquet (if player has MLB rows)      → source = "built"
      3. League average (no MLB history in parquet)       → source = "league_avg"
    """
    from src.hitters.profiles import (
        build_profile,
        get_league_average_profile,
        load_profile,
        save_profile,
    )

    # 1. Try JSON
    try:
        profile = load_profile(player_id, PROFILE_DIR)
        return profile, False, False, "json"
    except FileNotFoundError:
        pass

    # 2. Try parquet build
    parquet_df = _get_parquet()
    if parquet_df is not None:
        player_rows = parquet_df[parquet_df["batter"] == player_id]
        if len(player_rows) > 0:
            try:
                profile = build_profile(player_id, parquet_df, date_cutoff="2026-01-01")
                profile.player_name = player_name  # override — parquet player_name is pitcher's name
                save_profile(profile, PROFILE_DIR)
                return profile, False, False, "built"
            except Exception as e:
                print(f"  WARN: build_profile failed for {player_name}: {e}")
        else:
            # Player present in CSV but has zero rows in parquet = no MLB history
            avg = get_league_average_profile()
            avg.player_name = player_name
            return avg, True, True, "league_avg"

    # 3. No parquet available at all
    avg = get_league_average_profile()
    avg.player_name = player_name
    return avg, True, False, "league_avg"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pipeline() -> dict:
    import joblib
    import xgboost as xgb

    pipeline = {}
    for stage in ["swing", "contact", "hard_contact"]:
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_DIR / f"{stage}_model.json"))
        cal_path = MODEL_DIR / f"{stage}_calibrator.pkl"
        cal = joblib.load(cal_path) if cal_path.exists() else None
        pipeline[stage] = (m, cal)
    with open(MODEL_DIR / "feature_cols.json") as f:
        pipeline["feature_cols"] = json.load(f)
    with open(MODEL_DIR / "encodings.json") as f:
        pipeline["encodings"] = json.load(f)
    return pipeline


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def _era_flag_enc(game_date) -> int:
    import datetime
    if isinstance(game_date, str):
        try:
            game_date = datetime.date.fromisoformat(str(game_date)[:10])
        except Exception:
            return 3
    elif hasattr(game_date, "date"):
        game_date = game_date.date()
    if game_date < datetime.date(2021, 6, 21):  return 0
    if game_date < datetime.date(2022, 1, 1):   return 1
    if game_date < datetime.date(2023, 1, 1):   return 2
    return 3


def _plate_to_statcast_zone(plate_x: float, plate_z: float) -> int:
    x1, x2 = -0.85 + 1.70 / 3, -0.85 + 2 * 1.70 / 3
    z1, z2 = 1.5 + 2.0 / 3, 1.5 + 2 * 2.0 / 3
    in_x = -0.85 <= plate_x <= 0.85
    in_z = 1.5   <= plate_z <= 3.5
    if in_x and in_z:
        col = 0 if plate_x < x1 else (1 if plate_x <= x2 else 2)
        row = 0 if plate_z > z2 else (1 if plate_z >= z1 else 2)
        return 1 + row * 3 + col
    high = plate_z >= 2.5
    left = plate_x < 0
    if high: return 11 if left else 12
    return 13 if left else 14


def _build_row(row: pd.Series, hitter_features: dict, feature_cols: list, encodings: dict) -> np.ndarray:
    from src.hitters.profiles import PITCH_FAMILIES

    ptm = encodings["PITCH_TYPE_MAP"]
    prm = encodings["PREV_RESULT_MAP"]

    pitch_type      = str(row.get("pitch_type", "FF") or "FF")
    prev_pitch_type = str(row.get("prev_pitch_type", "FIRST_PITCH") or "FIRST_PITCH")
    prev_result     = str(row.get("pitch_description_prev", "FIRST_PITCH") or "FIRST_PITCH")
    if prev_result not in prm:
        prev_result = "FIRST_PITCH" if prev_pitch_type == "FIRST_PITCH" else "OTHER"

    release_speed = float(row.get("release_speed", 90) or 90)
    spin_rate     = float(row.get("release_spin_rate", 0) or 0)
    balls         = int(row.get("balls", 0) or 0)
    strikes       = int(row.get("strikes", 0) or 0)
    plate_x       = float(row.get("plate_x", 0) or 0)
    plate_z       = float(row.get("plate_z", 2.5) or 2.5)

    zone = _plate_to_statcast_zone(plate_x, plate_z)
    family = "other"
    for fam, types in PITCH_FAMILIES.items():
        if pitch_type in types:
            family = fam
            break

    swing_z   = hitter_features.get(f"hitter_swing_rate_z{zone}",         hitter_features.get("hitter_swing_rate", 0.47))
    whiff_z   = hitter_features.get(f"hitter_whiff_rate_z{zone}",         hitter_features.get("hitter_whiff_rate", 0.25))
    swing_fam = hitter_features.get(f"hitter_swing_rate_{family}",        hitter_features.get("hitter_swing_rate", 0.47))
    whiff_fam = hitter_features.get(f"hitter_whiff_rate_{family}",        hitter_features.get("hitter_whiff_rate", 0.25))
    swing_cnt = hitter_features.get(f"hitter_swing_rate_count_{balls}_{strikes}", hitter_features.get("hitter_swing_rate", 0.47))

    assembled = {
        **hitter_features,
        "release_speed":          release_speed,
        "spin_to_velo_ratio":     spin_rate / release_speed if release_speed else 0.0,
        "pfx_x":                  float(row.get("pfx_x", 0) or 0),
        "pfx_z":                  float(row.get("pfx_z", 0) or 0),
        "release_pos_x":          float(row.get("release_pos_x", 0) or 0),
        "release_pos_z":          float(row.get("release_pos_z", 0) or 0),
        "release_extension":      float(row.get("release_extension", 0) or 0),
        "plate_x":                plate_x,
        "plate_z":                plate_z,
        "pitch_type_enc":         ptm.get(pitch_type, ptm.get("OTHER", 17)),
        "era_flag_enc":           _era_flag_enc(row.get("game_date", "2025-04-01")),
        "balls":                  balls,
        "strikes":                strikes,
        "pitch_number":           int(row.get("pitch_number", 1) or 1),
        "prev_pitch_type_enc":    ptm.get(prev_pitch_type, ptm.get("OTHER", 17)),
        "prev_pitch_speed":       float(row.get("release_speed_prev", 0) or 0),
        "prev_pitch_result_enc":  prm.get(prev_result, prm.get("OTHER", 7)),
        "on_1b_flag":             int(bool(row.get("on_1b"))),
        "on_2b_flag":             int(bool(row.get("on_2b"))),
        "on_3b_flag":             int(bool(row.get("on_3b"))),
        "inning":                 int(row.get("inning", 1) or 1),
        "score_diff":             int(row.get("bat_score", 0) or 0) - int(row.get("fld_score", 0) or 0),
        "times_through_order":    int(row.get("times_through_order", 1) or 1),
        "hitter_stand_enc":       0 if str(row.get("stand", "R") or "R") == "L" else 1,
        "p_throws_enc":           0 if str(row.get("p_throws", "R") or "R") == "L" else 1,
        "hitter_swing_rate_this_zone":   swing_z,
        "hitter_whiff_rate_this_zone":   whiff_z,
        "hitter_swing_rate_this_family": swing_fam,
        "hitter_whiff_rate_this_family": whiff_fam,
        "hitter_swing_rate_this_count":  swing_cnt,
    }

    return np.array([assembled[col] for col in feature_cols], dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def _predict_df(df: pd.DataFrame, hitter_features: dict, pipeline: dict) -> pd.DataFrame:
    feature_cols = pipeline["feature_cols"]
    encodings    = pipeline["encodings"]
    swing_model, swing_cal     = pipeline["swing"]
    contact_model, contact_cal = pipeline["contact"]
    hc_model, hc_cal           = pipeline["hard_contact"]

    p_swings, p_contacts, p_hcs = [], [], []
    p_swing_raws, p_hc_raws     = [], []

    for _, row in df.iterrows():
        try:
            feat = _build_row(row, hitter_features, feature_cols, encodings)
        except Exception:
            p_swings.append(np.nan); p_contacts.append(np.nan); p_hcs.append(np.nan)
            p_swing_raws.append(np.nan); p_hc_raws.append(np.nan)
            continue

        raw_sw = float(swing_model.predict_proba(feat)[0, 1])
        raw_co = float(contact_model.predict_proba(feat)[0, 1])
        raw_hc = float(hc_model.predict_proba(feat)[0, 1])

        p_swing_raws.append(raw_sw)
        p_hc_raws.append(raw_hc)

        p_swings.append(  float(np.clip(swing_cal.predict([raw_sw])[0],   0, 0.97)) if swing_cal   else raw_sw)
        p_contacts.append(float(np.clip(contact_cal.predict([raw_co])[0], 0, 0.97)) if contact_cal else raw_co)
        p_hcs.append(     float(np.clip(hc_cal.predict([raw_hc])[0],      0, 0.97)) if hc_cal      else raw_hc)

    out = df.copy()
    out["p_swing"]      = p_swings
    out["p_contact_sw"] = p_contacts
    out["p_hard_con"]   = p_hcs
    out["p_swing_raw"]  = p_swing_raws
    out["p_hard_raw"]   = p_hc_raws
    out["p_composite"]  = out["p_swing"] * out["p_contact_sw"] * out["p_hard_con"]
    return out


# ---------------------------------------------------------------------------
# Per-hitter metrics
# ---------------------------------------------------------------------------

def _outcome_gradient(df: pd.DataFrame):
    df2 = df.copy()
    df2["group"] = df2["description"].map(OUTCOME_GROUPS)
    df2 = df2.dropna(subset=["group", "p_composite"])
    means = df2.groupby("group")["p_composite"].mean().to_dict()
    vals  = [means[g] for g in GRADIENT_ORDER if g in means]
    swaps = sum(1 for a, b in zip(vals, vals[1:]) if a >= b)
    return means, swaps <= 1


def _ff_in_zone_mean(df: pd.DataFrame) -> float:
    mask = (
        (df["pitch_type"] == "FF") &
        (df["plate_x"].between(-0.85, 0.85)) &
        (df["plate_z"].between(1.5, 3.5))
    )
    sub = df[mask]["p_composite"].dropna()
    return float(sub.mean()) if len(sub) > 0 else float("nan")


def _pitch_type_breakdown(df: pd.DataFrame) -> dict:
    return {
        str(pt): float(grp["p_composite"].mean())
        for pt, grp in df.groupby("pitch_type")
        if len(grp) >= 20
    }


def _saturation_check(df: pd.DataFrame):
    sat_mask = df["p_hard_con"] >= 0.97
    sat_frac = float(sat_mask.mean())
    warn     = sat_frac > 0.20
    sample   = df[sat_mask][["pitch_type", "plate_x", "plate_z", "balls", "strikes",
                              "p_hard_raw", "p_hard_con"]].head(5)
    return sat_frac, warn, sample


def _breaking_ball_mean(pt_breakdown: dict) -> float:
    vals = [v for k, v in pt_breakdown.items() if k in BREAKING_TYPES]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Step 7 — Calibrator diagnosis
# ---------------------------------------------------------------------------

def _diagnose_calibrator(pipeline: dict):
    print("\n" + "=" * 70)
    print("STEP 7 — Hard-contact calibrator diagnosis")
    print("=" * 70)
    _, hc_cal = pipeline["hard_contact"]
    if hc_cal is None:
        print("  No hard_contact calibrator found.")
        return

    xs = getattr(hc_cal, "X_thresholds_", getattr(hc_cal, "X_", None))
    ys = getattr(hc_cal, "y_thresholds_", getattr(hc_cal, "y_", None))

    if xs is None:
        print("  Could not access calibrator thresholds.")
        return

    print(f"  Isotonic steps: {len(xs)}")
    ceil_mask = np.array(ys) >= 0.97
    if ceil_mask.any():
        ceil_xs = np.array(xs)[ceil_mask]
        raw_min, raw_max = ceil_xs.min(), ceil_xs.max()
        print(f"  Raw P(hard) range → calibrated ≥ 0.97: [{raw_min:.4f}, {raw_max:.4f}]")
        if raw_min < 0.80:
            print(f"  *** FLAG: raw values as low as {raw_min:.3f} ceiling-pin to 1.000.")
            print("      Calibrator needs retraining with more hard-contact-positive")
            print("      examples at the top end. Saturation is pathological.")
        else:
            print("  OK: ceiling only applies to raw values >= 0.80.")
    else:
        print("  No ceiling saturation detected in calibrator mapping.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {CSV_DIR}")
        return

    print("=" * 70)
    print("PITCH DUEL — Hitter Validation Report")
    print(f"Model: {MODEL_DIR}  |  Profiles: {PROFILE_DIR}")
    print(f"CSV directory: {CSV_DIR}  ({len(csv_files)} file(s))")
    print("=" * 70)

    # Step 1 — Resolve IDs from CSVs
    print("\nSTEP 1 — Resolving player IDs from CSV batter column:")
    hitters = _resolve_csv_hitters(CSV_DIR)
    if not hitters:
        print("ERROR: No hitters resolved.")
        return

    # Build name → player_id map for checks
    name_to_pid = {name: pid for _, pid, name in hitters}

    print("\nLoading models...", end=" ", flush=True)
    pipeline = _load_pipeline()
    print("done.")

    results = {}  # player_id → result dict

    for csv_path, player_id, player_name in hitters:
        df = pd.read_csv(csv_path, low_memory=False)

        print(f"\n{'─' * 70}")
        print(f"HITTER: {player_name}  (player_id={player_id}, {len(df)} pitches)")

        # Step 2 — Load or build profile
        from src.hitters.profiles import profile_to_feature_dict
        profile, used_fallback, no_mlb_data, source = _load_or_build_profile(player_id, player_name)

        if no_mlb_data:
            print(f"  Profile: NO MLB DATA — league average only")
        elif used_fallback:
            print(f"  Profile: no profile found, league average fallback")
        elif source == "built":
            print(f"  Profile: built from parquet and saved → {PROFILE_DIR}/{player_id}.json")
        else:
            print(f"  Profile loaded: {profile.player_name}")

        print(f"  swing_rate={profile.swing_rate:.3f}  chase_rate={profile.chase_rate:.3f}  "
              f"contact_rate={profile.contact_rate:.3f}  hard_hit_rate={profile.hard_hit_rate:.3f}  "
              f"whiff_rate={profile.whiff_rate:.3f}")
        print(f"  sample_size={profile.sample_size:,}  is_thin_sample={profile.is_thin_sample}  no_mlb_data={no_mlb_data}")

        hitter_features = profile_to_feature_dict(profile)

        # Predictions
        print(f"  Running predictions...", end=" ", flush=True)
        pred_df = _predict_df(df, hitter_features, pipeline)
        valid_n = pred_df["p_composite"].notna().sum()
        print(f"done ({valid_n}/{len(df)} rows valid)")

        # Step 3a — Outcome gradient
        grad_means, grad_pass = _outcome_gradient(pred_df)
        print(f"\n  Outcome gradient (mean composite P(hard)):")
        for g in GRADIENT_ORDER:
            if g in grad_means:
                print(f"    {g:<20} {grad_means[g]:.4f}")
        print(f"  Gradient check: {'PASS' if grad_pass else 'FAIL'}")

        # Step 3b — FF-in-zone baseline
        ff_zone = _ff_in_zone_mean(pred_df)
        if not np.isnan(ff_zone):
            print(f"\n  FF in-zone mean P(hard): {ff_zone:.4f}")
        else:
            print(f"\n  FF in-zone: no qualifying pitches")

        # Step 3c — Pitch-type breakdown
        pt_break = _pitch_type_breakdown(pred_df)
        print(f"\n  Pitch-type breakdown (≥20 pitches):")
        for pt, val in sorted(pt_break.items()):
            print(f"    {pt:<6} {val:.4f}")

        # Step 3d — Saturation
        sat_frac, sat_warn, sat_sample = _saturation_check(pred_df)
        print(f"\n  P(hard|contact) saturation (>= 0.97): {sat_frac:.1%}")
        if sat_warn:
            print("  *** SATURATION WARNING: > 20% of pitches hitting the ceiling cap.")
            print("      Raw pre-calibration P(hard) for 5 saturated pitches:")
            print(sat_sample.to_string(index=False))

        results[player_id] = {
            "name":        player_name,
            "profile":     profile,
            "ff_zone":     ff_zone,
            "grad_pass":   grad_pass,
            "sat_frac":    sat_frac,
            "sat_warn":    sat_warn,
            "pt_break":    pt_break,
            "comp_mean":   float(pred_df["p_composite"].mean()),
            "fallback":    used_fallback,
            "no_mlb_data": no_mlb_data,
        }

    # Step 4 — Cross-hitter comparison table
    print(f"\n{'=' * 70}")
    print("STEP 4 — Cross-hitter comparison table")
    print("=" * 70)
    header = (f"{'Hitter':<22} | {'Profile':>7} | {'Data':>10} | "
              f"{'FF-zone':>8} | {'Gradient':>8} | {'Sat%':>6} | {'Comp mean':>9}")
    print(header)
    print("─" * len(header))

    for pid, r in results.items():
        p     = r["profile"]
        size  = f"{p.sample_size:,}"
        if r["no_mlb_data"]:
            data_label = "NO MLB DATA"
        elif p.is_thin_sample:
            data_label = "thin-blend"
        else:
            data_label = "full"
        ffz  = f"{r['ff_zone']:.3f}" if not np.isnan(r["ff_zone"]) else "  N/A"
        grad = "PASS" if r["grad_pass"] else "FAIL"
        sat  = f"{r['sat_frac']:.1%}"
        comp = f"{r['comp_mean']:.3f}"
        print(f"{r['name']:<22} | {size:>7} | {data_label:>10} | "
              f"{ffz:>8} | {grad:>8} | {sat:>6} | {comp:>9}")

    # Step 5 — PASS/FAIL checks
    print(f"\n{'=' * 70}")
    print("STEP 5 — Validation checks")
    print("=" * 70)

    def _ff(name: str) -> float:
        pid = name_to_pid.get(name)
        if pid is None:
            return float("nan")
        return results.get(pid, {}).get("ff_zone", float("nan"))

    def _pt(name: str) -> dict:
        pid = name_to_pid.get(name)
        if pid is None:
            return {}
        return results.get(pid, {}).get("pt_break", {})

    passes = 0

    # CHECK 1 — Elite differentiation
    ohtani_ff  = _ff("Shohei Ohtani")
    judge_ff   = _ff("Aaron Judge")
    lindor_ff  = _ff("Francisco Lindor")
    ramirez_ff = _ff("José Ramírez")
    c1_vals    = [ohtani_ff, judge_ff, lindor_ff, ramirez_ff]
    if any(np.isnan(v) for v in c1_vals):
        c1 = False
        c1_note = "(one or more players not found in CSVs)"
    else:
        c1 = (ohtani_ff > lindor_ff and ohtani_ff > ramirez_ff and
              judge_ff  > lindor_ff and judge_ff  > ramirez_ff)
        c1_note = ""
    passes += c1
    print(f"\nCHECK 1 — Elite differentiation: {'PASS' if c1 else 'FAIL'}  {c1_note}")
    print(f"  Ohtani FF-zone={ohtani_ff:.3f}  Judge FF-zone={judge_ff:.3f}")
    print(f"  Lindor FF-zone={lindor_ff:.3f}  Ramírez FF-zone={ramirez_ff:.3f}")

    # CHECK 2 — Profile-driven spread
    all_ff = [r["ff_zone"] for r in results.values() if not np.isnan(r["ff_zone"])]
    spread = max(all_ff) - min(all_ff) if len(all_ff) >= 2 else 0.0
    c2 = spread >= 0.040
    passes += c2
    print(f"\nCHECK 2 — Profile-driven spread: {'PASS' if c2 else 'FAIL'}")
    print(f"  Spread = {spread:.4f}  (min={min(all_ff):.3f}, max={max(all_ff):.3f})")

    # CHECK 3 — Thin-sample honesty (rewritten)
    # Thin = is_thin_sample=True OR no_mlb_data=True
    # PASS if: all thin hitters' FF-zone values are within 0.030 of each other
    #          AND all thin hitters' FF-zone < lowest non-thin hitter's FF-zone
    thin_ff   = [r["ff_zone"] for r in results.values()
                 if (r["profile"].is_thin_sample or r["no_mlb_data"]) and not np.isnan(r["ff_zone"])]
    nonthin_ff = [r["ff_zone"] for r in results.values()
                  if not r["profile"].is_thin_sample and not r["no_mlb_data"] and not np.isnan(r["ff_zone"])]

    if len(thin_ff) < 2:
        c3 = False
        c3_note = "(fewer than 2 thin-sample hitters with FF-zone data)"
    elif not nonthin_ff:
        c3 = False
        c3_note = "(no non-thin hitters to compare against)"
    else:
        thin_spread   = max(thin_ff) - min(thin_ff)
        lowest_elite  = min(nonthin_ff)
        within_each   = thin_spread <= 0.030
        below_elite   = all(v < lowest_elite for v in thin_ff)
        c3 = within_each and below_elite
        c3_note = (f"thin spread={thin_spread:.4f} (≤0.030?{'Y' if within_each else 'N'})  "
                   f"lowest elite={lowest_elite:.3f}  "
                   f"all thin below elite?{'Y' if below_elite else 'N'}")
    passes += c3
    print(f"\nCHECK 3 — Thin-sample honesty: {'PASS' if c3 else 'FAIL'}")
    thin_results = [(r["name"], r["ff_zone"]) for r in results.values()
                    if r["profile"].is_thin_sample or r["no_mlb_data"]]
    for tname, tff in thin_results:
        label = "NO MLB DATA" if results[name_to_pid[tname]]["no_mlb_data"] else "thin"
        ffstr = f"{tff:.3f}" if not np.isnan(tff) else "N/A"
        print(f"  {tname} ({label}): FF-zone={ffstr}")
    print(f"  {c3_note}")

    # CHECK 4 — Gradient consistency
    grad_passes = [r["name"] for r in results.values() if r["grad_pass"]]
    grad_fails  = [r["name"] for r in results.values() if not r["grad_pass"]]
    c4 = len(grad_passes) >= 5
    passes += c4
    print(f"\nCHECK 4 — Gradient consistency: {'PASS' if c4 else 'FAIL'}")
    print(f"  {len(grad_passes)}/{len(results)} hitters pass gradient check.")
    if grad_fails:
        print(f"  Failed: {', '.join(grad_fails)}")

    # CHECK 5 — Judge vs Ramírez pitch-type variance
    judge_pt    = _pt("Aaron Judge")
    ramirez_pt  = _pt("José Ramírez")
    judge_ff2   = judge_pt.get("FF", float("nan"))
    ramirez_ff2 = ramirez_pt.get("FF", float("nan"))
    judge_bb    = _breaking_ball_mean(judge_pt)
    ramirez_bb  = _breaking_ball_mean(ramirez_pt)
    judge_gap   = judge_ff2 - judge_bb   if not (np.isnan(judge_ff2)   or np.isnan(judge_bb))   else float("nan")
    ramirez_gap = ramirez_ff2 - ramirez_bb if not (np.isnan(ramirez_ff2) or np.isnan(ramirez_bb)) else float("nan")
    c5 = not np.isnan(judge_gap) and not np.isnan(ramirez_gap) and judge_gap > ramirez_gap
    passes += c5
    print(f"\nCHECK 5 — Judge vs Ramírez pitch-type variance: {'PASS' if c5 else 'FAIL'}")
    print(f"  Judge   FF P(hard)={judge_ff2:.3f}  breaking P(hard)={judge_bb:.3f}  gap={judge_gap:.4f}")
    print(f"  Ramírez FF P(hard)={ramirez_ff2:.3f}  breaking P(hard)={ramirez_bb:.3f}  gap={ramirez_gap:.4f}")

    # Step 6 — Final verdict
    print(f"\n{'=' * 70}")
    print(f"STEP 6 — Final verdict: {passes}/5 checks passed")
    if passes == 5:
        print("  Model meaningfully differentiates hitters. Ready for coach demo.")
    elif passes >= 3:
        print("  Partial differentiation. Note which checks failed.")
    else:
        print("  Model does not sufficiently differentiate hitters. Investigate.")

    # Step 7 — Calibrator diagnosis
    _diagnose_calibrator(pipeline)
    print()


if __name__ == "__main__":
    main()
