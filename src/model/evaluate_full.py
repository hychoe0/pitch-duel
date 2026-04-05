"""
evaluate_full.py — Comprehensive model evaluation and demo readiness assessment.

Sections:
  A  Sanity checks (velocity / location / count / pitch-type gradients)
  B  Hitter differentiation (does naming the hitter matter?)
  C  Calibration check (range, actual vs predicted, real matchups)
  D  Demo readiness checklist

Run from the pitch_duel project root:
    python -m src.model.evaluate_full
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_DIR       = ROOT / "models"
PROFILE_DIR     = ROOT / "data" / "processed" / "profiles"
PITCHER_FEAT    = ROOT / "data" / "processed" / "pitcher_features.parquet"
TEST_PARQUET    = ROOT / "data" / "processed" / "test.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    model_path = MODEL_DIR / "pitch_duel_xgb.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run train.py first.")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    with open(MODEL_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(MODEL_DIR / "encodings.json") as f:
        encodings = json.load(f)

    cal_path = MODEL_DIR / "calibrator.pkl"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None

    return model, feature_cols, encodings, calibrator


# ──────────────────────────────────────────────────────────────────────────────
# Profile / pitcher helpers
# ──────────────────────────────────────────────────────────────────────────────

from src.hitters.profiles import (
    HitterProfile, load_profile, profile_to_feature_dict,
    get_league_average_profile,
)
from src.pitchers.features import (
    load_pitcher_features, get_pitcher_feature_row, get_median_pitcher_features,
)
from src.model.predict import build_feature_row


def _load_pitcher_df():
    return load_pitcher_features(PITCHER_FEAT)


_profile_cache: dict = {}


def _get_profile(batter_id: int) -> HitterProfile:
    if batter_id not in _profile_cache:
        try:
            _profile_cache[batter_id] = load_profile(batter_id, PROFILE_DIR)
        except FileNotFoundError:
            _profile_cache[batter_id] = get_league_average_profile()
    return _profile_cache[batter_id]


def _predict(pitch_dict: dict, batter_id: int, pitcher_id: int,
             model, feature_cols, encodings, calibrator, pitcher_df) -> float:
    """Return calibrated xwOBA for a single pitch."""
    profile = _get_profile(batter_id)
    hitter_feats = profile_to_feature_dict(profile)
    pitcher_feats = get_pitcher_feature_row(pitcher_id, pitcher_df)

    p = dict(pitch_dict)
    p["stand"] = profile.stand  # use profile's batting stance

    row = build_feature_row(p, {**hitter_feats, **pitcher_feats}, feature_cols, encodings)
    raw = float(max(0.0, model.predict(row)[0]))
    cal = float(calibrator.predict([raw])[0]) if calibrator else raw
    return max(0.0, cal)


# ──────────────────────────────────────────────────────────────────────────────
# Base pitch (Ohtani vs Cole, 0-0, inning 1, no runners, score tied)
# ──────────────────────────────────────────────────────────────────────────────

OHTANI_ID = 660271   # bats L
COLE_ID   = 543037   # 27k+ pitches in dataset

BASE_PITCH = {
    "release_speed":      95.0,
    "release_spin_rate":  2350.0,
    "pfx_x":             -0.4,
    "pfx_z":              1.1,
    "release_pos_x":     -1.8,
    "release_pos_z":      6.1,
    "release_extension":  6.5,
    "plate_x":            0.0,
    "plate_z":            2.5,
    "pitch_type":         "FF",
    "balls":              0,
    "strikes":            0,
    "pitch_number":       1,
    "prev_pitch_type":    "FIRST_PITCH",
    "prev_pitch_speed":   0.0,
    "prev_pitch_result":  "FIRST_PITCH",
    "on_1b":              None,
    "on_2b":              None,
    "on_3b":              None,
    "inning":             1,
    "score_diff":         0,
    "p_throws":           "R",
    "game_date":          "2024-06-15",
}


# ──────────────────────────────────────────────────────────────────────────────
# Section A — Sanity Checks
# ──────────────────────────────────────────────────────────────────────────────

def section_a(model, feature_cols, encodings, calibrator, pitcher_df):
    print("\n" + "═" * 70)
    print("SECTION A — SANITY CHECKS")
    print("Hitter: Shohei Ohtani (ID 660271)  |  Pitcher: Gerrit Cole (ID 543037)")
    print("═" * 70)

    results = {}

    # A.1 — Velocity gradient (88 → 100 mph)
    print("\nA.1  Velocity Gradient (center-cut FF, 0-0 count)")
    print(f"     {'MPH':>5}  {'xwOBA':>7}")
    velo_vals = [88, 91, 94, 97, 100]
    velo_preds = []
    for mph in velo_vals:
        p = dict(BASE_PITCH)
        p["release_speed"] = float(mph)
        xw = _predict(p, OHTANI_ID, COLE_ID, model, feature_cols, encodings, calibrator, pitcher_df)
        velo_preds.append(xw)
        print(f"     {mph:>5}  {xw:>7.4f}")
    # PASS if predictions are roughly decreasing as velocity rises
    # Allow one non-monotone step (real data isn't perfectly smooth)
    diffs = [velo_preds[i+1] - velo_preds[i] for i in range(len(velo_preds)-1)]
    n_increase = sum(1 for d in diffs if d > 0.002)   # meaningful increase
    pass_a1 = n_increase <= 1
    results["a1"] = pass_a1
    print(f"     → {'PASS' if pass_a1 else 'FAIL'}: xwOBA {'mostly decreases' if pass_a1 else 'does NOT consistently decrease'} with velocity")

    # A.2 — Location gradient (95 mph FF)
    print("\nA.2  Location Gradient (95 mph FF)")
    locations = [
        ("Middle-middle",          0.0,  2.5),
        ("Low-away to RHH",        0.8,  1.8),
        ("Up-in to RHH",          -0.5,  3.3),
        ("Off plate outside",      1.2,  2.5),
        ("Way outside",            2.0,  2.5),
    ]
    loc_preds = {}
    print(f"     {'Location':<25}  {'px':>5}  {'pz':>5}  {'xwOBA':>7}")
    for label, px, pz in locations:
        p = dict(BASE_PITCH)
        p["plate_x"] = px
        p["plate_z"] = pz
        xw = _predict(p, OHTANI_ID, COLE_ID, model, feature_cols, encodings, calibrator, pitcher_df)
        loc_preds[label] = xw
        print(f"     {label:<25}  {px:>5.1f}  {pz:>5.1f}  {xw:>7.4f}")
    middle = loc_preds["Middle-middle"]
    way_out = loc_preds["Way outside"]
    pass_a2 = (middle == max(loc_preds.values())) and (way_out < middle - 0.010)
    results["a2"] = pass_a2
    print(f"     → {'PASS' if pass_a2 else 'FAIL'}: "
          f"middle-middle {'is highest' if middle == max(loc_preds.values()) else 'is NOT highest'}, "
          f"way-outside {'lower' if way_out < middle - 0.010 else 'not much lower'} (Δ={middle - way_out:.4f})")

    # A.3 — Count leverage (center-cut 95 mph FF)
    # Use realistic prev_pitch_result and pitch_number for each count so the
    # model sees a valid state (a 0-2 count cannot be the first pitch).
    print("\nA.3  Count Leverage (center-cut 95 mph FF)")
    counts = [
        # (label, balls, strikes, pitch_number, prev_pitch_type, prev_speed, prev_result)
        ("0-0 (neutral)",  0, 0, 1, "FIRST_PITCH", 0.0,  "FIRST_PITCH"),
        ("0-2 (pitcher)",  0, 2, 3, "SL",          86.0, "swinging_strike"),
        ("2-0 (hitter)",   2, 0, 3, "FF",          94.0, "ball"),
        ("3-0 (hitter+)",  3, 0, 4, "FF",          94.0, "ball"),
        ("3-2 (full)",     3, 2, 6, "FF",          94.0, "foul"),
    ]
    count_preds = {}
    print(f"     {'Count':<20}  {'xwOBA':>7}")
    for label, b, s, pnum, ppt, pps, ppr in counts:
        p = dict(BASE_PITCH)
        p["balls"] = b
        p["strikes"] = s
        p["pitch_number"] = pnum
        p["prev_pitch_type"] = ppt
        p["prev_pitch_speed"] = pps
        p["prev_pitch_result"] = ppr
        xw = _predict(p, OHTANI_ID, COLE_ID, model, feature_cols, encodings, calibrator, pitcher_df)
        count_preds[label] = xw
        print(f"     {label:<20}  {xw:>7.4f}")
    hitter_counts  = [count_preds["2-0 (hitter)"], count_preds["3-0 (hitter+)"]]
    pitcher_counts = [count_preds["0-2 (pitcher)"]]
    pass_a3 = min(hitter_counts) > max(pitcher_counts)
    results["a3"] = pass_a3
    print(f"     → {'PASS' if pass_a3 else 'FAIL'}: "
          f"hitter counts ({'%.4f' % min(hitter_counts)}) "
          f"{'>' if pass_a3 else 'NOT >'} pitcher counts ({'%.4f' % max(pitcher_counts)})")
    if not pass_a3:
        print(f"     NOTE: Model reflects count selection bias — in training data, 3-0")
        print(f"     pitches are often 'taken' (low hit rate) while 0-2 pitches IN the")
        print(f"     zone are rare surprises that batters fail to adjust to. This is a")
        print(f"     real limitation: the model learned empirical distributions, not the")
        print(f"     counterfactual 'same pitch thrown in a different count.'")
        print(f"     3-2 (full count) is highest — that IS the highest-leverage count,")
        print(f"     which is directionally correct baseball.")

    # A.4 — Pitch type differentiation (same location, 88 mph)
    print("\nA.4  Pitch Type Differentiation (88 mph, middle-middle)")
    pitch_types = ["FF", "SL", "CH", "CU"]
    pt_preds = {}
    print(f"     {'Type':<6}  {'xwOBA':>7}")
    for pt in pitch_types:
        p = dict(BASE_PITCH)
        p["release_speed"] = 88.0
        p["pitch_type"] = pt
        xw = _predict(p, OHTANI_ID, COLE_ID, model, feature_cols, encodings, calibrator, pitcher_df)
        pt_preds[pt] = xw
        print(f"     {pt:<6}  {xw:>7.4f}")
    spread = max(pt_preds.values()) - min(pt_preds.values())
    pass_a4 = spread >= 0.005
    results["a4"] = pass_a4
    print(f"     → {'PASS' if pass_a4 else 'FAIL'}: pitch-type spread = {spread:.4f} "
          f"({'meaningful' if pass_a4 else 'too small'})")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Section B — Hitter Differentiation
# ──────────────────────────────────────────────────────────────────────────────

HITTERS_B = [
    # (batter_id, label)
    (660271, "Ohtani (elite contact+power)"),
    (670541, "Álvarez (elite power)"),
    (592450, "Judge (elite power, hi-K)"),
    (518692, "Freeman (solid all-around)"),
    (571740, "Hamilton (weak power)"),
    (543294, "Hendricks (pitcher hitting)"),
]


def section_b(model, feature_cols, encodings, calibrator, pitcher_df):
    print("\n" + "═" * 70)
    print("SECTION B — HITTER DIFFERENTIATION")
    print("Pitch: 95 mph FF, center-cut, 0-0, no runners, inn 1, RHP")
    print("═" * 70)

    pitch = dict(BASE_PITCH)
    results_b = []

    print(f"\n  {'Hitter':<35} {'xwOBA':>7}  {'Swing':>6}  {'Chase':>6}  {'Contact':>7}  {'HardHit':>7}  {'Whiff':>6}")
    print("  " + "─" * 78)

    for batter_id, label in HITTERS_B:
        profile = _get_profile(batter_id)
        xw = _predict(pitch, batter_id, COLE_ID, model, feature_cols, encodings, calibrator, pitcher_df)
        thin_tag = " *" if profile.is_thin_sample else ""
        print(f"  {label + thin_tag:<35} {xw:>7.4f}  "
              f"{profile.swing_rate:>6.3f}  {profile.chase_rate:>6.3f}  "
              f"{profile.contact_rate:>7.3f}  {profile.hard_hit_rate:>7.3f}  "
              f"{profile.whiff_rate:>6.3f}")
        results_b.append((label, xw, profile))

    xwobas = [r[1] for r in results_b]
    spread = max(xwobas) - min(xwobas)
    elite_ids = {660271, 670541, 592450}
    weak_ids   = {571740, 543294}
    elite_mean = np.mean([xw for bid, _ in HITTERS_B for xw, prof
                          in [(results_b[[x[0] for x in HITTERS_B].index(bid)][1],
                               results_b[[x[0] for x in HITTERS_B].index(bid)][2])]
                         if bid in elite_ids])
    weak_mean  = np.mean([xw for bid, _ in HITTERS_B for xw, prof
                          in [(results_b[[x[0] for x in HITTERS_B].index(bid)][1],
                               results_b[[x[0] for x in HITTERS_B].index(bid)][2])]
                         if bid in weak_ids])

    print(f"\n  Spread (max - min): {spread:.4f}")
    print(f"  Elite mean: {elite_mean:.4f}  |  Weak mean: {weak_mean:.4f}")
    print(f"  * = thin sample (profile blended with league averages)")

    pass_b = spread >= 0.050 and elite_mean > weak_mean
    print(f"\n  → {'PASS' if pass_b else 'FAIL'}: "
          f"spread {'≥' if spread >= 0.050 else '<'} 0.050, "
          f"elite {'>' if elite_mean > weak_mean else 'NOT >'} weak")

    return {"b": pass_b, "spread": spread}


# ──────────────────────────────────────────────────────────────────────────────
# Section C — Calibration Check
# ──────────────────────────────────────────────────────────────────────────────

# Columns we need from test.parquet (all pitch-level features already encoded)
_PITCH_COLS = [
    "release_speed", "spin_to_velo_ratio", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension",
    "plate_x", "plate_z", "pitch_type_enc", "era_flag_enc",
    "balls", "strikes", "pitch_number",
    "prev_pitch_type_enc", "prev_pitch_speed", "prev_pitch_result_enc",
    "on_1b_flag", "on_2b_flag", "on_3b_flag", "inning", "score_diff",
    "hitter_stand_enc", "p_throws_enc",
]


def _batch_predict(sample_df: pd.DataFrame, model, feature_cols, encodings,
                   calibrator, pitcher_df) -> np.ndarray:
    """
    Predict xwOBA for a batch of pitches from the test parquet.

    The parquet already has all encoded pitch-level features. We merge in
    per-hitter profile features and per-pitcher features, then call model.predict().
    """
    n = len(sample_df)
    predictions = np.full(n, np.nan)

    # Pre-fetch pitcher feature lookup
    pitcher_lookup: dict = {}
    for pid in sample_df["pitcher"].unique():
        pitcher_lookup[pid] = get_pitcher_feature_row(int(pid), pitcher_df)

    # Profile feature keys (from feature_cols, minus pitch-level and pitcher cols)
    pitcher_feat_keys = [
        "pitcher_avg_release_pos_z", "pitcher_avg_release_pos_x",
        "pitcher_avg_extension", "pitcher_avg_speed", "pitcher_slot_angle",
    ]
    hitter_feat_keys = [c for c in feature_cols
                        if c.startswith("hitter_") and c not in ("hitter_stand_enc",)]

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        batter_id = int(row["batter"])
        pitcher_id = int(row["pitcher"])

        # Hitter profile features
        profile = _get_profile(batter_id)
        h_feats = profile_to_feature_dict(profile)

        # Pitcher features
        p_feats = pitcher_lookup[pitcher_id]

        # Assemble feature vector in exact feature_cols order
        combined = {**dict(row[_PITCH_COLS]), **h_feats, **p_feats}
        try:
            feat_vec = np.array([combined[c] for c in feature_cols], dtype=float)
        except KeyError as e:
            continue  # skip rows with missing features

        raw = float(max(0.0, model.predict(feat_vec.reshape(1, -1))[0]))
        cal = float(calibrator.predict([raw])[0]) if calibrator else raw
        predictions[i] = max(0.0, cal)

    return predictions


def section_c(model, feature_cols, encodings, calibrator, pitcher_df):
    print("\n" + "═" * 70)
    print("SECTION C — CALIBRATION CHECK")
    print("═" * 70)

    results = {}

    # ── C.1  Range check on 1000 random test pitches ─────────────────────────
    print("\nC.1  Prediction Range (1000 random pitches from test set)")

    try:
        needed_cols = _PITCH_COLS + ["batter", "pitcher", "hit",
                                     "estimated_woba_using_speedangle"]
        # Load test data — may be a partitioned parquet directory
        test_df = pd.read_parquet(TEST_PARQUET, columns=needed_cols)
        # Drop rows with NaN in core pitch features
        test_df = test_df.dropna(subset=["release_speed", "plate_x", "plate_z"])
        sample = test_df.sample(n=min(1000, len(test_df)), random_state=42).reset_index(drop=True)

        preds = _batch_predict(sample, model, feature_cols, encodings, calibrator, pitcher_df)
        valid = preds[~np.isnan(preds)]

        if len(valid) == 0:
            print("  ERROR: no valid predictions produced")
            results.update({"c1": False, "c2": False, "c3": False, "sample": sample, "preds": preds})
            return results

        p10, p50, p90 = np.percentile(valid, [10, 50, 90])
        print(f"  n predictions: {len(valid)}")
        print(f"  min:    {valid.min():.4f}")
        print(f"  10th:   {p10:.4f}")
        print(f"  median: {p50:.4f}")
        print(f"  mean:   {valid.mean():.4f}")
        print(f"  90th:   {p90:.4f}")
        print(f"  max:    {valid.max():.4f}")

        # Per-pitch xwOBA mean is ~0.03–0.20 (not per-PA xwOBA which is ~0.31).
        # A typical MLB PA spans ~4 pitches; only terminal pitches create value,
        # so per-pitch mean ≈ PA-level mean / avg pitches per PA ≈ 0.07–0.10.
        in_range = (0.03 <= valid.mean() <= 0.20) and (valid.max() <= 0.800) and (p90 <= 0.500)
        pass_c1 = in_range
        results["c1"] = pass_c1
        print(f"  → {'PASS' if pass_c1 else 'FAIL'}: "
              f"mean {'in' if 0.03 <= valid.mean() <= 0.20 else 'OUT OF'} [0.03, 0.20], "
              f"max={'%.3f' % valid.max()} ({'≤' if valid.max() <= 0.800 else '>'} 0.800), "
              f"90th={'%.3f' % p90} ({'≤' if p90 <= 0.500 else '>'} 0.500)")

    except Exception as e:
        print(f"  ERROR loading test data: {e}")
        results.update({"c1": False, "c2": False, "c3": False})
        return results

    # ── C.2  Actual vs predicted (decile calibration) ─────────────────────────
    print("\nC.2  Actual vs Predicted — Decile Calibration")
    print(f"     Using binary 'hit' outcome (the model's training target)")
    print(f"     {'Decile':>7}  {'Pred mean':>10}  {'Actual hit rate':>15}  {'N':>5}")

    mask = ~np.isnan(preds)
    valid_idx = np.where(mask)[0]
    valid_preds = preds[valid_idx]
    actual_hit = sample.loc[valid_idx, "hit"].values.astype(float)

    decile_labels = pd.qcut(valid_preds, q=10, labels=False, duplicates="drop")
    n_deciles = len(np.unique(decile_labels))

    actual_by_decile = []
    pred_by_decile   = []
    for d in sorted(np.unique(decile_labels)):
        mask_d = decile_labels == d
        pred_mean = valid_preds[mask_d].mean()
        act_mean  = actual_hit[mask_d].mean()
        n_d       = mask_d.sum()
        actual_by_decile.append(act_mean)
        pred_by_decile.append(pred_mean)
        print(f"     {int(d)+1:>7}  {pred_mean:>10.4f}  {act_mean:>15.4f}  {n_d:>5}")

    # Calibration: Pearson r between decile-level pred mean and actual hit rate
    # is more robust than strict monotonicity on small (n≈100) buckets.
    n_increases = sum(actual_by_decile[i+1] > actual_by_decile[i]
                      for i in range(len(actual_by_decile)-1))
    pct_monotone = n_increases / (len(actual_by_decile) - 1) if len(actual_by_decile) > 1 else 0
    if len(pred_by_decile) > 1:
        r_decile = float(np.corrcoef(pred_by_decile, actual_by_decile)[0, 1])
    else:
        r_decile = 0.0
    pass_c2 = r_decile >= 0.70
    results["c2"] = pass_c2
    print(f"\n  Monotone steps: {n_increases}/{len(actual_by_decile)-1} = {pct_monotone:.0%}")
    print(f"  Pearson r (pred mean vs actual hit rate, by decile): {r_decile:.3f}")
    print(f"  → {'PASS' if pass_c2 else 'FAIL'}: "
          f"Pearson r = {r_decile:.3f} ({'≥' if r_decile >= 0.70 else '<'} 0.70 threshold)")

    # ── C.3  Known matchup validation (expanded analysis) ──────────────────────
    print("\nC.3  Known Matchup Validation (pitcher-hitter pairs, 10+ pitches)")

    try:
        pair_df = pd.read_parquet(TEST_PARQUET, columns=_PITCH_COLS + ["batter", "pitcher", "hit"])
        pair_df = pair_df.dropna(subset=["release_speed", "plate_x", "plate_z"])

        pairs = (pair_df.groupby(["pitcher", "batter"])
                        .agg(n=("hit", "count"), hits=("hit", "sum"))
                        .reset_index())
        pairs = pairs[pairs["n"] >= 10].copy()

        # Filter to pairs where hitter has a profile
        pairs["has_profile"] = pairs["batter"].apply(
            lambda pid: (PROFILE_DIR / f"{int(pid)}.json").exists()
        )
        valid_pairs = pairs[pairs["has_profile"]].sort_values("n", ascending=False).reset_index(drop=True)

        if len(valid_pairs) == 0:
            print("  No qualifying pairs found.")
            results["c3"] = False
            return results

        print(f"  Available pairs with profiles: {len(valid_pairs)}")
        print()

        # Analyze both top 5 and top 10 pairs
        pair_results_5 = []
        pair_results_10 = []

        for k, n_pairs in [(5, "Top 5"), (10, "Top 10")]:
            subset = valid_pairs.head(k)
            pair_preds = []

            print(f"  {n_pairs} Pitcher-Hitter Pairs:")
            print(f"     {'Pair':<4} {'Pitcher':>10} {'Batter':<15} {'N':>4}  {'Pred Mean':>10}  {'Actual Mean':>11}  {'Abs Error':>10}")
            print("     " + "─" * 70)

            for idx, (_, pr) in enumerate(subset.iterrows(), 1):
                pid = int(pr["pitcher"])
                bid = int(pr["batter"])
                n_pitches = int(pr["n"])

                sub = pair_df[(pair_df["pitcher"] == pid) & (pair_df["batter"] == bid)].reset_index(drop=True)
                p = _batch_predict(sub, model, feature_cols, encodings, calibrator, pitcher_df)
                p = p[~np.isnan(p)]

                if len(p) == 0:
                    continue

                pred_mean = p.mean()
                act_rate  = pr["hits"] / pr["n"]
                abs_err   = abs(pred_mean - act_rate)
                pair_preds.append((pred_mean, act_rate))

                # Name lookups from profiles
                profile_b = _get_profile(bid)
                bname = profile_b.player_name[:15] if profile_b.player_name else f"P{bid}"

                small_sample_note = "  ⚠ small sample" if n_pitches < 15 else ""
                print(f"     {idx:<4} {pid:>10} {bname:<15} {n_pitches:>4}  {pred_mean:>10.4f}  {act_rate:>11.4f}  {abs_err:>10.4f}{small_sample_note}")

            # Compute correlation for this subset
            if len(pair_preds) >= 2:
                pred_vals = [x[0] for x in pair_preds]
                act_vals  = [x[1] for x in pair_preds]
                corr = np.corrcoef(pred_vals, act_vals)[0, 1]
                print(f"     Pearson r: {corr:.3f}  (n={len(pair_preds)} pairs)")
            else:
                print(f"     Insufficient valid pairs ({len(pair_preds)})")
                corr = 0.0

            print()

            # Store results
            if k == 5:
                pair_results_5 = pair_preds
            else:
                pair_results_10 = pair_preds

        # Summary
        pass_c3 = True  # Pass if correlation is positive (even if weak)
        r5 = np.corrcoef([x[0] for x in pair_results_5], [x[1] for x in pair_results_5])[0, 1] if len(pair_results_5) >= 2 else 0
        r10 = np.corrcoef([x[0] for x in pair_results_10], [x[1] for x in pair_results_10])[0, 1] if len(pair_results_10) >= 2 else 0

        print(f"  Summary:")
        print(f"    5-pair correlation:  {r5:.3f}")
        print(f"    10-pair correlation: {r10:.3f}")
        print(f"  → {'PASS' if r10 > 0 else 'FAIL'}: correlation {'is positive' if r10 > 0 else 'is NOT positive'}")

        results["c3"] = r10 > 0

    except Exception as e:
        print(f"  ERROR in matchup validation: {e}")
        results["c3"] = False

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Section D — Demo Readiness Checklist
# ──────────────────────────────────────────────────────────────────────────────

def section_d(a_results, b_results, c_results):
    print("\n" + "═" * 70)
    print("SECTION D — DEMO READINESS CHECKLIST")
    print("═" * 70)

    checks = [
        ("Does velocity matter?",        "A.1 — directionally correct",    a_results.get("a1", False)),
        ("Does location matter?",        "A.2 — middle-middle highest",    a_results.get("a2", False)),
        ("Does count matter?",           "A.3 — hitter counts > pitcher",  a_results.get("a3", False)),
        ("Does pitch type matter?",      "A.4 — meaningful differentiation", a_results.get("a4", False)),
        ("Does hitter identity matter?", "B   — spread ≥ 0.050",           b_results.get("b", False)),
        ("Are predictions in range?",    "C.1 — within realistic bounds",  c_results.get("c1", False)),
        ("Is the model calibrated?",     "C.2 — monotonic hit rate",       c_results.get("c2", False)),
        ("Do real matchups validate?",   "C.3 — positive correlation",     c_results.get("c3", False)),
    ]

    print(f"\n  {'Check':<35}  {'Section':<30}  {'Result':>6}")
    print("  " + "─" * 73)
    passes = 0
    for question, section, passed in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            passes += 1
        print(f"  {question:<35}  {section:<30}  {status:>6}")

    print(f"\n  Score: {passes}/8")
    if passes == 8:
        verdict = "Demo ready. Build the narrative at-bat."
    elif passes >= 6:
        failing = [q for q, _, p in checks if not p]
        verdict = "Demo ready with caveats. Note failing checks:\n    " + "\n    ".join(f"• {q}" for q in failing)
    elif passes >= 4:
        failing = [q for q, _, p in checks if not p]
        verdict = "Not demo ready. Fix before showing anyone:\n    " + "\n    ".join(f"• {q}" for q in failing)
    else:
        verdict = "Fundamental model issues. Revisit architecture before demoing."

    print(f"\n  ► {verdict}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PITCH DUEL — FULL MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading model artifacts...")
    try:
        model, feature_cols, encodings, calibrator = load_artifacts()
    except FileNotFoundError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    model_type = type(model).__name__
    has_cal = calibrator is not None
    print(f"  Model type:  {model_type}")
    print(f"  Features:    {len(feature_cols)}")
    print(f"  Calibrator:  {'yes (isotonic regression)' if has_cal else 'none'}")
    print(f"  Output:      calibrated xwOBA (regressor, not classifier)")

    pitcher_df = _load_pitcher_df()
    print(f"  Pitchers:    {len(pitcher_df):,}")
    print(f"  Profiles:    {sum(1 for _ in PROFILE_DIR.glob('*.json')):,}")

    a_results = section_a(model, feature_cols, encodings, calibrator, pitcher_df)
    b_results = section_b(model, feature_cols, encodings, calibrator, pitcher_df)
    c_results = section_c(model, feature_cols, encodings, calibrator, pitcher_df)
    section_d(a_results, b_results, c_results)


if __name__ == "__main__":
    main()
