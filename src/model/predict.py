"""
predict.py — Three-stage pitch outcome inference.

Runs three XGBoost classifiers in sequence:
  Stage 1: P(swing)
  Stage 2: P(contact | swing)
  Stage 3: P(hard_contact | contact)

Usage:
    from src.model.predict import predict_pitch

    result = predict_pitch(
        pitch={
            'release_speed': 95.2,
            'release_spin_rate': 2350,
            'pfx_x': -0.5,
            'pfx_z': 1.2,
            'release_pos_x': -1.8,
            'release_pos_z': 6.1,
            'release_extension': 6.5,
            'plate_x': 0.1,
            'plate_z': 2.8,
            'pitch_type': 'FF',
            'balls': 1,
            'strikes': 2,
            'pitch_number': 4,
            'prev_pitch_type': 'SL',
            'prev_pitch_speed': 88.0,
            'prev_pitch_result': 'swinging_strike',
            'on_1b': None, 'on_2b': None, 'on_3b': None,
            'inning': 7,
            'score_diff': -1,
            'p_throws': 'R',
        },
        hitter_name="Shohei Ohtani",
    )
    # {'p_swing': 0.72, 'p_contact_given_swing': 0.65, 'p_hard_given_contact': 0.48, ...}
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

MODEL_DIR    = Path("models")
PROFILE_DIR  = Path("data/processed/profiles")
PROCESSED_PATH = Path("data/processed/statcast_processed.parquet")

REQUIRED_RAW_KEYS = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension",
    "plate_x", "plate_z", "pitch_type",
    "balls", "strikes", "pitch_number",
    "prev_pitch_type", "prev_pitch_speed", "prev_pitch_result",
    "on_1b", "on_2b", "on_3b",
    "inning", "score_diff",
    "p_throws",
]

# Pitcher features are disconnected from the model pipeline.
# src/pitchers/features.py and the parquet file are preserved.
# PITCHER_FEATURES = [
#     "pitcher_avg_release_pos_z",
#     "pitcher_avg_release_pos_x",
#     "pitcher_avg_extension",
#     "pitcher_avg_speed",
#     "pitcher_slot_angle",
# ]


# ---------------------------------------------------------------------------
# Model loading (all three stages cached after first load)
# ---------------------------------------------------------------------------

_cache: dict = {}


def load_models(model_dir: Path = MODEL_DIR) -> tuple:
    """
    Load and cache all three (model, calibrator) pairs plus shared artifacts.

    Returns: (swing_pair, contact_pair, hard_contact_pair, feature_cols, encodings)
    where each pair is (XGBClassifier, IsotonicRegression | None).

    Cache is keyed by model_dir so multiple model versions can coexist.
    """
    cache_key = str(model_dir.resolve())
    if cache_key in _cache:
        c = _cache[cache_key]
        return (
            c["swing"], c["contact"], c["hard_contact"],
            c["feature_cols"], c["encodings"],
        )

    entry = {}
    stages = ["swing", "contact", "hard_contact"]
    for stage in stages:
        model_path = model_dir / f"{stage}_model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run train.py first."
            )
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        cal_path = model_dir / f"{stage}_calibrator.pkl"
        calibrator = joblib.load(cal_path) if cal_path.exists() else None
        entry[stage] = (model, calibrator)

    with open(model_dir / "feature_cols.json") as f:
        entry["feature_cols"] = json.load(f)
    with open(model_dir / "encodings.json") as f:
        entry["encodings"] = json.load(f)

    _cache[cache_key] = entry

    return (
        entry["swing"], entry["contact"], entry["hard_contact"],
        entry["feature_cols"], entry["encodings"],
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_pitch_dict(pitch: dict) -> None:
    missing = [k for k in REQUIRED_RAW_KEYS if k not in pitch]
    if missing:
        raise ValueError(f"Pitch dict missing required keys: {missing}")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _era_flag_enc(game_date) -> int:
    """Mirror the era flag logic from preprocess.py for live prediction."""
    import datetime
    if isinstance(game_date, str):
        game_date = datetime.date.fromisoformat(game_date)
    crackdown     = datetime.date(2021, 6, 21)
    crackdown_end = datetime.date(2022, 1, 1)
    pitch_clock   = datetime.date(2023, 1, 1)
    if game_date < crackdown:
        return 0
    if game_date < crackdown_end:
        return 1
    if game_date < pitch_clock:
        return 2
    return 3


def encode_pitch_dict(pitch: dict, encodings: dict) -> dict:
    """Apply integer encodings to raw pitch dict. Returns new dict with _enc keys."""
    import datetime
    ptm = encodings["PITCH_TYPE_MAP"]
    prm = encodings["PREV_RESULT_MAP"]

    encoded = dict(pitch)

    encoded["pitch_type_enc"] = ptm.get(pitch["pitch_type"], ptm.get("OTHER", 15))
    encoded["prev_pitch_type_enc"] = ptm.get(
        pitch.get("prev_pitch_type", "FIRST_PITCH"), ptm.get("OTHER", 15)
    )
    encoded["prev_pitch_result_enc"] = prm.get(
        pitch.get("prev_pitch_result", "FIRST_PITCH"), prm.get("OTHER", 7)
    )
    encoded["hitter_stand_enc"] = 0 if pitch.get("stand", "R") == "L" else 1
    encoded["p_throws_enc"]     = 0 if pitch.get("p_throws", "R") == "L" else 1
    encoded["on_1b_flag"] = int(bool(pitch.get("on_1b")))
    encoded["on_2b_flag"] = int(bool(pitch.get("on_2b")))
    encoded["on_3b_flag"] = int(bool(pitch.get("on_3b")))
    # Default to 1st time through the order; callers can pass times_through_order to override.
    encoded["times_through_order"] = int(pitch.get("times_through_order", 1))

    release_speed     = pitch["release_speed"]
    release_spin_rate = pitch.get("release_spin_rate", 0) or 0
    encoded["spin_to_velo_ratio"] = release_spin_rate / release_speed if release_speed else 0.0

    game_date = pitch.get("game_date", __import__("datetime").date.today())
    encoded["era_flag_enc"] = _era_flag_enc(game_date)

    if pitch["pitch_type"] not in ptm:
        warnings.warn(
            f"Unknown pitch_type '{pitch['pitch_type']}' — encoded as OTHER. "
            "Update PITCH_TYPE_MAP in preprocess.py if this type is common.",
            stacklevel=3,
        )

    return encoded


# ---------------------------------------------------------------------------
# Feature row assembly
# ---------------------------------------------------------------------------

def build_feature_row(
    pitch: dict,
    hitter_feature_dict: dict,
    feature_cols: list,
    encodings: dict,
) -> np.ndarray:
    """
    Assemble a (1, n_features) array in the exact order of feature_cols.
    Raises KeyError if any expected column is absent.
    """
    encoded = encode_pitch_dict(pitch, encodings)
    merged  = {**encoded, **hitter_feature_dict}

    try:
        row = [merged[col] for col in feature_cols]
    except KeyError as e:
        raise KeyError(
            f"Feature '{e.args[0]}' not found in assembled pitch+profile dict."
        ) from e

    return np.array(row, dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Statcast zone lookup
# ---------------------------------------------------------------------------

_SZ_X_MIN, _SZ_X_MAX = -0.85, 0.85
_SZ_Z_MIN, _SZ_Z_MAX = 1.5, 3.5
_CHASE_BORDER = 0.5


def _plate_to_statcast_zone(plate_x: float, plate_z: float) -> int:
    """Map plate coordinates to Statcast zone integer (1-9 in zone, 11-14 outside)."""
    x_range = _SZ_X_MAX - _SZ_X_MIN
    z_range = _SZ_Z_MAX - _SZ_Z_MIN
    x1 = _SZ_X_MIN + x_range / 3
    x2 = _SZ_X_MIN + 2 * x_range / 3
    z1 = _SZ_Z_MIN + z_range / 3
    z2 = _SZ_Z_MIN + 2 * z_range / 3

    in_x = _SZ_X_MIN <= plate_x <= _SZ_X_MAX
    in_z = _SZ_Z_MIN <= plate_z <= _SZ_Z_MAX

    if in_x and in_z:
        col = 0 if plate_x < x1 else (1 if plate_x <= x2 else 2)
        row = 0 if plate_z > z2 else (1 if plate_z >= z1 else 2)
        return 1 + row * 3 + col
    else:
        z_mid = (_SZ_Z_MIN + _SZ_Z_MAX) / 2
        high_half = plate_z >= z_mid
        left_half = plate_x < 0
        if high_half:
            return 11 if left_half else 12
        else:
            return 13 if left_half else 14


def _classify_zone(plate_x: float, plate_z: float) -> str:
    """Return 'in_zone', 'chase', or 'waste'."""
    in_x = _SZ_X_MIN <= plate_x <= _SZ_X_MAX
    in_z = _SZ_Z_MIN <= plate_z <= _SZ_Z_MAX
    if in_x and in_z:
        return "in_zone"
    chase_x = (_SZ_X_MIN - _CHASE_BORDER) <= plate_x <= (_SZ_X_MAX + _CHASE_BORDER)
    chase_z = (_SZ_Z_MIN - _CHASE_BORDER) <= plate_z <= (_SZ_Z_MAX + _CHASE_BORDER)
    if chase_x and chase_z:
        return "chase"
    return "waste"


# ---------------------------------------------------------------------------
# Contextual hitter feature resolution
# ---------------------------------------------------------------------------

def _resolve_contextual_hitter_features(
    pitch: dict,
    hitter_features: dict,
) -> dict:
    """
    Resolve 5 contextual hitter features for a single pitch at inference time.
    Mirrors add_contextual_hitter_features() from preprocess.py.
    """
    from src.hitters.profiles import PITCH_FAMILIES

    zone = _plate_to_statcast_zone(
        float(pitch.get("plate_x", 0)),
        float(pitch.get("plate_z", 2.5)),
    )
    swing_rate_z = hitter_features.get(
        f"hitter_swing_rate_z{zone}",
        hitter_features.get("hitter_swing_rate", 0.47),
    )
    whiff_rate_z = hitter_features.get(
        f"hitter_whiff_rate_z{zone}",
        hitter_features.get("hitter_whiff_rate", 0.25),
    )

    pitch_type = pitch.get("pitch_type", "FF")
    family = "other"
    for fam, types in PITCH_FAMILIES.items():
        if pitch_type in types:
            family = fam
            break
    swing_rate_fam = hitter_features.get(
        f"hitter_swing_rate_{family}",
        hitter_features.get("hitter_swing_rate", 0.47),
    )
    whiff_rate_fam = hitter_features.get(
        f"hitter_whiff_rate_{family}",
        hitter_features.get("hitter_whiff_rate", 0.25),
    )

    balls   = int(pitch.get("balls", 0))
    strikes = int(pitch.get("strikes", 0))
    swing_rate_count = hitter_features.get(
        f"hitter_swing_rate_count_{balls}_{strikes}",
        hitter_features.get("hitter_swing_rate", 0.47),
    )

    return {
        "hitter_swing_rate_this_zone":   swing_rate_z,
        "hitter_whiff_rate_this_zone":   whiff_rate_z,
        "hitter_swing_rate_this_family": swing_rate_fam,
        "hitter_whiff_rate_this_family": whiff_rate_fam,
        "hitter_swing_rate_this_count":  swing_rate_count,
    }


# Pitcher feature resolution — disconnected from pipeline.
# Kept here so src/pitchers/features.py can still be imported independently.
# _pitcher_df_cache: dict = {}
#
# def _load_pitcher_df():
#     if "df" not in _pitcher_df_cache:
#         from src.pitchers.features import load_pitcher_features
#         _pitcher_df_cache["df"] = load_pitcher_features()
#     return _pitcher_df_cache["df"]
#
# def _resolve_pitcher_features(pitcher_id: int = None) -> dict:
#     from src.pitchers.features import get_pitcher_feature_row, get_median_pitcher_features
#     pitcher_df = _load_pitcher_df()
#     if pitcher_id is None:
#         return get_median_pitcher_features(pitcher_df)
#     return get_pitcher_feature_row(pitcher_id, pitcher_df)


# ---------------------------------------------------------------------------
# Hitter profile resolution
# ---------------------------------------------------------------------------

def _resolve_hitter_profile(hitter_name: str):
    """
    Load a pre-built profile by name. Falls back to league average if not found.
    Returns (feature_dict, is_thin_sample, used_fallback).
    """
    from src.hitters.profiles import (
        get_league_average_profile,
        get_player_id,
        load_profile,
        profile_to_feature_dict,
    )

    try:
        import pandas as pd
        df = pd.read_parquet(PROCESSED_PATH, columns=["batter", "player_name"])
        pid = get_player_id(hitter_name, df)
        profile = load_profile(pid, PROFILE_DIR)
        return profile_to_feature_dict(profile), profile.is_thin_sample, False
    except FileNotFoundError:
        pass
    except ValueError:
        pass

    warnings.warn(
        f"Hitter '{hitter_name}' not found — using league-average profile.",
        stacklevel=3,
    )
    avg_profile = get_league_average_profile()
    return profile_to_feature_dict(avg_profile), True, True


# ---------------------------------------------------------------------------
# Pitch quality interpretation
# ---------------------------------------------------------------------------

def interpret_pitch(
    p_swing: float,
    p_contact: float,
    p_hard: float,
    plate_x: float,
    plate_z: float,
) -> str:
    """
    Human-readable pitch quality assessment from three-stage probabilities.

    Args:
        p_swing:   P(swing) — probability hitter swings
        p_contact: P(contact) joint — p_swing * p_contact_given_swing
        p_hard:    P(hard_contact | contact) — conditional hard-hit rate
        plate_x, plate_z: pitch location
    """
    zone = _classify_zone(plate_x, plate_z)

    if zone == "waste":
        if p_swing < 0.15:
            return "Ball — hitter won't swing. Count advances."
        return (
            f"Chase pitch — {p_swing:.0%} swing chance but "
            f"only {p_contact:.0%} contact. Effective if thrown."
        )

    if zone == "chase":
        if p_swing >= 0.30:
            if p_contact < 0.20:
                return "Elite chase — hitter swings and misses."
            elif p_hard < 0.25:
                return "Good chase — weak contact likely."
            else:
                return "Risky chase — hitter can damage this."
        return "Ball — on the edge, hitter likely takes it."

    # in_zone
    if p_swing >= 0.70 and p_hard >= 0.30:
        return f"Danger — {p_swing:.0%} swing, {p_hard:.0%} hard contact chance."
    if p_swing >= 0.50 and p_contact >= 0.60:
        return "Competitive — hitter engages but contact quality varies."
    if p_swing < 0.40:
        return "Frozen — hitter likely takes this pitch."
    return f"Pitcher's pitch — in zone but only {p_contact:.0%} contact chance."


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_pitch(
    pitch: dict,
    hitter_name: str,
    model_dir: Path = MODEL_DIR,
) -> dict:
    """
    Run three-stage prediction for a single pitch against a named hitter.

    Returns:
        {
            "p_swing": float,
            "p_contact_given_swing": float,
            "p_hard_given_contact": float,
            "p_contact": float,       # p_swing * p_contact_given_swing
            "p_hard_contact": float,  # p_contact * p_hard_given_contact
            "pitch_quality": str,
            "zone": str,              # in_zone / chase / waste
            "hitter": str,
            "pitch_type": str,
            "count": str,
            "is_thin_sample": bool,
            "used_fallback": bool,
        }
    """
    validate_pitch_dict(pitch)
    swing_pair, contact_pair, hc_pair, feature_cols, encodings = load_models(model_dir)
    swing_model, swing_cal = swing_pair
    contact_model, contact_cal = contact_pair
    hc_model, hc_cal = hc_pair

    hitter_features, is_thin, used_fallback = _resolve_hitter_profile(hitter_name)
    contextual_features = _resolve_contextual_hitter_features(pitch, hitter_features)
    combined_features = {**hitter_features, **contextual_features}
    row = build_feature_row(pitch, combined_features, feature_cols, encodings)

    def _calibrated(model, cal, row):
        raw = float(model.predict_proba(row)[0, 1])
        val = float(cal.predict([raw])[0]) if cal else raw
        # Cap at 0.97 to prevent calibration cliff-edge saturation.
        # The isotonic calibrator has a step from ~0.832 to 1.0 with no
        # examples between raw ≈ 0.89 and 1.0, causing many borderline
        # cases to hard-saturate. 0.97 preserves the meaning (very likely)
        # while keeping the output differentiable.
        return float(np.clip(val, 0.0, 0.97))

    p_swing             = _calibrated(swing_model, swing_cal, row)
    p_contact_given_sw  = _calibrated(contact_model, contact_cal, row)
    p_hard_given_con    = _calibrated(hc_model, hc_cal, row)

    p_contact      = p_swing * p_contact_given_sw
    p_hard_contact = p_contact * p_hard_given_con

    px = float(pitch.get("plate_x", 0))
    pz = float(pitch.get("plate_z", 2.5))
    quality = interpret_pitch(p_swing, p_contact, p_hard_given_con, px, pz)
    zone    = _classify_zone(px, pz)

    return {
        "p_swing":               round(p_swing, 4),
        "p_contact_given_swing": round(p_contact_given_sw, 4),
        "p_hard_given_contact":  round(p_hard_given_con, 4),
        "p_contact":             round(p_contact, 4),
        "p_hard_contact":        round(p_hard_contact, 4),
        "pitch_quality":         quality,
        "zone":                  zone,
        "hitter":                hitter_name,
        "pitch_type":            pitch["pitch_type"],
        "count":                 f"{pitch['balls']}-{pitch['strikes']}",
        "is_thin_sample":        is_thin,
        "used_fallback":         used_fallback,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _find_weakest_hitter(profile_dir: Path) -> str:
    import json as _json
    best_name = None
    best_rate = float("inf")
    for path in profile_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = _json.load(f)
        except Exception:
            continue
        if data.get("is_thin_sample", True):
            continue
        if data.get("sample_size", 0) < 500:
            continue
        rate = data.get("hard_hit_rate", float("inf"))
        if rate < best_rate:
            best_rate = rate
            best_name = data.get("player_name", "")
    return best_name or "league average"


def _print_result(label: str, result: dict) -> None:
    note = ""
    if result.get("used_fallback"):
        note = "  [league-average profile]"
    elif result.get("is_thin_sample"):
        note = "  [thin sample — blended]"
    print(f"  {label}{note}")
    print(f"    P(swing)={result['p_swing']:.3f}  "
          f"P(contact|swing)={result['p_contact_given_swing']:.3f}  "
          f"P(hard|contact)={result['p_hard_given_contact']:.3f}")
    print(f"    P(contact)={result['p_contact']:.3f}  "
          f"P(hard_contact)={result['p_hard_contact']:.3f}")
    print(f"    Quality: {result['pitch_quality']}")
    print(f"    Pitch: {result['pitch_type']}  Count: {result['count']}")


if __name__ == "__main__":
    _BASE = {
        "release_spin_rate": 2350.0,
        "pfx_x": -0.4, "pfx_z": 1.1,
        "release_pos_x": -1.8, "release_pos_z": 6.1, "release_extension": 6.5,
        "plate_x": 0.0, "plate_z": 2.5,
        "pitch_number": 1,
        "prev_pitch_type": "FIRST_PITCH", "prev_pitch_speed": 0.0,
        "prev_pitch_result": "FIRST_PITCH",
        "on_1b": None, "on_2b": None, "on_3b": None,
        "inning": 1, "score_diff": 0, "p_throws": "R",
        "game_date": "2024-06-15",
    }

    print("=" * 60)
    print("PITCH DUEL — predict.py three-stage demo")
    print("=" * 60)

    # 95 mph FF center-cut vs Ohtani — 1-2 count
    p1 = {**_BASE, "release_speed": 95.0, "pitch_type": "FF",
          "balls": 1, "strikes": 2, "pitch_number": 3,
          "prev_pitch_type": "FF", "prev_pitch_speed": 94.0,
          "prev_pitch_result": "called_strike"}
    r1 = predict_pitch(p1, "Shohei Ohtani")
    _print_result("95 mph FF center-cut vs Ohtani — 1-2 count", r1)

    print()

    # Same pitch vs weakest hitter
    weakest = _find_weakest_hitter(PROFILE_DIR)
    r3 = predict_pitch(p1, weakest)
    _print_result(f"95 mph FF center-cut vs {weakest} — 1-2 count", r3)

    print()
    hard_spread = abs(r1["p_hard_contact"] - r3["p_hard_contact"])
    swing_spread = abs(r1["p_swing"] - r3["p_swing"])
    print(f"  P(hard_contact) spread Ohtani vs {weakest}: {hard_spread:.3f}")
    print(f"  P(swing) spread: {swing_spread:.3f}  (should be small — both swing at center-cut FF)")
