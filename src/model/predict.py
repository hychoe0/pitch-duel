"""
predict.py — Single-pitch hit probability inference.

Accepts a pitch dict (from Trackman, Rapsodo, or manual input),
merges the named hitter's profile, and returns a hit probability.

Usage:
    from src.model.predict import predict_hit_probability

    result = predict_hit_probability(
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
            'on_1b': None,
            'on_2b': None,
            'on_3b': None,
            'inning': 7,
            'score_diff': -1,
            'p_throws': 'R',
        },
        hitter_name="Shohei Ohtani",
    )
    print(result)
    # {'hit_probability': 0.187, 'hitter': 'Shohei Ohtani', ...}
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

MODEL_DIR = Path("models")
PROFILE_DIR = Path("data/processed/profiles")
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


# ---------------------------------------------------------------------------
# Model loading (cached after first load)
# ---------------------------------------------------------------------------

_cache: dict = {}


def load_model(model_dir: Path = MODEL_DIR) -> tuple:
    """
    Load and cache (model, feature_cols, encodings, calibrator, is_classifier).
    Auto-detects regressor vs classifier from the model's objective function.
    Uses XGBoost native JSON format for version safety.
    """
    if _cache:
        return (_cache["model"], _cache["feature_cols"], _cache["encodings"],
                _cache["calibrator"], _cache["is_classifier"])

    model_path = model_dir / "pitch_duel_xgb.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")

    # Load as Booster first to read objective without assuming wrapper type
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    objective = booster.attr("objective") or ""

    _CLASSIFIER_OBJECTIVES = {"binary:logistic", "binary:hinge",
                               "multi:softmax", "multi:softprob"}
    is_classifier = objective in _CLASSIFIER_OBJECTIVES

    if is_classifier:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    with open(model_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(model_dir / "encodings.json") as f:
        encodings = json.load(f)

    calibrator_path = model_dir / "calibrator.pkl"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None

    _cache.update({"model": model, "feature_cols": feature_cols,
                   "encodings": encodings, "calibrator": calibrator,
                   "is_classifier": is_classifier})
    return model, feature_cols, encodings, calibrator, is_classifier


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
    crackdown = datetime.date(2021, 6, 21)
    crackdown_end = datetime.date(2022, 1, 1)
    pitch_clock = datetime.date(2023, 1, 1)
    if game_date < crackdown:
        return 0  # pre_crackdown
    if game_date < crackdown_end:
        return 1  # post_crackdown
    if game_date < pitch_clock:
        return 2  # ambiguous
    return 3      # pitch_clock


def encode_pitch_dict(pitch: dict, encodings: dict) -> dict:
    """Apply integer encodings to raw pitch dict. Returns new dict with _enc keys."""
    import datetime
    ptm = encodings["PITCH_TYPE_MAP"]
    prm = encodings["PREV_RESULT_MAP"]

    encoded = dict(pitch)  # shallow copy

    encoded["pitch_type_enc"] = ptm.get(pitch["pitch_type"], ptm.get("OTHER", 15))
    encoded["prev_pitch_type_enc"] = ptm.get(
        pitch.get("prev_pitch_type", "FIRST_PITCH"), ptm.get("OTHER", 15)
    )
    encoded["prev_pitch_result_enc"] = prm.get(
        pitch.get("prev_pitch_result", "FIRST_PITCH"), prm.get("OTHER", 7)
    )
    encoded["hitter_stand_enc"] = 0 if pitch.get("stand", "R") == "L" else 1
    encoded["p_throws_enc"] = 0 if pitch.get("p_throws", "R") == "L" else 1
    encoded["on_1b_flag"] = int(bool(pitch.get("on_1b")))
    encoded["on_2b_flag"] = int(bool(pitch.get("on_2b")))
    encoded["on_3b_flag"] = int(bool(pitch.get("on_3b")))

    # Derived features that must match preprocessing
    release_speed = pitch["release_speed"]
    release_spin_rate = pitch.get("release_spin_rate", 0) or 0
    encoded["spin_to_velo_ratio"] = release_spin_rate / release_speed if release_speed else 0.0

    game_date = pitch.get("game_date", datetime.date.today())
    encoded["era_flag_enc"] = _era_flag_enc(game_date)

    # Warn on unknown pitch types so callers know to update PITCH_TYPE_MAP
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
    merged = {**encoded, **hitter_feature_dict}

    try:
        row = [merged[col] for col in feature_cols]
    except KeyError as e:
        raise KeyError(f"Feature '{e.args[0]}' not found in assembled pitch+profile dict.") from e

    return np.array(row, dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Pitcher feature resolution
# ---------------------------------------------------------------------------

_pitcher_df_cache: dict = {}


def _load_pitcher_df():
    """Load pitcher features parquet once, cache in memory."""
    if "df" not in _pitcher_df_cache:
        from src.pitchers.features import load_pitcher_features
        _pitcher_df_cache["df"] = load_pitcher_features()
    return _pitcher_df_cache["df"]


def _resolve_pitcher_features(pitcher_id: int = None) -> dict:
    """
    Return the 5 pitcher aggregate features for inference.
    Falls back to population medians if pitcher_id is None or unknown.
    """
    from src.pitchers.features import get_pitcher_feature_row, get_median_pitcher_features
    pitcher_df = _load_pitcher_df()
    if pitcher_id is None:
        return get_median_pitcher_features(pitcher_df)
    return get_pitcher_feature_row(pitcher_id, pitcher_df)


# ---------------------------------------------------------------------------
# Hitter profile resolution
# ---------------------------------------------------------------------------

def _resolve_hitter_profile(hitter_name: str):
    """
    Load a pre-built profile by name. Falls back to league average if not found.
    Returns (profile, used_fallback).
    """
    from src.hitters.profiles import (
        get_league_average_profile,
        get_player_id,
        load_profile,
        profile_to_feature_dict,
    )

    # Try loading from saved profiles directory
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

    # Fallback to league average
    warnings.warn(
        f"Hitter '{hitter_name}' not found — using league-average profile.",
        stacklevel=3,
    )
    avg_profile = get_league_average_profile()
    return profile_to_feature_dict(avg_profile), True, True


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_hit_probability(
    pitch: dict,
    hitter_name: str,
    pitcher_id: int = None,
    model_dir: Path = MODEL_DIR,
) -> dict:
    """
    Predict xwOBA for a single pitch against a named hitter.

    Model type is detected automatically:
    - XGBRegressor  → model.predict() returns xwOBA directly
    - XGBClassifier → model.predict_proba()[:, 1] returns hit probability

    Args:
        pitch: dict with all raw pitch keys (see REQUIRED_RAW_KEYS)
        hitter_name: player name string (e.g. "Shohei Ohtani")
        pitcher_id: optional MLBAM pitcher ID — if provided, looks up that
                    pitcher's aggregate features; otherwise uses population medians.
        model_dir: path to directory containing trained model artifacts

    Returns:
        {
            'xwoba_prediction': float,       # calibrated xwOBA (regressor) or hit prob (classifier)
            'xwoba_prediction_raw': float,   # uncalibrated output
            'hit_probability': float,        # binarized: 1.0 if xwoba_prediction > 0.15 else 0.0
            'model_type': str,               # 'regressor' or 'classifier'
            'hitter': str,
            'pitch_type': str,
            'count': str,                    # e.g. "1-2"
            'is_thin_sample': bool,
            'used_fallback': bool,
        }
    """
    validate_pitch_dict(pitch)
    model, feature_cols, encodings, calibrator, is_classifier = load_model(model_dir)

    hitter_features, is_thin, used_fallback = _resolve_hitter_profile(hitter_name)

    # Load pitcher aggregate features (5 columns: release height, horizontal position,
    # extension, avg fastball speed, arm-slot angle). Falls back to population medians
    # when pitcher_id is None or unknown — ensures the feature vector is always complete.
    pitcher_features = _resolve_pitcher_features(pitcher_id)

    # Merge hitter + pitcher features; build_feature_row adds encoded pitch features
    combined_features = {**hitter_features, **pitcher_features}
    row = build_feature_row(pitch, combined_features, feature_cols, encodings)

    # Predict using the detected model type
    if is_classifier:
        raw_output = float(model.predict_proba(row)[0, 1])
    else:
        raw_output = float(max(0.0, model.predict(row)[0]))

    if calibrator is not None:
        cal_output = float(calibrator.predict([raw_output])[0])
    else:
        cal_output = raw_output
    cal_output = max(0.0, cal_output)

    return {
        "xwoba_prediction":     round(cal_output, 4),
        "xwoba_prediction_raw": round(raw_output, 4),
        "hit_probability":      1.0 if cal_output > 0.15 else 0.0,
        "model_type":           "classifier" if is_classifier else "regressor",
        "hitter": hitter_name,
        "pitch_type": pitch["pitch_type"],
        "count": f"{pitch['balls']}-{pitch['strikes']}",
        "is_thin_sample": is_thin,
        "used_fallback": used_fallback,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _find_weakest_hitter(profile_dir: Path) -> str:
    """
    Scan all saved profiles and return the player name with the lowest
    hard_hit_rate among profiles with at least 500 sample pitches.
    Excludes profiles flagged as thin (blended with league averages).
    """
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
    model_tag = f"[{result['model_type']}]" if result.get("model_type") else ""
    xw = result["xwoba_prediction"]
    note = ""
    if result.get("used_fallback"):
        note = "  ⚠ league-average profile"
    elif result.get("is_thin_sample"):
        note = "  ⚠ thin sample (blended)"
    print(f"  {label}")
    print(f"    xwOBA: {xw:.4f}  {model_tag}{note}")
    print(f"    Pitch: {result['pitch_type']}  Count: {result['count']}")


if __name__ == "__main__":
    _BASE = {
        "release_spin_rate": 2350.0,
        "pfx_x":             -0.4,
        "pfx_z":              1.1,
        "release_pos_x":     -1.8,
        "release_pos_z":      6.1,
        "release_extension":  6.5,
        "plate_x":            0.0,
        "plate_z":            2.5,
        "pitch_number":       1,
        "prev_pitch_type":   "FIRST_PITCH",
        "prev_pitch_speed":   0.0,
        "prev_pitch_result": "FIRST_PITCH",
        "on_1b":  None,
        "on_2b":  None,
        "on_3b":  None,
        "inning":     1,
        "score_diff": 0,
        "p_throws":  "R",
        "game_date": "2024-06-15",
    }

    print("=" * 50)
    print("PITCH DUEL — predict.py sanity demo")
    print("=" * 50)

    # Load model once and print type
    _, _, _, _, is_clf = load_model()
    mtype = "XGBClassifier (predict_proba)" if is_clf else "XGBRegressor (predict)"
    print(f"  Model detected: {mtype}\n")

    # 1. 95 mph FF to Ohtani, 1-2 count
    p1 = {**_BASE, "release_speed": 95.0, "pitch_type": "FF",
          "balls": 1, "strikes": 2, "pitch_number": 3,
          "prev_pitch_type": "FF", "prev_pitch_speed": 94.0,
          "prev_pitch_result": "called_strike"}
    r1 = predict_hit_probability(p1, "Shohei Ohtani")
    _print_result("95 mph FF vs Ohtani — 1-2 count", r1)

    # 2. 85 mph CU to Ohtani, 0-0 count
    p2 = {**_BASE, "release_speed": 85.0, "pitch_type": "CU",
          "pfx_z": -0.5,   # downward break on a curve
          "plate_z": 2.0,  # low in zone
          "balls": 0, "strikes": 0}
    r2 = predict_hit_probability(p2, "Shohei Ohtani")
    _print_result("85 mph CU vs Ohtani — 0-0 count", r2)

    # 3. 95 mph FF to the weakest hitter in profiles, 1-2 count
    weakest = _find_weakest_hitter(PROFILE_DIR)
    r3 = predict_hit_probability(p1, weakest)
    _print_result(f"95 mph FF vs {weakest} (weakest by hard-hit rate) — 1-2 count", r3)

    print()
    spread = abs(r1["xwoba_prediction"] - r3["xwoba_prediction"])
    print(f"  Ohtani vs weakest spread: {spread:.4f}")
