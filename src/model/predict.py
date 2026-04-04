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
    Load and cache (model, feature_cols, encodings, calibrator).
    Uses XGBoost native JSON format for version safety.
    """
    if _cache:
        return _cache["model"], _cache["feature_cols"], _cache["encodings"], _cache["calibrator"]

    model_path = model_dir / "pitch_duel_xgb.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    with open(model_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(model_dir / "encodings.json") as f:
        encodings = json.load(f)

    calibrator_path = model_dir / "calibrator.pkl"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None

    _cache.update({"model": model, "feature_cols": feature_cols,
                   "encodings": encodings, "calibrator": calibrator})
    return model, feature_cols, encodings, calibrator


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
    model_dir: Path = MODEL_DIR,
) -> dict:
    """
    Predict xwOBA for a single pitch against a named hitter.

    Args:
        pitch: dict with all raw pitch keys (see REQUIRED_RAW_KEYS)
        hitter_name: player name string (e.g. "Shohei Ohtani")
        model_dir: path to directory containing trained model artifacts

    Returns:
        {
            'xwoba_prediction': float,       # calibrated xwOBA (0.0–2.0 approx)
            'xwoba_prediction_raw': float,   # uncalibrated regressor output
            'hit_probability': float,        # binarized: 1.0 if xwoba > 0.15 else 0.0
            'hitter': str,
            'pitch_type': str,
            'count': str,                    # e.g. "1-2"
            'is_thin_sample': bool,
            'used_fallback': bool,
        }
    """
    validate_pitch_dict(pitch)
    model, feature_cols, encodings, calibrator = load_model(model_dir)
    hitter_features, is_thin, used_fallback = _resolve_hitter_profile(hitter_name)
    row = build_feature_row(pitch, hitter_features, feature_cols, encodings)
    raw_xwoba = float(max(0.0, model.predict(row)[0]))
    cal_xwoba = float(calibrator.predict([raw_xwoba])[0]) if calibrator is not None else raw_xwoba
    cal_xwoba = max(0.0, cal_xwoba)

    return {
        "xwoba_prediction":     round(cal_xwoba, 4),
        "xwoba_prediction_raw": round(raw_xwoba, 4),
        "hit_probability":      1.0 if cal_xwoba > 0.15 else 0.0,
        "hitter": hitter_name,
        "pitch_type": pitch["pitch_type"],
        "count": f"{pitch['balls']}-{pitch['strikes']}",
        "is_thin_sample": is_thin,
        "used_fallback": used_fallback,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict hit probability for a sample pitch")
    parser.add_argument("--hitter", type=str, default="Shohei Ohtani")
    args = parser.parse_args()

    sample_pitch = {
        "release_speed": 95.2,
        "release_spin_rate": 2350.0,
        "pfx_x": -0.5,
        "pfx_z": 1.2,
        "release_pos_x": -1.8,
        "release_pos_z": 6.1,
        "release_extension": 6.5,
        "plate_x": 0.1,
        "plate_z": 2.8,
        "pitch_type": "FF",
        "balls": 1,
        "strikes": 2,
        "pitch_number": 4,
        "prev_pitch_type": "SL",
        "prev_pitch_speed": 88.0,
        "prev_pitch_result": "swinging_strike",
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "inning": 7,
        "score_diff": -1,
        "p_throws": "R",
    }

    result = predict_hit_probability(sample_pitch, args.hitter)
    print(f"\nxwOBA prediction for {result['hitter']}: {result['xwoba_prediction']:.4f}")
    print(f"  Pitch: {result['pitch_type']}  |  Count: {result['count']}")
    if result["used_fallback"]:
        print("  (league-average profile used — hitter not in database)")
    elif result["is_thin_sample"]:
        print("  (thin sample — profile blended with league averages)")
