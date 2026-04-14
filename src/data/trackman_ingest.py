"""
trackman_ingest.py — Convert Trackman Baseball CSV exports into pitch dicts
compatible with src/model/predict.py's predict_pitch() function.

Usage:
    from src.data.trackman_ingest import load_trackman_session
    from src.model.predict import predict_pitch

    pitches = load_trackman_session("data/sample/kbo_pitcher_trackman.csv")
    result = predict_pitch(pitches[0], hitter_name="Shohei Ohtani")
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pitch type mapping: Trackman label → Statcast code
# ---------------------------------------------------------------------------

PITCH_TYPE_MAP = {
    "Fastball":    "FF",
    "Sinker":      "SI",
    "Cutter":      "FC",
    "Slider":      "SL",
    "Curveball":   "CU",
    "Changeup":    "CH",
    "Splitter":    "FS",
    "Knuckleball": "KN",
    "Sweeper":     "ST",
    "Screwball":   "SC",
    "Other":       "OTHER",
    "Undefined":   None,  # triggers fallback
}

# PitchCall → prev_pitch_result mapping
PITCH_CALL_MAP = {
    "StrikeCalled":     "called_strike",
    "StrikeSwinging":   "swinging_strike",
    "FoulBall":         "foul",
    "FoulTip":          "foul_tip",
    "BallCalled":       "ball",
    "HitByPitch":       "ball",
    "InPlay":           "hit_into_play",
    "BallIntentional":  "ball",
}

# Trackman column → model key (direct renames, no unit conversion)
DIRECT_MAP = {
    "RelSpeed":        "release_speed",
    "SpinRate":        "release_spin_rate",
    "RelHeight":       "release_pos_z",
    "RelSide":         "release_pos_x",
    "Extension":       "release_extension",
    "PlateLocSide":    "plate_x",
    "PlateLocHeight":  "plate_z",
    "Inning":          "inning",
    "Balls":           "balls",
    "Strikes":         "strikes",
    "PitchNo":         "pitch_number",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_pitch_type(tagged: object, auto: object) -> str:
    """Resolve Trackman pitch type label to Statcast code."""
    for raw in (tagged, auto):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        label = str(raw).strip()
        if label in ("", "Undefined", "Unknown"):
            continue
        code = PITCH_TYPE_MAP.get(label)
        if code is not None:
            return code
        # Unknown label — warn once and return OTHER
        warnings.warn(
            f"Unknown Trackman pitch type '{label}' — encoded as OTHER.",
            stacklevel=4,
        )
        return "OTHER"
    warnings.warn("Both TaggedPitchType and AutoPitchType are null — encoding as OTHER.", stacklevel=4)
    return "OTHER"


def _infer_p_throws(row: pd.Series) -> str:
    """
    Resolve pitcher handedness from PitcherThrows column or RelSide heuristic.
    Negative RelSide (arm-side left of rubber from catcher view) → RHP.
    """
    if "PitcherThrows" in row.index:
        val = str(row["PitcherThrows"]).strip()
        if val in ("Right", "R"):
            return "R"
        if val in ("Left", "L"):
            return "L"
    # Heuristic: RHP stands to the right of rubber, so RelSide is negative
    rel_side = row.get("RelSide", np.nan)
    if not (isinstance(rel_side, float) and np.isnan(rel_side)):
        if float(rel_side) < 0:
            return "R"
        else:
            return "L"
    warnings.warn("Cannot determine pitcher handedness — defaulting to R.", stacklevel=4)
    return "R"


def _map_pitch_call(call: object) -> str:
    if call is None or (isinstance(call, float) and np.isnan(call)):
        return "ball"
    return PITCH_CALL_MAP.get(str(call).strip(), "ball")


def _score_diff(row: pd.Series, absent_cols: set) -> int:
    """Compute score_diff from batting team perspective."""
    if "HomeScore" not in row.index or "AwayScore" not in row.index or "Top/Bottom" not in row.index:
        absent_cols.add("score_diff (HomeScore/AwayScore/Top/Bottom absent)")
        return 0
    home = int(row.get("HomeScore", 0) or 0)
    away = int(row.get("AwayScore", 0) or 0)
    half = str(row.get("Top/Bottom", "Top")).strip()
    return (away - home) if half == "Top" else (home - away)


def _runner(row: pd.Series, col: str, absent_cols: set) -> object:
    """Return runner ID (truthy) or None from RunnerXB column."""
    if col not in row.index:
        absent_cols.add(col)
        return None
    val = row[col]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# PA boundary detection and prev-pitch feature derivation
# ---------------------------------------------------------------------------

def _detect_pa_group(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of PA group keys (one integer per PA).
    Uses PAofInning if present; otherwise approximates boundaries by detecting
    Balls+Strikes resetting to 0+0 within a (Pitcher, Batter, Inning, Half) group.
    """
    if "PAofInning" in df.columns:
        half_col = "Top/Bottom" if "Top/Bottom" in df.columns else None
        if half_col:
            return (
                df["Pitcher"].astype(str) + "|" +
                df["Inning"].astype(str) + "|" +
                df[half_col].astype(str) + "|" +
                df["PAofInning"].astype(str)
            )
        return df["Pitcher"].astype(str) + "|" + df["Inning"].astype(str) + "|" + df["PAofInning"].astype(str)

    # Approximation: group on (Pitcher, Batter, Inning, Half) and detect 0-0 count resets
    half_col = "Top/Bottom" if "Top/Bottom" in df.columns else None
    base_keys = df["Pitcher"].astype(str) + "|" + df["Batter"].astype(str) + "|" + df["Inning"].astype(str)
    if half_col:
        base_keys = base_keys + "|" + df[half_col].astype(str)

    pa_ids = []
    pa_counter = 0
    prev_key = None
    prev_balls = -1
    prev_strikes = -1

    for i, row in df.iterrows():
        key = base_keys[i]
        balls   = int(row.get("Balls", 0) or 0)
        strikes = int(row.get("Strikes", 0) or 0)

        new_pa = (
            key != prev_key or
            (balls == 0 and strikes == 0 and (prev_balls > 0 or prev_strikes > 0))
        )
        if new_pa:
            pa_counter += 1

        pa_ids.append(pa_counter)
        prev_key     = key
        prev_balls   = balls
        prev_strikes = strikes

    return pd.Series(pa_ids, index=df.index)


def _add_prev_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prev_pitch_type, prev_pitch_speed, prev_pitch_result columns."""
    df = df.copy()
    df["_pa_group"] = _detect_pa_group(df)

    # Within each PA, shift pitch_type_raw, release_speed, and pitch_call
    df["prev_pitch_type"]   = df.groupby("_pa_group")["pitch_type"].shift(1)
    df["prev_pitch_speed"]  = df.groupby("_pa_group")["release_speed"].shift(1)
    df["prev_pitch_call"]   = df.groupby("_pa_group")["PitchCall"].shift(1)

    # Sentinel values for first pitch of each PA
    df["prev_pitch_type"]  = df["prev_pitch_type"].fillna("FIRST_PITCH")
    df["prev_pitch_speed"] = df["prev_pitch_speed"].fillna(-1.0)
    df["prev_pitch_call"]  = df["prev_pitch_call"].fillna("FIRST_PITCH")

    df["prev_pitch_result"] = df["prev_pitch_call"].map(
        lambda c: "FIRST_PITCH" if c == "FIRST_PITCH" else _map_pitch_call(c)
    )

    df.drop(columns=["_pa_group", "prev_pitch_call"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_trackman_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a Trackman CSV and return a clean DataFrame with all conversions
    and derived columns applied. One row = one pitch.
    Each row can be passed directly to row_to_pitch_dict().
    """
    path = Path(path)
    raw  = pd.read_csv(path, low_memory=False)
    absent_cols: set = set()

    # --- Resolve pitch type ---
    tagged = raw.get("TaggedPitchType", pd.Series([None] * len(raw)))
    auto   = raw.get("AutoPitchType",   pd.Series([None] * len(raw)))
    raw["pitch_type"] = [
        _map_pitch_type(t, a) for t, a in zip(tagged, auto)
    ]

    # --- Direct renames ---
    for tm_col, model_col in DIRECT_MAP.items():
        if tm_col in raw.columns:
            raw[model_col] = pd.to_numeric(raw[tm_col], errors="coerce")
        else:
            absent_cols.add(tm_col)
            raw[model_col] = np.nan

    # --- Unit conversions: inches → feet ---
    if "InducedVertBreak" in raw.columns:
        raw["pfx_z"] = pd.to_numeric(raw["InducedVertBreak"], errors="coerce") / 12.0
    else:
        absent_cols.add("InducedVertBreak")
        raw["pfx_z"] = np.nan

    if "HorzBreak" in raw.columns:
        raw["pfx_x"] = pd.to_numeric(raw["HorzBreak"], errors="coerce") / 12.0
    else:
        absent_cols.add("HorzBreak")
        raw["pfx_x"] = np.nan

    # --- Pitcher handedness (per-row) ---
    raw["p_throws"] = raw.apply(_infer_p_throws, axis=1)

    # --- HorzBreak sign correction for LHP ---
    # Trackman HorzBreak: pitcher's POV. Statcast pfx_x: catcher's POV.
    # For LHP, flip pfx_x to match Statcast convention.
    lhp_mask = raw["p_throws"] == "L"
    raw.loc[lhp_mask, "pfx_x"] = -raw.loc[lhp_mask, "pfx_x"]

    # --- score_diff ---
    raw["score_diff"] = raw.apply(
        lambda r: _score_diff(r, absent_cols), axis=1
    )

    # --- Runners ---
    for base, col in [(1, "Runner1B"), (2, "Runner2B"), (3, "Runner3B")]:
        model_col = f"on_{base}b"
        raw[model_col] = raw.apply(
            lambda r, c=col: _runner(r, c, absent_cols), axis=1
        )

    # --- PitchCall passthrough (needed for prev_pitch derivation) ---
    if "PitchCall" not in raw.columns:
        absent_cols.add("PitchCall")
        raw["PitchCall"] = np.nan

    # --- Batter column (passthrough for grouping) ---
    if "Batter" not in raw.columns:
        raw["Batter"] = "Unknown"
    if "Pitcher" not in raw.columns:
        raw["Pitcher"] = "Unknown"

    # --- prev-pitch features (must come after pitch_type and release_speed are set) ---
    raw = _add_prev_pitch_features(raw)

    # --- Drop rows with NaN in critical fields ---
    critical = ["release_speed", "plate_x", "plate_z"]
    before   = len(raw)
    raw      = raw.dropna(subset=critical).reset_index(drop=True)
    dropped  = before - len(raw)

    # --- Data quality report ---
    print(f"\n{'─' * 50}")
    print(f"Trackman ingest: {path.name}")
    print(f"  Total pitches loaded: {len(raw)}" + (f"  ({dropped} dropped — NaN in critical fields)" if dropped else ""))
    print(f"  Velocity: {raw['release_speed'].min():.1f}–{raw['release_speed'].max():.1f} mph  "
          f"(mean {raw['release_speed'].mean():.1f})")
    pt_dist = raw["pitch_type"].value_counts().to_dict()
    print(f"  Pitch types: {pt_dist}")
    if absent_cols:
        unique_absent = sorted({c.split(" (")[0] for c in absent_cols})
        print(f"  Defaulted (absent columns): {unique_absent}")
    print(f"{'─' * 50}\n")

    return raw


def row_to_pitch_dict(row: pd.Series) -> dict:
    """
    Convert a single preprocessed Trackman row (from load_trackman_csv)
    to the dict format expected by predict_pitch().
    """
    return {
        "release_speed":      float(row["release_speed"]),
        "release_spin_rate":  float(row["release_spin_rate"]) if not _isnan(row.get("release_spin_rate")) else 0.0,
        "pfx_x":              float(row["pfx_x"])  if not _isnan(row.get("pfx_x"))  else 0.0,
        "pfx_z":              float(row["pfx_z"])  if not _isnan(row.get("pfx_z"))  else 0.0,
        "release_pos_x":      float(row["release_pos_x"]) if not _isnan(row.get("release_pos_x")) else 0.0,
        "release_pos_z":      float(row["release_pos_z"]) if not _isnan(row.get("release_pos_z")) else 0.0,
        "release_extension":  float(row["release_extension"]) if not _isnan(row.get("release_extension")) else 0.0,
        "plate_x":            float(row["plate_x"]),
        "plate_z":            float(row["plate_z"]),
        "pitch_type":         str(row["pitch_type"]),
        "balls":              int(row["balls"]) if not _isnan(row.get("balls")) else 0,
        "strikes":            int(row["strikes"]) if not _isnan(row.get("strikes")) else 0,
        "pitch_number":       int(row["pitch_number"]) if not _isnan(row.get("pitch_number")) else 1,
        "prev_pitch_type":    str(row.get("prev_pitch_type", "FIRST_PITCH")),
        "prev_pitch_speed":   float(row.get("prev_pitch_speed", -1.0) or -1.0),
        "prev_pitch_result":  str(row.get("prev_pitch_result", "FIRST_PITCH")),
        "on_1b":              row.get("on_1b"),
        "on_2b":              row.get("on_2b"),
        "on_3b":              row.get("on_3b"),
        "inning":             int(row["inning"]) if not _isnan(row.get("inning")) else 1,
        "score_diff":         int(row.get("score_diff", 0) or 0),
        "p_throws":           str(row.get("p_throws", "R")),
    }


def _isnan(val) -> bool:
    try:
        return val is None or (isinstance(val, float) and np.isnan(val))
    except (TypeError, ValueError):
        return False


def load_trackman_session(path: str | Path) -> list[dict]:
    """
    Top-level function: load CSV, apply all transformations, return list of
    pitch dicts ready for predict_pitch().
    """
    df      = load_trackman_csv(path)
    pitches = [row_to_pitch_dict(row) for _, row in df.iterrows()]
    return pitches
