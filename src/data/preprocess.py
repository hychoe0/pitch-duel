"""
preprocess.py — Clean raw Statcast data and engineer model features.

Produces a parquet file at data/processed/statcast_processed.parquet
with all features ready for training, plus separate train/test splits.

Usage:
    python -m src.data.preprocess
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Encoding maps — must stay in sync with predict.py
# ---------------------------------------------------------------------------

PITCH_TYPE_MAP = {
    "FF": 0,   # 4-seam fastball
    "SI": 1,   # sinker
    "FC": 2,   # cutter
    "SL": 3,   # slider
    "CU": 4,   # curveball
    "CH": 5,   # changeup
    "FS": 6,   # splitter
    "KC": 7,   # knuckle-curve
    "CS": 8,   # slow curve
    "KN": 9,   # knuckleball
    "EP": 10,  # eephus
    "FO": 11,  # forkball
    "SC": 12,  # screwball
    "SV": 13,  # sweeper
    "FIRST_PITCH": 14,  # sentinel: no previous pitch
    "OTHER": 15,        # catch-all for unmapped types
}

PREV_RESULT_MAP = {
    "FIRST_PITCH": 0,
    "ball": 1,
    "blocked_ball": 1,
    "called_strike": 2,
    "swinging_strike": 3,
    "swinging_strike_blocked": 3,
    "foul": 4,
    "foul_tip": 4,
    "foul_bunt": 4,
    "hit_into_play": 5,
    "pitchout": 6,
    "OTHER": 7,
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}

# Era flag encoding — reflects spin rate integrity and rule changes
ERA_FLAG_MAP = {
    "pre_crackdown": 0,   # before Jun 21 2021 — Spider Tack era
    "post_crackdown": 1,  # Jun 21 – Dec 2021 — cleanest spin data
    "ambiguous": 2,       # 2022 — workarounds to umpire checks
    "pitch_clock": 3,     # 2023 onward
}

# Training sample weights by era (target: current pitch-clock baseball)
# 2020 is excluded entirely (weight 0.0 → dropped in clean())
ERA_SAMPLE_WEIGHTS = {
    "pitch_clock_2026": 2.0,
    "pitch_clock_2025": 1.8,
    "pitch_clock_2024": 1.5,
    "pitch_clock_2023": 1.2,
    "ambiguous": 0.6,
    "post_crackdown": 0.7,
    "pre_crackdown": 0.3,
    "pre_statcast": 0.3,  # 2017-2019 (pitch quality features only)
}

# Feature column groups — single source of truth imported by train.py / predict.py
PITCH_FEATURES = [
    "release_speed",
    "spin_to_velo_ratio",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "plate_x",
    "plate_z",
    "pitch_type_enc",
    "era_flag_enc",
]

CONTEXT_FEATURES = [
    "balls",
    "strikes",
    "pitch_number",
    "prev_pitch_type_enc",
    "prev_pitch_speed",
    "prev_pitch_result_enc",
    "on_1b_flag",
    "on_2b_flag",
    "on_3b_flag",
    "inning",
    "score_diff",
]

HITTER_PROFILE_FEATURES = [
    "hitter_swing_rate",
    "hitter_chase_rate",
    "hitter_contact_rate",
    "hitter_hard_hit_rate",
    "hitter_whiff_rate",
    "hitter_stand_enc",
    "p_throws_enc",
]

ALL_FEATURES = PITCH_FEATURES + CONTEXT_FEATURES + HITTER_PROFILE_FEATURES


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and cast types. Excludes 2020 (COVID season)."""
    before = len(df)

    df["game_date"] = pd.to_datetime(df["game_date"])

    # Exclude 2020 — COVID season, no fans, abnormal conditions
    df = df[df["game_date"].dt.year != 2020]

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "UN")]
    df = df[df["release_speed"].notna()]
    df = df[df["plate_x"].notna() & df["plate_z"].notna()]
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    print(f"Cleaned: {before:,} → {len(df):,} rows (2020 excluded)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def make_target(df: pd.DataFrame) -> pd.Series:
    """1 if this pitch was put in play for a hit, 0 otherwise."""
    return df["events"].isin(HIT_EVENTS).astype(int)


# ---------------------------------------------------------------------------
# score_diff from batting team's perspective
# ---------------------------------------------------------------------------

def make_score_diff(df: pd.DataFrame) -> pd.Series:
    """Positive = batting team leads."""
    return np.where(
        df["inning_topbot"] == "Top",
        df["away_score"] - df["home_score"],
        df["home_score"] - df["away_score"],
    )


# ---------------------------------------------------------------------------
# Previous-pitch features via shift within PA
# ---------------------------------------------------------------------------

def make_prev_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds prev_pitch_type, prev_pitch_speed, prev_pitch_result columns.
    Groups by (game_pk, at_bat_number) so PA boundaries are respected.
    First pitch of each PA receives sentinel values.
    """
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    grp = df.groupby(["game_pk", "at_bat_number"], sort=False)

    df["prev_pitch_type"] = grp["pitch_type"].shift(1).fillna("FIRST_PITCH")
    df["prev_pitch_speed"] = grp["release_speed"].shift(1).fillna(-1.0)
    df["prev_pitch_result"] = grp["description"].shift(1).fillna("FIRST_PITCH")

    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    df["pitch_type_enc"] = (
        df["pitch_type"].map(PITCH_TYPE_MAP).fillna(PITCH_TYPE_MAP["OTHER"]).astype(int)
    )
    df["prev_pitch_type_enc"] = (
        df["prev_pitch_type"].map(PITCH_TYPE_MAP).fillna(PITCH_TYPE_MAP["OTHER"]).astype(int)
    )
    return df


def encode_prev_result(df: pd.DataFrame) -> pd.DataFrame:
    df["prev_pitch_result_enc"] = (
        df["prev_pitch_result"].map(PREV_RESULT_MAP).fillna(PREV_RESULT_MAP["OTHER"]).astype(int)
    )
    return df


def encode_handedness(df: pd.DataFrame) -> pd.DataFrame:
    df["hitter_stand_enc"] = (df["stand"] == "R").astype(int)
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    return df


# ---------------------------------------------------------------------------
# Base-runner flags
# ---------------------------------------------------------------------------

def make_runner_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["on_1b_flag"] = df["on_1b"].notna().astype(int)
    df["on_2b_flag"] = df["on_2b"].notna().astype(int)
    df["on_3b_flag"] = df["on_3b"].notna().astype(int)
    return df


# ---------------------------------------------------------------------------
# Era flag
# ---------------------------------------------------------------------------

_CRACKDOWN_DATE = pd.Timestamp("2021-06-21")
_CRACKDOWN_END = pd.Timestamp("2022-01-01")
_PITCH_CLOCK_START = pd.Timestamp("2023-01-01")


def make_era_flag(df: pd.DataFrame) -> pd.Series:
    """
    Assign each row an era string based on game_date, then encode to int.
      pre_crackdown  (0): before Jun 21 2021
      post_crackdown (1): Jun 21 – Dec 2021
      ambiguous      (2): 2022
      pitch_clock    (3): 2023 onward
    """
    conditions = [
        df["game_date"] < _CRACKDOWN_DATE,
        (df["game_date"] >= _CRACKDOWN_DATE) & (df["game_date"] < _CRACKDOWN_END),
        (df["game_date"] >= _CRACKDOWN_END) & (df["game_date"] < _PITCH_CLOCK_START),
    ]
    choices = [
        ERA_FLAG_MAP["pre_crackdown"],
        ERA_FLAG_MAP["post_crackdown"],
        ERA_FLAG_MAP["ambiguous"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=ERA_FLAG_MAP["pitch_clock"]),
        index=df.index,
        dtype=int,
    )


def make_spin_to_velo_ratio(df: pd.DataFrame) -> pd.Series:
    """spin_to_velo_ratio replaces raw release_spin_rate as the primary spin feature."""
    return df["release_spin_rate"] / df["release_speed"]


def make_sample_weights(df: pd.DataFrame) -> pd.Series:
    """
    Assign a training sample weight to each row based on era and year.
    2026 partial and full pitch-clock seasons are upweighted; pre-crackdown downweighted.
    """
    year = df["game_date"].dt.year
    era_flag = df["era_flag_enc"]

    weights = pd.Series(ERA_SAMPLE_WEIGHTS["pre_crackdown"], index=df.index, dtype=float)

    weights = np.where(era_flag == ERA_FLAG_MAP["post_crackdown"], ERA_SAMPLE_WEIGHTS["post_crackdown"], weights)
    weights = np.where(era_flag == ERA_FLAG_MAP["ambiguous"], ERA_SAMPLE_WEIGHTS["ambiguous"], weights)
    weights = np.where((era_flag == ERA_FLAG_MAP["pitch_clock"]) & (year == 2023), ERA_SAMPLE_WEIGHTS["pitch_clock_2023"], weights)
    weights = np.where((era_flag == ERA_FLAG_MAP["pitch_clock"]) & (year == 2024), ERA_SAMPLE_WEIGHTS["pitch_clock_2024"], weights)
    weights = np.where((era_flag == ERA_FLAG_MAP["pitch_clock"]) & (year == 2025), ERA_SAMPLE_WEIGHTS["pitch_clock_2025"], weights)
    weights = np.where((era_flag == ERA_FLAG_MAP["pitch_clock"]) & (year == 2026), ERA_SAMPLE_WEIGHTS["pitch_clock_2026"], weights)

    return pd.Series(weights, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Per-pitcher median imputation for extension / spin rate
# ---------------------------------------------------------------------------

def impute_pitcher_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute release_extension and release_spin_rate with per-pitcher medians.
    Falls back to pitch-type median, then global median.
    """
    for col in ["release_extension", "release_spin_rate"]:
        pitcher_median = df.groupby("pitcher")[col].transform("median")
        pitch_type_median = df.groupby("pitch_type")[col].transform("median")
        global_median = df[col].median()
        df[col] = df[col].fillna(pitcher_median).fillna(pitch_type_median).fillna(global_median)
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame) -> tuple:
    """Temporal split: train < 2023, test >= 2023."""
    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["game_date"] < cutoff].copy()
    test = df[df["game_date"] >= cutoff].copy()
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "statcast_processed.parquet", index=False)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"Saved processed data to {PROCESSED_DIR}/")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(raw_df: pd.DataFrame) -> tuple:
    """
    End-to-end preprocessing.
    Returns (full_df, train_df, test_df).
    Hitter profile features are NOT merged here — that happens in profiles.py
    after per-hitter profiles are built.
    """
    df = clean(raw_df)
    df = make_prev_pitch_features(df)
    df = impute_pitcher_medians(df)

    df["hit"] = make_target(df)
    df["score_diff"] = make_score_diff(df)

    df = encode_pitch_types(df)
    df = encode_prev_result(df)
    df = encode_handedness(df)
    df = make_runner_flags(df)

    # Era integrity features
    df["era_flag_enc"] = make_era_flag(df)
    df["spin_to_velo_ratio"] = make_spin_to_velo_ratio(df)
    df["sample_weight"] = make_sample_weights(df)

    train, test = split_data(df)
    save_processed(df, train, test)
    return df, train, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.fetch import load_raw

    raw = load_raw()
    run_preprocessing(raw)
