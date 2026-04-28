"""
preprocess_minors.py — Clean raw Triple-A Statcast data for AAA hitter profile building.

Key differences vs preprocess.py:
  - AAA Statcast only exists from 2023+ (pitch-clock era only) — no era handling needed
  - Universal DH at all levels since 2022 — no pitcher-batting filter
  - ABS (automated ball-strike) descriptions normalized to MLB equivalents
  - Output is a lightweight parquet for profile building, NOT a model training set

Output: data/processed/statcast_minors_processed.parquet

Usage:
    python -m src.data.preprocess_minors
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    ERA_SAMPLE_WEIGHTS,
    HIT_EVENTS,
    PITCH_TYPE_MAP,
    PREV_RESULT_MAP,
)

PROCESSED_DIR   = Path("data/processed")
OUTPUT_PARQUET  = PROCESSED_DIR / "statcast_minors_processed.parquet"

# Columns required for profile building — keep all raw Statcast columns
# that profiles.py uses, plus a sample_weight column we add here.
PROFILE_COLUMNS = [
    "batter",
    "pitcher",
    "player_name",
    "stand",
    "p_throws",
    "game_pk",
    "game_date",
    "game_year",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "description",
    "events",
    "launch_speed",
    "launch_angle",
    "zone",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "estimated_woba_using_speedangle",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_team",
    "away_team",
]

# ABS (Automated Ball-Strike system) pitch outcomes from AAA — normalize to MLB equivalents
# so that profiles.py's _SWING_DESCRIPTIONS / _CONTACT_DESCRIPTIONS match correctly.
ABS_DESCRIPTION_MAP = {
    "automatic_ball":   "ball",
    "automatic_strike": "called_strike",
}

# Recency weights for AAA profile building
# Same pattern as MLB profile weights — 2023 is the earliest available year
AAA_RECENCY_WEIGHTS = {
    2026: 3.0,
    2025: 2.5,
    2024: 2.0,
    2023: 1.5,
}

# Sample weights for profile building (by data recency).
# Mapped to the same key names as MLB ERA_SAMPLE_WEIGHTS for the pitch-clock era.
AAA_SAMPLE_WEIGHTS = {
    2026: ERA_SAMPLE_WEIGHTS["pitch_clock_2026"],   # 2.0
    2025: ERA_SAMPLE_WEIGHTS["pitch_clock_2025"],   # 1.8
    2024: ERA_SAMPLE_WEIGHTS["pitch_clock_2024"],   # 1.5
    2023: ERA_SAMPLE_WEIGHTS["pitch_clock_2023"],   # 1.2
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ABS (Automated Ball-Strike) descriptions to their MLB equivalents.
    'automatic_ball' → 'ball', 'automatic_strike' → 'called_strike'.
    profiles.py's swing/contact/whiff sets don't include ABS codes, so
    these rows would otherwise be silently dropped from rate calculations.
    """
    before = df["description"].value_counts().get("automatic_ball", 0) + \
             df["description"].value_counts().get("automatic_strike", 0)
    df["description"] = df["description"].replace(ABS_DESCRIPTION_MAP)
    print(f"  ABS descriptions normalized: {before:,} rows remapped")
    return df


def make_game_year(df: pd.DataFrame) -> pd.DataFrame:
    df["game_year"] = df["game_date"].dt.year.astype(int)
    return df


def make_sample_weight(df: pd.DataFrame) -> pd.Series:
    """Per-row sample weight based on data recency year."""
    year = df["game_year"] if "game_year" in df.columns else df["game_date"].dt.year
    default = min(AAA_SAMPLE_WEIGHTS.values())
    return year.map(lambda y: AAA_SAMPLE_WEIGHTS.get(y, default))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unusable rows. No 2020 exclusion (no AAA Statcast before 2023).
    No pitcher-batter filter (universal DH at all levels since 2022).
    """
    before = len(df)

    df["game_date"] = pd.to_datetime(df["game_date"])

    # Keep only regular-season games
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]

    # Drop rows missing critical pitch attributes
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(["UN", ""])]
    df = df[df["release_speed"].notna()]
    df = df[df["plate_x"].notna() & df["plate_z"].notna()]

    # Deduplicate
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    print(f"  Cleaned: {before:,} → {len(df):,} rows  ({before - len(df):,} dropped)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(raw_df: pd.DataFrame, output_path: Path = OUTPUT_PARQUET) -> pd.DataFrame:
    """
    Clean and prepare AAA Statcast data for profile building.
    Does NOT build hitter profiles — that is Phase 4 (profiles.py).

    Returns the processed DataFrame.
    """
    print(f"\nAAA preprocessing: {len(raw_df):,} raw rows")

    df = clean(raw_df)
    df = make_game_year(df)
    df = normalize_descriptions(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        df["sample_weight"] = make_sample_weight(df)

    # Year breakdown
    year_counts = df["game_year"].value_counts().sort_index()
    print("  Rows by year:")
    for yr, n in year_counts.items():
        print(f"    {yr}: {n:,}")

    # Pitch type distribution
    pt_counts = df["pitch_type"].value_counts()
    print(f"  Pitch types (top 10): {dict(pt_counts.head(10))}")

    # Batter count
    n_batters = df["batter"].nunique()
    print(f"  Unique batters: {n_batters:,}")

    # Keep only the columns profiles.py needs + extras we added (deduplicated, ordered)
    seen = set()
    keep = []
    for c in PROFILE_COLUMNS + ["sample_weight"]:
        if c in df.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    df = df[keep].copy()  # copy() resolves pandas fragmentation warning

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n  Saved → {output_path}  ({len(df):,} rows, {len(df.columns)} columns)")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.fetch_minors import load_raw_minors

    raw = load_raw_minors()
    run_preprocessing(raw)
