"""
features.py — Build pitcher-level aggregate features from Statcast training data.

For each unique pitcher computes average release height, horizontal release point,
extension, fastball velocity, and a derived arm-slot angle.

These are identity features — they capture who is throwing (arm slot, deception,
approach angle) rather than what a specific pitch did.

Usage:
    python -m src.pitchers.features           # builds and saves parquet
    from src.pitchers.features import build_pitcher_features, load_pitcher_features
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

PITCHER_FEATURES_PATH = Path("data/processed/pitcher_features.parquet")
DATE_CUTOFF = "2025-01-01"
MIN_PITCHES = 100   # below this, fill with population medians

FASTBALL_TYPES = {"FF", "SI"}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_pitcher_features(
    df: pd.DataFrame,
    date_cutoff: str = DATE_CUTOFF,
    save_path: Path = PITCHER_FEATURES_PATH,
) -> pd.DataFrame:
    """
    Compute aggregate pitcher features from pitches before date_cutoff.

    Args:
        df: full processed DataFrame with pitcher, game_date, release_pos_z,
            release_pos_x, release_extension, release_speed, pitch_type columns.
        date_cutoff: ISO date string — only pitches before this date are used.
        save_path: where to save the parquet; None to skip saving.

    Returns:
        DataFrame indexed by pitcher MLBAM ID with 5 feature columns.
    """
    cutoff = pd.Timestamp(date_cutoff)
    train = df[df["game_date"] < cutoff].copy()

    # --- per-pitcher aggregates ---
    agg = train.groupby("pitcher").agg(
        pitcher_avg_release_pos_z=("release_pos_z", "mean"),
        pitcher_avg_release_pos_x=("release_pos_x", "mean"),
        pitcher_avg_extension=("release_extension", "mean"),
        n_pitches=("release_speed", "count"),
    ).reset_index()

    # Fastball speed: mean of FF/SI only; fall back to overall mean if none
    fb = train[train["pitch_type"].isin(FASTBALL_TYPES)]
    fb_speed = fb.groupby("pitcher")["release_speed"].mean().rename("pitcher_avg_speed")
    overall_speed = train.groupby("pitcher")["release_speed"].mean().rename("_overall_speed")

    agg = agg.merge(fb_speed, on="pitcher", how="left")
    agg = agg.merge(overall_speed, on="pitcher", how="left")
    agg["pitcher_avg_speed"] = agg["pitcher_avg_speed"].fillna(agg["_overall_speed"])
    agg = agg.drop(columns=["_overall_speed"])

    # Arm slot angle: arctan(release_pos_x / release_pos_z) in degrees
    # Positive = first-base side (right-hander norm), negative = third-base side
    agg["pitcher_slot_angle"] = np.degrees(
        np.arctan2(agg["pitcher_avg_release_pos_x"], agg["pitcher_avg_release_pos_z"])
    )

    # Fill thin-sample pitchers with population medians
    feature_cols = [
        "pitcher_avg_release_pos_z",
        "pitcher_avg_release_pos_x",
        "pitcher_avg_extension",
        "pitcher_avg_speed",
        "pitcher_slot_angle",
    ]
    medians = agg[feature_cols].median()
    thin_mask = agg["n_pitches"] < MIN_PITCHES
    n_thin = thin_mask.sum()
    if n_thin > 0:
        agg.loc[thin_mask, feature_cols] = medians.values
        print(f"  Filled {n_thin:,} thin-sample pitchers with population medians.")

    result = agg[["pitcher"] + feature_cols]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(save_path, index=False)
        print(f"  Pitcher features saved to {save_path}  ({len(result):,} pitchers)")

    return result


# ---------------------------------------------------------------------------
# Load / lookup
# ---------------------------------------------------------------------------

def load_pitcher_features(path: Path = PITCHER_FEATURES_PATH) -> pd.DataFrame:
    """Load saved pitcher features parquet, indexed by pitcher ID."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pitcher features not found at {path}. Run build_pitcher_features() first."
        )
    return pd.read_parquet(path)


def get_pitcher_feature_row(
    pitcher_id: int,
    pitcher_df: pd.DataFrame,
) -> dict:
    """
    Return the 5 pitcher features as a dict for a given MLBAM pitcher ID.
    Falls back to population medians if the pitcher is not found.
    """
    feature_cols = [
        "pitcher_avg_release_pos_z",
        "pitcher_avg_release_pos_x",
        "pitcher_avg_extension",
        "pitcher_avg_speed",
        "pitcher_slot_angle",
    ]
    row = pitcher_df[pitcher_df["pitcher"] == pitcher_id]
    if len(row) == 0:
        medians = pitcher_df[feature_cols].median()
        return medians.to_dict()
    return row[feature_cols].iloc[0].to_dict()


def get_median_pitcher_features(pitcher_df: pd.DataFrame) -> dict:
    """Return population-median pitcher features (used when pitcher_id is unknown)."""
    feature_cols = [
        "pitcher_avg_release_pos_z",
        "pitcher_avg_release_pos_x",
        "pitcher_avg_extension",
        "pitcher_avg_speed",
        "pitcher_slot_angle",
    ]
    return pitcher_df[feature_cols].median().to_dict()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building pitcher aggregate features...")
    df = pd.read_parquet("data/processed/statcast_processed.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    pf = build_pitcher_features(df)
    print(f"\nSample (5 rows):")
    print(pf.sample(5, random_state=42).to_string(index=False))
    print(f"\nPopulation medians:")
    print(pf.drop(columns="pitcher").median().to_string())
