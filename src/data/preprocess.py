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
    "SV": 13,  # slurve (2023+ Statcast code; pre-2023 SV = sweeper → use ST)
    "FIRST_PITCH": 14,  # sentinel: no previous pitch
    "ST": 16,  # sweeper (2023+ Statcast code; pre-2023 sweepers were labeled SV)
    "OTHER": 17,        # catch-all for unmapped types
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

# Universal DH started 2022 — pitcher-batting filter only applies before this
DH_ERA_START = pd.Timestamp("2022-01-01")
# Fraction of (pitcher + batter) appearances that must be as pitcher to be flagged
PITCHER_BATTER_PCT_THRESHOLD = 0.80

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
    "times_through_order",
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

HITTER_PITCH_TYPE_FEATURES = [
    "hitter_contact_rate_FF",
    "hitter_contact_rate_CH",
    "hitter_contact_rate_CU",
    "hitter_contact_rate_FC",
    "hitter_contact_rate_KN",
    "hitter_contact_rate_SC",
    "hitter_contact_rate_SI",
    "hitter_contact_rate_SL",
    "hitter_contact_rate_SV",
    "hitter_contact_rate_FS",
    "hitter_contact_rate_ST",
]

HITTER_ZONE_FEATURES = [
    "hitter_swing_rate_z1",  "hitter_swing_rate_z2",  "hitter_swing_rate_z3",
    "hitter_swing_rate_z4",  "hitter_swing_rate_z5",  "hitter_swing_rate_z6",
    "hitter_swing_rate_z7",  "hitter_swing_rate_z8",  "hitter_swing_rate_z9",
    "hitter_swing_rate_z11", "hitter_swing_rate_z12",
    "hitter_swing_rate_z13", "hitter_swing_rate_z14",
    "hitter_whiff_rate_z1",  "hitter_whiff_rate_z2",  "hitter_whiff_rate_z3",
    "hitter_whiff_rate_z4",  "hitter_whiff_rate_z5",  "hitter_whiff_rate_z6",
    "hitter_whiff_rate_z7",  "hitter_whiff_rate_z8",  "hitter_whiff_rate_z9",
    "hitter_whiff_rate_z11", "hitter_whiff_rate_z12",
    "hitter_whiff_rate_z13", "hitter_whiff_rate_z14",
]

HITTER_FAMILY_FEATURES = [
    "hitter_swing_rate_fastball", "hitter_swing_rate_breaking",
    "hitter_swing_rate_offspeed", "hitter_swing_rate_other",
    "hitter_whiff_rate_fastball", "hitter_whiff_rate_breaking",
    "hitter_whiff_rate_offspeed", "hitter_whiff_rate_other",
]

PITCHER_FEATURES = [
    "pitcher_avg_release_pos_z",
    "pitcher_avg_release_pos_x",
    "pitcher_avg_extension",
    "pitcher_avg_speed",
    "pitcher_slot_angle",
]

HITTER_CONTEXTUAL_FEATURES = [
    # Resolved from each pitch's zone/family/count — replace routing logic XGBoost must learn
    "hitter_swing_rate_this_zone",
    "hitter_whiff_rate_this_zone",
    "hitter_swing_rate_this_family",
    "hitter_whiff_rate_this_family",
    "hitter_swing_rate_this_count",
]

# Per-pitch context features resolved directly from profile dicts (not expanded static columns).
# These are distinct from HITTER_CONTEXTUAL_FEATURES above:
#   - zone_xwoba / zone_hard_hit_rate: damage quality signals not in the old group
#   - contact_rate_this_pitch: single pitch_type lookup (vs 11-column expansion)
#   - zone/family swing/whiff: same semantic content, different computation path
# hitter_swing_rate_this_count is intentionally omitted here — it already appears in
# HITTER_CONTEXTUAL_FEATURES; duplicating it in ALL_FEATURES would create duplicate
# DataFrame columns at training time.
HITTER_CONTEXT_FEATURES = [
    "hitter_zone_swing_rate",
    "hitter_zone_whiff_rate",
    "hitter_zone_xwoba",
    "hitter_zone_hard_hit_rate",
    "hitter_contact_rate_this_pitch",
    "hitter_family_swing_rate",
    "hitter_family_whiff_rate",
]

# Hitter-calibrated pitch physics scores (KNN similarity over historical swings).
# Appended last so existing feature indices are unchanged during incremental retrains.
STUFF_VS_HITTER_FEATURES = [
    "stuff_vs_hitter_xwoba",
    "stuff_vs_hitter_whiff",
]

# ALL_FEATURES: 80 (baseline) + 7 (HITTER_CONTEXT_FEATURES) + 2 (STUFF_VS_HITTER) = 89 total.
# The model currently trained on 80 features uses feature_cols.json to select
# its subset — adding new columns here does not break inference until retrain.
ALL_FEATURES = (
    PITCH_FEATURES
    + CONTEXT_FEATURES
    + HITTER_PROFILE_FEATURES
    + HITTER_PITCH_TYPE_FEATURES
    + HITTER_ZONE_FEATURES
    + HITTER_FAMILY_FEATURES
    + HITTER_CONTEXTUAL_FEATURES
    + HITTER_CONTEXT_FEATURES
    + STUFF_VS_HITTER_FEATURES
    # PITCHER_FEATURES intentionally excluded — disconnected from model pipeline.
    # Keep src/pitchers/features.py and the parquet file; re-add here to re-enable.
    # + PITCHER_FEATURES
)


# ---------------------------------------------------------------------------
# Three-stage behavioral target descriptions
# ---------------------------------------------------------------------------

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

IN_PLAY_DESCRIPTIONS = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}


# ---------------------------------------------------------------------------
# Pitcher-batter identification and removal
# ---------------------------------------------------------------------------

def identify_pitcher_batters(
    df: pd.DataFrame,
    pct_threshold: float = PITCHER_BATTER_PCT_THRESHOLD,
) -> frozenset:
    """
    Return a frozenset of MLBAM player IDs who are primarily pitchers.

    A player is flagged if ≥pct_threshold of their combined (as pitcher +
    as batter) row appearances in pre-2022 data are in the pitcher column.
    Only examines pre-2022 rows because universal DH started in 2022.
    """
    pre_dh = df[df["game_date"] < DH_ERA_START]
    as_pitcher = pre_dh.groupby("pitcher").size().rename("n_pitcher")
    as_batter  = pre_dh.groupby("batter").size().rename("n_batter")
    combined = (
        pd.DataFrame({"n_pitcher": as_pitcher, "n_batter": as_batter})
        .fillna(0)
    )
    combined["pct_pitcher"] = combined["n_pitcher"] / (combined["n_pitcher"] + combined["n_batter"])
    flagged = combined[
        (combined["pct_pitcher"] >= pct_threshold) & (combined["n_batter"] > 0)
    ]
    return frozenset(flagged.index.astype(int))


def drop_pitcher_batting_rows(
    df: pd.DataFrame,
    pitcher_batter_ids: frozenset,
) -> pd.DataFrame:
    """
    Remove pre-2022 rows where the batter is a flagged pitcher-batter.
    Post-2022 rows are never touched (universal DH — everyone batting is a real hitter).
    """
    before = len(df)
    mask = (df["game_date"] < DH_ERA_START) & (df["batter"].isin(pitcher_batter_ids))
    n_removed = int(mask.sum())
    df = df[~mask].reset_index(drop=True)
    pct = n_removed / before * 100
    print(
        f"Pitcher-batting rows removed: {n_removed:,} "
        f"({pct:.2f}% of {before:,} rows)"
    )
    print(
        f"  Flagged {len(pitcher_batter_ids):,} pitcher-batters "
        f"(≥{PITCHER_BATTER_PCT_THRESHOLD:.0%} of pre-2022 appearances as pitcher)"
    )
    return df


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

    # Remove pitcher-batting PAs (pre-2022 NL only). Identify from this slice of data
    # before any rows are lost, so the fraction calculation is stable.
    pitcher_batter_ids = identify_pitcher_batters(df)
    df = drop_pitcher_batting_rows(df, pitcher_batter_ids)

    print(f"Cleaned: {before:,} → {len(df):,} rows")
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
# Times through the order
# ---------------------------------------------------------------------------

def make_times_through_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how many times the batter has faced this pitcher in this game.

    Groups by (game_pk, batter, pitcher). Within each group, each unique
    at_bat_number represents one plate appearance. The first PA = 1, second = 2,
    third+ = 3 (capped at 3 to avoid sparse high-TTO cells).

    Stored as integer column "times_through_order".
    """
    # Rank each unique at_bat_number within the (game, batter, pitcher) group.
    # We use at_bat_number directly — lower value = earlier in game.
    ab_rank = (
        df[["game_pk", "batter", "pitcher", "at_bat_number"]]
        .drop_duplicates(subset=["game_pk", "batter", "pitcher", "at_bat_number"])
        .copy()
    )
    ab_rank["tto"] = (
        ab_rank.groupby(["game_pk", "batter", "pitcher"])["at_bat_number"]
        .rank(method="dense")
        .astype(int)
        .clip(upper=3)
    )

    df = df.merge(
        ab_rank[["game_pk", "batter", "pitcher", "at_bat_number", "tto"]],
        on=["game_pk", "batter", "pitcher", "at_bat_number"],
        how="left",
    )
    df["times_through_order"] = df["tto"].fillna(1).astype(int)
    df.drop(columns=["tto"], inplace=True)

    dist = df["times_through_order"].value_counts().sort_index()
    print(f"times_through_order distribution: {dist.to_dict()}")
    return df


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
    """Temporal split: train < 2025, test >= 2025."""
    cutoff = pd.Timestamp("2025-01-01")
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
# Contextual hitter feature resolution
# ---------------------------------------------------------------------------

def add_contextual_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve 5 per-pitch contextual hitter features from the matched zone,
    pitch family, and count for each row.

    Requires the static zone/family/count columns already merged from profiles:
      - hitter_swing_rate_z{1-9,11-14}, hitter_whiff_rate_z{1-9,11-14}
      - hitter_swing_rate_{fastball,breaking,offspeed,other}
      - hitter_whiff_rate_{fastball,breaking,offspeed,other}
      - hitter_swing_rate_count_{b}_{s}  (for b in 0-3, s in 0-2)

    Output columns (appended):
      hitter_swing_rate_this_zone
      hitter_whiff_rate_this_zone
      hitter_swing_rate_this_family
      hitter_whiff_rate_this_family
      hitter_swing_rate_this_count
    """
    from src.hitters.profiles import ALL_ZONES, PITCH_FAMILY_LIST, PITCH_FAMILIES

    # Build pitch_type → family lookup
    pitch_type_to_family = {}
    for family, types in PITCH_FAMILIES.items():
        for pt in types:
            pitch_type_to_family[pt] = family

    df = df.copy()

    # ── Zone features ──
    # Fallback: overall swing/whiff rate for rows with NaN zone
    df["hitter_swing_rate_this_zone"] = df["hitter_swing_rate"].copy()
    df["hitter_whiff_rate_this_zone"] = df["hitter_whiff_rate"].copy()

    zone_col = df["zone"] if "zone" in df.columns else pd.Series(np.nan, index=df.index)
    for z in ALL_ZONES:
        mask = zone_col == z
        if mask.any():
            df.loc[mask, "hitter_swing_rate_this_zone"] = df.loc[mask, f"hitter_swing_rate_z{z}"]
            df.loc[mask, "hitter_whiff_rate_this_zone"] = df.loc[mask, f"hitter_whiff_rate_z{z}"]

    # ── Pitch-family features ──
    df["_pitch_family"] = df["pitch_type"].map(pitch_type_to_family).fillna("other")
    df["hitter_swing_rate_this_family"] = df["hitter_swing_rate"].copy()
    df["hitter_whiff_rate_this_family"] = df["hitter_whiff_rate"].copy()

    for fam in PITCH_FAMILY_LIST:
        mask = df["_pitch_family"] == fam
        if mask.any():
            df.loc[mask, "hitter_swing_rate_this_family"] = df.loc[mask, f"hitter_swing_rate_{fam}"]
            df.loc[mask, "hitter_whiff_rate_this_family"] = df.loc[mask, f"hitter_whiff_rate_{fam}"]
    df.drop(columns=["_pitch_family"], inplace=True)

    # ── Count feature ──
    df["hitter_swing_rate_this_count"] = df["hitter_swing_rate"].copy()
    for b in range(4):
        for s in range(3):
            mask = (df["balls"] == b) & (df["strikes"] == s)
            if mask.any():
                col = f"hitter_swing_rate_count_{b}_{s}"
                if col in df.columns:
                    df.loc[mask, "hitter_swing_rate_this_count"] = df.loc[mask, col]

    n_zone_filled = (zone_col.notna()).sum()
    n_total = len(df)
    print(
        f"Contextual hitter features resolved: "
        f"{n_zone_filled:,}/{n_total:,} rows have zone data "
        f"({n_zone_filled/n_total*100:.1f}%)"
    )
    return df


# ---------------------------------------------------------------------------
# Three-stage behavioral target construction
# ---------------------------------------------------------------------------

def make_three_stage_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary target columns for three-stage hitter behavior modeling plus
    a continuous xwOBA regression target for in-play contact.

    Stage 1 — target_swing:  did the hitter swing? (all pitches)
    Stage 2 — target_contact: did they make contact? (meaningful on swings only)
    Stage 3 — target_hard_contact: was it hard hit ≥ 95 mph? (meaningful on in-play only)
    Stage 4 — target_xwoba_on_contact: continuous xwOBA (in-play rows only; NaN elsewhere)

    The existing 'hit' and 'xwoba' columns are preserved unchanged.
    """
    df["target_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    df["target_contact"] = df["description"].isin(CONTACT_DESCRIPTIONS).astype(int)
    df["target_hard_contact"] = (
        (df["launch_speed"].fillna(0) >= 95)
        & df["description"].isin(IN_PLAY_DESCRIPTIONS)
    ).astype(int)

    # xwOBA regression target — NaN preserved for non-in-play rows.
    # Do NOT fill NaN with 0: drop NaN rows at training time instead.
    df["target_xwoba_on_contact"] = df["estimated_woba_using_speedangle"]
    contact_mask = df["description"].isin(IN_PLAY_DESCRIPTIONS)
    n_contact = int(contact_mask.sum())
    n_valid   = int(df.loc[contact_mask, "target_xwoba_on_contact"].notna().sum())
    n_nan     = n_contact - n_valid

    swing_rate   = df["target_swing"].mean()
    contact_rate = df[df["target_swing"] == 1]["target_contact"].mean()
    hc_rate      = df[df["target_contact"] == 1]["target_hard_contact"].mean()
    print(
        f"Three-stage targets: swing={swing_rate:.3f}, "
        f"contact|swing={contact_rate:.3f}, hard|contact={hc_rate:.3f}"
    )
    print(
        f"xwOBA on contact: {n_valid:,} valid / {n_nan:,} NaN "
        f"out of {n_contact:,} in-play rows"
    )
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(raw_df: pd.DataFrame) -> tuple:
    """
    End-to-end preprocessing, including hitter profile building and merging.
    Returns (full_df, train_df, test_df).
    Pipeline: clean → pitch features → three-stage targets → hitter profiles → split & save
    Pitcher features are disconnected from the pipeline (commented out).
    """
    df = clean(raw_df)
    df = make_prev_pitch_features(df)
    df = make_times_through_order(df)
    df = impute_pitcher_medians(df)

    df["hit"] = make_target(df)
    df["xwoba"] = df["estimated_woba_using_speedangle"].fillna(0.0)
    df["score_diff"] = make_score_diff(df)

    # Three-stage behavioral targets (added after cleaning so description column is intact)
    df = make_three_stage_targets(df)

    df = encode_pitch_types(df)
    df = encode_prev_result(df)
    df = encode_handedness(df)
    df = make_runner_flags(df)

    # Era integrity features
    df["era_flag_enc"] = make_era_flag(df)
    df["spin_to_velo_ratio"] = make_spin_to_velo_ratio(df)
    df["sample_weight"] = make_sample_weights(df)

    # Pitcher identity features — disconnected from model pipeline.
    # src/pitchers/features.py and the parquet file are preserved.
    # Uncomment this block to re-enable pitcher features in preprocessing.
    # from src.pitchers.features import (
    #     PITCHER_FEATURES_PATH, build_pitcher_features, load_pitcher_features
    # )
    # if not PITCHER_FEATURES_PATH.exists():
    #     print("Building pitcher features...")
    #     pitcher_df = build_pitcher_features(df)
    # else:
    #     pitcher_df = load_pitcher_features()
    # df = df.drop(columns=[c for c in PITCHER_FEATURES if c in df.columns])
    # df = df.merge(pitcher_df, on="pitcher", how="left")
    # for col in PITCHER_FEATURES:
    #     df[col] = df[col].fillna(pitcher_df[col].median())

    # Hitter identity features — build/merge all hitter profiles from pre-cutoff data
    print("Building and merging hitter profiles...")
    from src.hitters.profiles import merge_profiles_into_df, add_hitter_context_features
    df = merge_profiles_into_df(df)

    # Contextual matched features: resolve zone/family/count for each pitch row
    print("Resolving contextual hitter features...")
    df = add_contextual_hitter_features(df)

    # Per-pitch context features from profile dicts (7 new HITTER_CONTEXT_FEATURES)
    print("Resolving per-pitch hitter context features...")
    df = add_hitter_context_features(df)

    # Hitter-calibrated physics scores (KNN over historical swings)
    print("Computing stuff_vs_hitter features...")
    from src.model.stuff_vs_hitter import add_stuff_vs_hitter_features
    df = add_stuff_vs_hitter_features(df)

    train, test = split_data(df)
    save_processed(df, train, test)
    return df, train, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Preprocess Statcast data")
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help=(
            "Filter to DEMO_HITTERS before add_hitter_context_features, "
            "save to data/processed/statcast_demo_context.parquet. "
            "Does NOT overwrite statcast_processed.parquet."
        ),
    )
    args = parser.parse_args()

    if args.demo_only:
        import pandas as pd
        from src.hitters.profiles import DEMO_HITTERS, add_hitter_context_features

        t0 = time.time()
        print("Loading full processed parquet...")
        df = pd.read_parquet("data/processed/statcast_processed.parquet")

        demo_ids = set(DEMO_HITTERS.values())
        demo_df = df[df["batter"].isin(demo_ids)].copy().reset_index(drop=True)
        print(f"Demo subset: {len(demo_df):,} rows for {len(demo_ids)} hitters")

        print("Resolving per-pitch hitter context features...")
        demo_df = add_hitter_context_features(demo_df)

        print("Computing stuff_vs_hitter features...")
        from src.model.stuff_vs_hitter import add_stuff_vs_hitter_features
        demo_df = add_stuff_vs_hitter_features(demo_df)

        out_path = "data/processed/statcast_demo_context.parquet"
        demo_df.to_parquet(out_path, index=False)
        elapsed = time.time() - t0
        print(f"\nSaved {out_path}  ({elapsed:.1f}s)")

        # NaN check across all new feature columns
        new_cols = [
            "hitter_zone_swing_rate", "hitter_zone_whiff_rate",
            "hitter_zone_xwoba", "hitter_zone_hard_hit_rate",
            "hitter_contact_rate_this_pitch",
            "hitter_family_swing_rate", "hitter_family_whiff_rate",
            "stuff_vs_hitter_xwoba", "stuff_vs_hitter_whiff",
        ]
        nan_counts = demo_df[new_cols].isna().sum()
        if nan_counts.any():
            print(f"\nWARN: NaN counts in new columns:\n{nan_counts[nan_counts > 0]}")
        else:
            print(f"\nNo NaN values in any of the {len(new_cols)} new feature columns.")

        print("\nPer-hitter mean of new context + stuff features:")
        name_map = {v: k for k, v in DEMO_HITTERS.items()}
        summary = (
            demo_df.groupby("batter")[new_cols]
            .mean()
            .rename(index=name_map)
        )
        print(summary.to_string(float_format="{:.3f}".format))
        print(f"\nElapsed: {elapsed:.1f}s")
    else:
        from src.data.fetch import load_raw

        raw = load_raw()
        run_preprocessing(raw)
