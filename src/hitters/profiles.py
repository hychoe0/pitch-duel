"""
profiles.py — Build and manage per-hitter statistical profiles.

Each profile encodes swing/chase/contact/hard-hit/whiff rates computed
from the hitter's Statcast PA history (training data only — no leakage).

Usage:
    python -m src.hitters.profiles          # build profiles for all batters
    python -m src.hitters.profiles --name "Shohei Ohtani"
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_DIR = Path("data/processed/profiles")
MIN_WEIGHTED_PA = 200  # weighted plate appearances; below this, blend with league averages

# Standard strike zone bounds (universal approximation)
ZONE_X_MIN, ZONE_X_MAX = -0.85, 0.85  # feet from center of plate
ZONE_Z_MIN, ZONE_Z_MAX = 1.5, 3.5     # feet from ground

# League-average fallback values (2015-2022 MLB)
LEAGUE_AVG = {
    "swing_rate": 0.47,
    "chase_rate": 0.30,
    "contact_rate": 0.78,
    "hard_hit_rate": 0.38,
    "whiff_rate": 0.25,
}

# Recency weights for hitter profiles — aggressive toward pitch-clock era
PROFILE_RECENCY_WEIGHTS = {
    2026: 3.0,
    2025: 2.5,
    2024: 2.0,
    2023: 1.5,
    2022: 1.0,
}
PROFILE_RECENCY_DEFAULT = 0.5  # 2021 and earlier


def _row_weights(df: pd.DataFrame) -> np.ndarray:
    """Return per-row recency weights based on game_date year."""
    years = df["game_date"].dt.year
    return years.map(lambda y: PROFILE_RECENCY_WEIGHTS.get(y, PROFILE_RECENCY_DEFAULT)).values


def _weighted_pa_count(df: pd.DataFrame) -> float:
    """
    Count weighted plate appearances (last pitch of each PA, weighted by recency).
    A PA is identified by (game_pk, at_bat_number).
    """
    last_pitches = df.groupby(["game_pk", "at_bat_number"]).last().reset_index()
    weights = _row_weights(last_pitches)
    return float(weights.sum())


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class HitterProfile:
    player_id: int
    player_name: str
    stand: str                       # 'L' or 'R'
    swing_rate: float
    chase_rate: float
    contact_rate: float
    hard_hit_rate: float
    whiff_rate: float
    swing_rate_by_count: dict = field(default_factory=dict)   # (balls, strikes) -> float
    contact_rate_by_pitch_type: dict = field(default_factory=dict)  # pitch_type_str -> float
    sample_size: int = 0
    is_thin_sample: bool = False


# ---------------------------------------------------------------------------
# Swing / outcome helpers
# ---------------------------------------------------------------------------

_SWING_DESCRIPTIONS = {
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

_CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

_WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
}


def _is_swing(desc: pd.Series) -> pd.Series:
    return desc.isin(_SWING_DESCRIPTIONS)


def _is_contact(desc: pd.Series) -> pd.Series:
    return desc.isin(_CONTACT_DESCRIPTIONS)


def _is_whiff(desc: pd.Series) -> pd.Series:
    return desc.isin(_WHIFF_DESCRIPTIONS)


def _is_out_of_zone(df: pd.DataFrame) -> pd.Series:
    ox = (df["plate_x"] < ZONE_X_MIN) | (df["plate_x"] > ZONE_X_MAX)
    oz = (df["plate_z"] < ZONE_Z_MIN) | (df["plate_z"] > ZONE_Z_MAX)
    return ox | oz


# ---------------------------------------------------------------------------
# Stat computers
# ---------------------------------------------------------------------------

def compute_swing_rate(df: pd.DataFrame, w: np.ndarray) -> float:
    swings = _is_swing(df["description"]).values
    return float(np.average(swings, weights=w)) if w.sum() > 0 else LEAGUE_AVG["swing_rate"]


def compute_chase_rate(df: pd.DataFrame, w: np.ndarray) -> float:
    ooz = _is_out_of_zone(df).values
    w_ooz = w * ooz
    if w_ooz.sum() == 0:
        return LEAGUE_AVG["chase_rate"]
    swings = _is_swing(df["description"]).values
    return float((w * ooz * swings).sum() / w_ooz.sum())


def compute_contact_rate(df: pd.DataFrame, w: np.ndarray) -> float:
    swings = _is_swing(df["description"]).values
    w_swing = w * swings
    if w_swing.sum() == 0:
        return LEAGUE_AVG["contact_rate"]
    contacts = _is_contact(df["description"]).values
    return float((w * swings * contacts).sum() / w_swing.sum())


def compute_hard_hit_rate(df: pd.DataFrame, w: np.ndarray) -> float:
    batted = df["description"].isin({"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}).values
    w_batted = w * batted
    if w_batted.sum() == 0:
        return LEAGUE_AVG["hard_hit_rate"]
    hard = (batted & (df["launch_speed"].fillna(0) >= 95).values)
    return float((w * hard).sum() / w_batted.sum())


def compute_whiff_rate(df: pd.DataFrame, w: np.ndarray) -> float:
    swings = _is_swing(df["description"]).values
    w_swing = w * swings
    if w_swing.sum() == 0:
        return LEAGUE_AVG["whiff_rate"]
    whiffs = _is_whiff(df["description"]).values
    return float((w * whiffs).sum() / w_swing.sum())


def compute_swing_rate_by_count(df: pd.DataFrame, w: np.ndarray) -> dict:
    result = {}
    for (balls, strikes), idx in df.groupby(["balls", "strikes"]).groups.items():
        grp = df.loc[idx]
        grp_w = w[df.index.get_indexer(idx)]
        result[f"{balls}-{strikes}"] = compute_swing_rate(grp, grp_w)
    return result


def compute_contact_rate_by_pitch_type(df: pd.DataFrame, w: np.ndarray) -> dict:
    result = {}
    for pitch_type, idx in df.groupby("pitch_type").groups.items():
        grp = df.loc[idx]
        grp_w = w[df.index.get_indexer(idx)]
        result[str(pitch_type)] = compute_contact_rate(grp, grp_w)
    return result


# ---------------------------------------------------------------------------
# Thin-sample blending
# ---------------------------------------------------------------------------

def _blend(computed: float, league_avg: float, weighted_pa: float) -> float:
    """Blend computed rate toward league average when weighted PA count is thin."""
    weight = min(weighted_pa / MIN_WEIGHTED_PA, 1.0)
    return weight * computed + (1.0 - weight) * league_avg


def _blend_profile(profile: HitterProfile, weighted_pa: float) -> HitterProfile:
    profile.swing_rate = _blend(profile.swing_rate, LEAGUE_AVG["swing_rate"], weighted_pa)
    profile.chase_rate = _blend(profile.chase_rate, LEAGUE_AVG["chase_rate"], weighted_pa)
    profile.contact_rate = _blend(profile.contact_rate, LEAGUE_AVG["contact_rate"], weighted_pa)
    profile.hard_hit_rate = _blend(profile.hard_hit_rate, LEAGUE_AVG["hard_hit_rate"], weighted_pa)
    profile.whiff_rate = _blend(profile.whiff_rate, LEAGUE_AVG["whiff_rate"], weighted_pa)
    return profile


# ---------------------------------------------------------------------------
# Player ID lookup
# ---------------------------------------------------------------------------

def get_player_id(name: str, df: pd.DataFrame) -> int:
    """
    Look up MLBAM batter ID by name.
    Tries exact match first, then case-insensitive substring.
    Raises ValueError listing closest matches if not found.
    """
    if "player_name" not in df.columns:
        raise KeyError("DataFrame missing 'player_name' column.")

    exact = df[df["player_name"] == name]["batter"].unique()
    if len(exact) == 1:
        return int(exact[0])
    if len(exact) > 1:
        return int(exact[0])  # take first (all rows share the same player_id)

    # Case-insensitive substring match
    lower = name.lower()
    candidates = df[df["player_name"].str.lower().str.contains(lower, na=False)]["player_name"].unique()
    if len(candidates) == 0:
        raise ValueError(f"Player '{name}' not found. No similar names in dataset.")
    raise ValueError(f"Player '{name}' not found. Did you mean one of: {list(candidates[:5])}")


# ---------------------------------------------------------------------------
# Build profile
# ---------------------------------------------------------------------------

def build_profile(
    player_id: int,
    df: pd.DataFrame,
    date_cutoff: str = "2023-01-01",
) -> HitterProfile:
    """
    Build a HitterProfile from the player's pitches before date_cutoff.
    Thin-sample blending applied automatically.
    """
    cutoff = pd.Timestamp(date_cutoff)
    sub = df[(df["batter"] == player_id) & (df["game_date"] < cutoff)].copy()

    # Resolve player name
    name = "Unknown"
    if "player_name" in sub.columns and len(sub) > 0:
        name = sub["player_name"].iloc[0]

    stand = sub["stand"].mode().iloc[0] if len(sub) > 0 else "R"
    n = len(sub)

    w = _row_weights(sub) if n > 0 else np.array([])
    weighted_pa = _weighted_pa_count(sub) if n > 0 else 0.0
    is_thin = weighted_pa < MIN_WEIGHTED_PA

    profile = HitterProfile(
        player_id=player_id,
        player_name=name,
        stand=stand,
        swing_rate=compute_swing_rate(sub, w) if n > 0 else LEAGUE_AVG["swing_rate"],
        chase_rate=compute_chase_rate(sub, w) if n > 0 else LEAGUE_AVG["chase_rate"],
        contact_rate=compute_contact_rate(sub, w) if n > 0 else LEAGUE_AVG["contact_rate"],
        hard_hit_rate=compute_hard_hit_rate(sub, w) if n > 0 else LEAGUE_AVG["hard_hit_rate"],
        whiff_rate=compute_whiff_rate(sub, w) if n > 0 else LEAGUE_AVG["whiff_rate"],
        swing_rate_by_count=compute_swing_rate_by_count(sub, w) if n > 0 else {},
        contact_rate_by_pitch_type=compute_contact_rate_by_pitch_type(sub, w) if n > 0 else {},
        sample_size=n,
        is_thin_sample=is_thin,
    )

    if is_thin:
        profile = _blend_profile(profile, weighted_pa)

    return profile


# ---------------------------------------------------------------------------
# Feature dict for model input
# ---------------------------------------------------------------------------

def profile_to_feature_dict(profile: HitterProfile) -> dict:
    # hitter_stand_enc is intentionally excluded here — it is already set
    # per-row by encode_handedness() in preprocessing from the actual stand column.
    return {
        "hitter_swing_rate": profile.swing_rate,
        "hitter_chase_rate": profile.chase_rate,
        "hitter_contact_rate": profile.contact_rate,
        "hitter_hard_hit_rate": profile.hard_hit_rate,
        "hitter_whiff_rate": profile.whiff_rate,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_profile(profile: HitterProfile, profile_dir: Path = PROFILE_DIR) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{profile.player_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2, cls=_NumpyEncoder)


def load_profile(player_id: int, profile_dir: Path = PROFILE_DIR) -> HitterProfile:
    path = profile_dir / f"{player_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No profile for player_id={player_id} at {path}")
    with open(path) as f:
        data = json.load(f)
    return HitterProfile(**data)


def get_league_average_profile() -> HitterProfile:
    return HitterProfile(
        player_id=-1,
        player_name="League Average",
        stand="R",
        **LEAGUE_AVG,
        sample_size=0,
        is_thin_sample=True,
    )


# ---------------------------------------------------------------------------
# Merge profiles into DataFrame (used before training)
# ---------------------------------------------------------------------------

def merge_profiles_into_df(
    df: pd.DataFrame,
    profile_dir: Path = PROFILE_DIR,
    date_cutoff: str = "2023-01-01",
) -> pd.DataFrame:
    """
    For each unique batter, build (or load) their profile using only
    pre-cutoff data, then broadcast profile features onto all their rows.

    Builds a lookup DataFrame and does a single pd.merge — not row-by-row.
    """
    unique_batters = df["batter"].unique()
    print(f"Building profiles for {len(unique_batters):,} unique batters...")

    records = []
    for player_id in unique_batters:
        path = profile_dir / f"{player_id}.json"
        profile = None
        if path.exists():
            try:
                profile = load_profile(player_id, profile_dir)
            except (json.JSONDecodeError, KeyError):
                path.unlink()  # delete corrupted file, rebuild below
        if profile is None:
            profile = build_profile(player_id, df, date_cutoff)
            save_profile(profile, profile_dir)

        row = {"batter": player_id, **profile_to_feature_dict(profile)}
        records.append(row)

    profile_df = pd.DataFrame(records)
    merged = df.merge(profile_df, on="batter", how="left")

    # Fill any unmatched batters with league average
    avg = profile_to_feature_dict(get_league_average_profile())
    for col, val in avg.items():
        merged[col] = merged[col].fillna(val)

    print(f"Profile features merged. Shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build hitter profiles")
    parser.add_argument("--name", type=str, help="Build profile for a specific player by name")
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/statcast_processed.parquet")

    if args.name:
        pid = get_player_id(args.name, df)
        profile = build_profile(pid, df)
        save_profile(profile)
        print(f"\n{profile.player_name} (id={profile.player_id})")
        print(f"  Sample size : {profile.sample_size:,} pitches {'(thin — below 200 weighted PAs)' if profile.is_thin_sample else ''}")
        print(f"  Swing rate  : {profile.swing_rate:.3f}")
        print(f"  Chase rate  : {profile.chase_rate:.3f}")
        print(f"  Contact rate: {profile.contact_rate:.3f}")
        print(f"  Hard hit    : {profile.hard_hit_rate:.3f}")
        print(f"  Whiff rate  : {profile.whiff_rate:.3f}")
    else:
        merged = merge_profiles_into_df(df)
        merged.to_parquet("data/processed/statcast_processed.parquet", index=False)
        print("Saved updated processed file with hitter profile features.")
