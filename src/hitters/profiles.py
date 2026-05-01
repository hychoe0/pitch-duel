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

PROFILE_DIR     = Path("data/processed/profiles")
AAA_PROFILE_DIR = Path("data/processed/profiles/aaa")
MIN_WEIGHTED_PA = 200  # weighted plate appearances; below this, blend with league averages

# Standard strike zone bounds (universal approximation)
ZONE_X_MIN, ZONE_X_MAX = -0.85, 0.85  # feet from center of plate
ZONE_Z_MIN, ZONE_Z_MAX = 1.5, 3.5     # feet from ground

# League-average fallback values (2015-2022 MLB)
LEAGUE_AVG = {
    "swing_rate":    0.47,
    "chase_rate":    0.30,
    "contact_rate":  0.78,
    "hard_hit_rate": 0.38,
    "whiff_rate":    0.25,
    "zone_xwoba":    0.380,  # MLB avg xwOBA on batted balls (pitch-clock era)
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

# ---------------------------------------------------------------------------
# Zone and pitch-family definitions
# ---------------------------------------------------------------------------

STRIKE_ZONES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BALL_ZONES   = [11, 12, 13, 14]
ALL_ZONES    = STRIKE_ZONES + BALL_ZONES

# Zone layout (from catcher's perspective):
#   11 (high-inside) | 12 (high-outside)
#              ┌───┬───┬───┐
#              │ 1 │ 2 │ 3 │  ← top row
#              ├───┼───┼───┤
#              │ 4 │ 5 │ 6 │  ← middle row
#              ├───┼───┼───┤
#              │ 7 │ 8 │ 9 │  ← bottom row
#              └───┴───┴───┘
#   13 (low-inside)  | 14 (low-outside)

PITCH_FAMILIES = {
    "fastball": ["FF", "SI", "FC"],
    "breaking": ["SL", "CU", "KC", "SV", "CS"],
    "offspeed": ["CH", "FS", "FO"],
    "other":    ["KN", "EP", "SC"],
}
PITCH_FAMILY_LIST = ["fastball", "breaking", "offspeed", "other"]

MIN_ZONE_PITCHES   = 20   # weighted pitches per zone before fallback
MIN_FAMILY_PITCHES = 30   # weighted pitches per pitch family before fallback
MIN_ZONE_BATTED_BALLS = 10   # tunable — adjust as model accuracy improves


def _get_pitch_family(pitch_type: str) -> str:
    for family, types in PITCH_FAMILIES.items():
        if pitch_type in types:
            return family
    return "other"


def assign_pitch_family(df: pd.DataFrame) -> pd.DataFrame:
    """Add a pitch_family column by mapping pitch_type through PITCH_FAMILIES."""
    df = df.copy()
    df["pitch_family"] = df["pitch_type"].map(_get_pitch_family).fillna("other")
    return df


# Cache populated once by merge_profiles_into_df before iterating batters
_NAME_CACHE: dict = {}


def _build_name_cache(player_ids: list) -> None:
    """Bulk-fetch batter names from pybaseball and populate _NAME_CACHE."""
    global _NAME_CACHE
    try:
        import pybaseball
        result = pybaseball.playerid_reverse_lookup(player_ids, key_type="mlbam")
        for _, row in result.iterrows():
            pid = int(row["key_mlbam"])
            _NAME_CACHE[pid] = f"{str(row['name_last']).title()}, {str(row['name_first']).title()}"
        print(f"Resolved names for {len(_NAME_CACHE):,} players.")
    except Exception as e:
        print(f"Name lookup failed ({e}) — profiles will use player IDs as names.")


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
    stand: str                       # 'L', 'R', or 'S' (switch)
    swing_rate: float
    chase_rate: float
    contact_rate: float
    hard_hit_rate: float
    whiff_rate: float
    swing_rate_by_count: dict = field(default_factory=dict)        # "balls-strikes" -> float
    contact_rate_by_pitch_type: dict = field(default_factory=dict) # pitch_type_str -> float
    zone_swing_rates: dict = field(default_factory=dict)           # zone_int -> float
    zone_whiff_rates: dict = field(default_factory=dict)           # zone_int -> float
    zone_xwoba_rates: dict = field(default_factory=dict)           # str(zone_id) -> float
    zone_hard_hit_rates: dict = field(default_factory=dict)        # str(zone_id) -> float
    family_swing_rates: dict = field(default_factory=dict)         # family_str -> float
    family_whiff_rates: dict = field(default_factory=dict)         # family_str -> float
    sample_size: int = 0
    is_thin_sample: bool = False
    league: str = "MLB"  # "MLB" or "AAA"; default keeps old saved profiles loading correctly


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


def compute_zone_swing_rates(df: pd.DataFrame, w: np.ndarray, fallback: float) -> dict:
    """
    Weighted swing rate per Statcast zone (1-9 strike, 11-14 ball).
    Rows with NaN zone are excluded from zone calculations only.
    Falls back to hitter's overall swing rate for zones below MIN_ZONE_PITCHES.
    """
    valid = df[df["zone"].notna()].copy()
    valid_w = w[df.index.get_indexer(valid.index)]
    result = {}
    for z in ALL_ZONES:
        mask = valid["zone"] == z
        grp = valid[mask]
        grp_w = valid_w[valid.index.get_indexer(grp.index)]
        if grp_w.sum() >= MIN_ZONE_PITCHES:
            result[z] = compute_swing_rate(grp, grp_w)
        else:
            result[z] = fallback
    return result


def compute_zone_whiff_rates(df: pd.DataFrame, w: np.ndarray, fallback: float) -> dict:
    """
    Weighted whiff rate per Statcast zone.
    Falls back to hitter's overall whiff rate for thin zones.
    """
    valid = df[df["zone"].notna()].copy()
    valid_w = w[df.index.get_indexer(valid.index)]
    result = {}
    for z in ALL_ZONES:
        mask = valid["zone"] == z
        grp = valid[mask]
        grp_w = valid_w[valid.index.get_indexer(grp.index)]
        if grp_w.sum() >= MIN_ZONE_PITCHES:
            result[z] = compute_whiff_rate(grp, grp_w)
        else:
            result[z] = fallback
    return result


def compute_family_swing_rates(df: pd.DataFrame, w: np.ndarray, fallback: float) -> dict:
    """
    Weighted swing rate per pitch family (fastball, breaking, offspeed, other).
    Falls back to overall swing rate for families below MIN_FAMILY_PITCHES.
    """
    df = assign_pitch_family(df)
    result = {}
    for fam in PITCH_FAMILY_LIST:
        mask = df["pitch_family"] == fam
        grp = df[mask]
        grp_w = w[df.index.get_indexer(grp.index)]
        if grp_w.sum() >= MIN_FAMILY_PITCHES:
            result[fam] = compute_swing_rate(grp, grp_w)
        else:
            result[fam] = fallback
    return result


def compute_family_whiff_rates(df: pd.DataFrame, w: np.ndarray, fallback: float) -> dict:
    """
    Weighted whiff rate per pitch family.
    Falls back to overall whiff rate for thin families.
    """
    df = assign_pitch_family(df)
    result = {}
    for fam in PITCH_FAMILY_LIST:
        mask = df["pitch_family"] == fam
        grp = df[mask]
        grp_w = w[df.index.get_indexer(grp.index)]
        if grp_w.sum() >= MIN_FAMILY_PITCHES:
            result[fam] = compute_whiff_rate(grp, grp_w)
        else:
            result[fam] = fallback
    return result


def compute_contact_rate_by_pitch_type(df: pd.DataFrame, w: np.ndarray) -> dict:
    # Pre-2023 Statcast labeled sweepers as "SV"; 2023+ uses "ST".
    # Merge pre-2023 SV rows into the ST bucket so the sweeper feature
    # has full historical coverage. Post-2023 SV is slurve only.
    df = df.copy()
    pre_2023_sv = (df["game_date"].dt.year < 2023) & (df["pitch_type"] == "SV")
    df.loc[pre_2023_sv, "pitch_type"] = "ST"

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
    profile.swing_rate    = _blend(profile.swing_rate,    LEAGUE_AVG["swing_rate"],    weighted_pa)
    profile.chase_rate    = _blend(profile.chase_rate,    LEAGUE_AVG["chase_rate"],    weighted_pa)
    profile.contact_rate  = _blend(profile.contact_rate,  LEAGUE_AVG["contact_rate"],  weighted_pa)
    profile.hard_hit_rate = _blend(profile.hard_hit_rate, LEAGUE_AVG["hard_hit_rate"], weighted_pa)
    profile.whiff_rate    = _blend(profile.whiff_rate,    LEAGUE_AVG["whiff_rate"],    weighted_pa)
    for z in ALL_ZONES:
        if z in profile.zone_swing_rates:
            profile.zone_swing_rates[z] = _blend(profile.zone_swing_rates[z], LEAGUE_AVG["swing_rate"], weighted_pa)
        if z in profile.zone_whiff_rates:
            profile.zone_whiff_rates[z] = _blend(profile.zone_whiff_rates[z], LEAGUE_AVG["whiff_rate"], weighted_pa)
    for fam in PITCH_FAMILY_LIST:
        if fam in profile.family_swing_rates:
            profile.family_swing_rates[fam] = _blend(profile.family_swing_rates[fam], LEAGUE_AVG["swing_rate"], weighted_pa)
        if fam in profile.family_whiff_rates:
            profile.family_whiff_rates[fam] = _blend(profile.family_whiff_rates[fam], LEAGUE_AVG["whiff_rate"], weighted_pa)
    return profile


# ---------------------------------------------------------------------------
# Player ID lookup
# ---------------------------------------------------------------------------

def get_player_id(name: str, df: pd.DataFrame) -> int:
    """
    Look up MLBAM batter ID by name using pybaseball's player lookup.
    Falls back to searching saved profile JSON names if lookup fails.
    Raises ValueError with suggestions if not found.
    """
    import pybaseball

    # Parse "last, first" or "first last" formats
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.strip().split()
        first, last = parts[0], parts[-1]

    try:
        result = pybaseball.playerid_lookup(last, first, fuzzy=True)
        if len(result) > 0:
            # Filter to players who appear as batters in our dataset
            for _, row in result.iterrows():
                pid = int(row["key_mlbam"])
                if pid in df["batter"].values:
                    return pid
            # If none matched our dataset, return the top lookup result anyway
            return int(result.iloc[0]["key_mlbam"])
    except Exception:
        pass

    # Fallback: search profile JSON names
    profile_dir = PROFILE_DIR
    lower = name.lower()
    matches = []
    for path in profile_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            if lower in data.get("player_name", "").lower():
                matches.append(data["player_name"])
        except Exception:
            continue

    if matches:
        raise ValueError(f"Player '{name}' not in dataset. Similar names in profiles: {matches[:5]}")
    raise ValueError(f"Player '{name}' not found.")


# ---------------------------------------------------------------------------
# Build profile
# ---------------------------------------------------------------------------

def build_profile(
    player_id: int,
    df: pd.DataFrame,
    date_cutoff: str = "2025-01-01",
    pitcher_batter_ids: frozenset = None,
) -> HitterProfile:
    """
    Build a HitterProfile from the player's pitches before date_cutoff.
    Thin-sample blending applied automatically.

    Raises ValueError if player_id is in pitcher_batter_ids — caller should
    substitute a league-average profile instead of building one.
    """
    if pitcher_batter_ids is not None and player_id in pitcher_batter_ids:
        raise ValueError(
            f"Player {player_id} is a pitcher-batter — "
            "profile build skipped. Use league-average profile."
        )
    cutoff = pd.Timestamp(date_cutoff)
    sub = df[(df["batter"] == player_id) & (df["game_date"] < cutoff)].copy()

    # Name is resolved externally via bulk lookup and passed in via _NAME_CACHE.
    # NOTE: player_name in Statcast rows is the PITCHER's name, not the batter's.
    name = _NAME_CACHE.get(player_id, f"Player {player_id}")

    if len(sub) > 0:
        stand_counts = sub["stand"].value_counts()
        if len(stand_counts) > 1 and stand_counts.min() / stand_counts.sum() >= 0.10:
            stand = "S"  # switch hitter
        else:
            stand = stand_counts.index[0]
    else:
        stand = "R"
    n = len(sub)

    w = _row_weights(sub) if n > 0 else np.array([])
    weighted_pa = _weighted_pa_count(sub) if n > 0 else 0.0
    is_thin = weighted_pa < MIN_WEIGHTED_PA

    swing_rate    = compute_swing_rate(sub, w)    if n > 0 else LEAGUE_AVG["swing_rate"]
    chase_rate    = compute_chase_rate(sub, w)    if n > 0 else LEAGUE_AVG["chase_rate"]
    contact_rate  = compute_contact_rate(sub, w)  if n > 0 else LEAGUE_AVG["contact_rate"]
    hard_hit_rate = compute_hard_hit_rate(sub, w) if n > 0 else LEAGUE_AVG["hard_hit_rate"]
    whiff_rate    = compute_whiff_rate(sub, w)    if n > 0 else LEAGUE_AVG["whiff_rate"]

    profile = HitterProfile(
        player_id=player_id,
        player_name=name,
        stand=stand,
        swing_rate=swing_rate,
        chase_rate=chase_rate,
        contact_rate=contact_rate,
        hard_hit_rate=hard_hit_rate,
        whiff_rate=whiff_rate,
        swing_rate_by_count=compute_swing_rate_by_count(sub, w) if n > 0 else {},
        contact_rate_by_pitch_type=compute_contact_rate_by_pitch_type(sub, w) if n > 0 else {},
        zone_swing_rates=compute_zone_swing_rates(sub, w, swing_rate) if n > 0 else {z: swing_rate for z in ALL_ZONES},
        zone_whiff_rates=compute_zone_whiff_rates(sub, w, whiff_rate) if n > 0 else {z: whiff_rate for z in ALL_ZONES},
        family_swing_rates=compute_family_swing_rates(sub, w, swing_rate) if n > 0 else {f: swing_rate for f in PITCH_FAMILY_LIST},
        family_whiff_rates=compute_family_whiff_rates(sub, w, whiff_rate) if n > 0 else {f: whiff_rate for f in PITCH_FAMILY_LIST},
        sample_size=n,
        is_thin_sample=is_thin,
    )

    if is_thin:
        profile = _blend_profile(profile, weighted_pa)

    return profile


# ---------------------------------------------------------------------------
# Feature dict for model input
# ---------------------------------------------------------------------------

# All pitch types for which we expose per-pitch-type contact rates as features.
# ST (sweeper, 2023+ code) absorbs pre-2023 SV; SV slot is slurve only.
PITCH_TYPE_CONTACT_KEYS = ["FF", "CH", "CU", "FC", "KN", "SC", "SI", "SL", "SV", "FS", "ST"]


def profile_to_feature_dict(profile: HitterProfile) -> dict:
    # hitter_stand_enc is intentionally excluded here — it is already set
    # per-row by encode_handedness() in preprocessing from the actual stand column.
    d = {
        "hitter_swing_rate":    profile.swing_rate,
        "hitter_chase_rate":    profile.chase_rate,
        "hitter_contact_rate":  profile.contact_rate,
        "hitter_hard_hit_rate": profile.hard_hit_rate,
        "hitter_whiff_rate":    profile.whiff_rate,
    }
    # Per-pitch-type contact rates (11 features)
    by_type = profile.contact_rate_by_pitch_type
    for pt in PITCH_TYPE_CONTACT_KEYS:
        d[f"hitter_contact_rate_{pt}"] = by_type.get(pt, profile.contact_rate)
    # Zone swing rates (13 features: zones 1-9, 11-14)
    for z in ALL_ZONES:
        d[f"hitter_swing_rate_z{z}"] = profile.zone_swing_rates.get(z, LEAGUE_AVG["swing_rate"])
    # Zone whiff rates (13 features)
    for z in ALL_ZONES:
        d[f"hitter_whiff_rate_z{z}"] = profile.zone_whiff_rates.get(z, LEAGUE_AVG["whiff_rate"])
    # Pitch family swing rates (4 features)
    for fam in PITCH_FAMILY_LIST:
        d[f"hitter_swing_rate_{fam}"] = profile.family_swing_rates.get(fam, LEAGUE_AVG["swing_rate"])
    # Pitch family whiff rates (4 features)
    for fam in PITCH_FAMILY_LIST:
        d[f"hitter_whiff_rate_{fam}"] = profile.family_whiff_rates.get(fam, LEAGUE_AVG["whiff_rate"])
    # Count-specific swing rates (12 features: all (balls, strikes) combos 0-0 through 3-2)
    # Used during preprocessing to resolve hitter_swing_rate_this_count per row.
    for b in range(4):
        for s in range(3):
            key = f"{b}-{s}"
            d[f"hitter_swing_rate_count_{b}_{s}"] = profile.swing_rate_by_count.get(key, profile.swing_rate)
    return d


def resolve_hitter_features_for_pitch(
    profile: "HitterProfile",
    zone: int,
) -> dict:
    """
    Returns the 5 global rate features plus zone-specific damage rates.
    Falls back to LEAGUE_AVG["zone_xwoba"] for xwOBA (no hitter-global stored),
    and to profile.hard_hit_rate for zone hard-hit rate (better prior than league avg).
    """
    return {
        "hitter_swing_rate":          profile.swing_rate,
        "hitter_chase_rate":          profile.chase_rate,
        "hitter_contact_rate":        profile.contact_rate,
        "hitter_hard_hit_rate":       profile.hard_hit_rate,
        "hitter_whiff_rate":          profile.whiff_rate,
        "hitter_zone_xwoba":          profile.zone_xwoba_rates.get(
                                          str(zone), LEAGUE_AVG["zone_xwoba"]
                                      ),
        "hitter_zone_hard_hit_rate":  profile.zone_hard_hit_rates.get(
                                          str(zone), profile.hard_hit_rate
                                      ),
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
        zone_swing_rates={z: LEAGUE_AVG["swing_rate"] for z in ALL_ZONES},
        zone_whiff_rates={z: LEAGUE_AVG["whiff_rate"] for z in ALL_ZONES},
        family_swing_rates={f: LEAGUE_AVG["swing_rate"] for f in PITCH_FAMILY_LIST},
        family_whiff_rates={f: LEAGUE_AVG["whiff_rate"] for f in PITCH_FAMILY_LIST},
        sample_size=0,
        is_thin_sample=True,
    )


# ---------------------------------------------------------------------------
# Merge profiles into DataFrame (used before training)
# ---------------------------------------------------------------------------

def merge_profiles_into_df(
    df: pd.DataFrame,
    profile_dir: Path = PROFILE_DIR,
    date_cutoff: str = "2025-01-01",
    pitcher_batter_ids: frozenset = None,
) -> pd.DataFrame:
    """
    For each unique batter, build (or load) their profile using only
    pre-cutoff data, then broadcast profile features onto all their rows.

    Pitcher-batters (pre-2022 NL) are skipped — their rows receive league-
    average profile values via the fillna pass at the end.

    Builds a lookup DataFrame and does a single pd.merge — not row-by-row.
    """
    if pitcher_batter_ids is None:
        from src.data.preprocess import identify_pitcher_batters
        pitcher_batter_ids = identify_pitcher_batters(df)

    # Filter to batters active in 2022+ (excludes retired players, pre-DH NL pitchers)
    all_batters = df["batter"].unique()
    active_cutoff = pd.Timestamp("2022-01-01")
    active_batters = set(df[df["game_date"] >= active_cutoff]["batter"].unique())
    unique_batters = np.array([b for b in all_batters if b in active_batters])
    n_filtered = len(all_batters) - len(unique_batters)
    print(f"Kept {len(unique_batters):,} active hitters, skipped {n_filtered:,} retired/inactive (no PA since 2022)")

    n_skipped = sum(1 for pid in unique_batters if pid in pitcher_batter_ids)
    print(
        f"Building profiles for {len(unique_batters):,} unique batters "
        f"({n_skipped:,} pitcher-batters skipped → league average)"
    )
    _build_name_cache(list(unique_batters))

    records = []
    for player_id in unique_batters:
        if player_id in pitcher_batter_ids:
            continue  # league average applied via fillna after merge

        path = profile_dir / f"{player_id}.json"
        profile = None
        if path.exists():
            try:
                profile = load_profile(player_id, profile_dir)
            except (json.JSONDecodeError, KeyError):
                path.unlink()  # delete corrupted file, rebuild below
        if profile is None:
            profile = build_profile(player_id, df, date_cutoff, pitcher_batter_ids)
            save_profile(profile, profile_dir)

        row = {"batter": player_id, **profile_to_feature_dict(profile)}
        records.append(row)

    profile_df = pd.DataFrame(records)

    # Drop pre-existing hitter profile columns from df to prevent _x/_y suffix collision
    profile_cols = [c for c in profile_df.columns if c != "batter"]
    df = df.drop(columns=[c for c in profile_cols if c in df.columns])

    merged = df.merge(profile_df, on="batter", how="left")

    # Fill any unmatched batters with league average
    avg = profile_to_feature_dict(get_league_average_profile())
    for col, val in avg.items():
        merged[col] = merged[col].fillna(val)

    print(f"Profile features merged. Shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# AAA profile building
# ---------------------------------------------------------------------------

def build_aaa_profile(
    player_id: int,
    df: pd.DataFrame,
    date_cutoff: str = "2026-12-31",
) -> HitterProfile:
    """
    Build an AAA HitterProfile. Internally delegates to build_profile() —
    the statistics math is identical. Sets league="AAA" on the returned object.

    date_cutoff defaults to end-of-current-year since AAA data has no leakage
    risk with the trained model (model was trained on MLB data only).

    MLB profile preference rule: callers should check PROFILE_DIR first.
    This function only runs when no MLB profile is available.
    """
    profile = build_profile(player_id, df, date_cutoff)
    profile.league = "AAA"
    return profile


def build_all_aaa_profiles(
    df: pd.DataFrame,
    output_dir: Path = AAA_PROFILE_DIR,
    date_cutoff: str = "2026-12-31",
    min_weighted_pa: float = MIN_WEIGHTED_PA,
    force_rebuild: bool = False,
) -> dict[int, HitterProfile]:
    """
    Build and save AAA profiles for all batters in df with sufficient sample.

    MLB preference rule: skips any player_id that already has a saved MLB profile
    in PROFILE_DIR (those hitters should be predicted with their MLB profile).

    Returns {player_id: HitterProfile} for all successfully built AAA profiles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = df["batter"].unique().tolist()

    # Resolve names in bulk (populates _NAME_CACHE)
    _build_name_cache(all_ids)

    # Identify batters with existing FULL (non-thin) MLB profiles — skip them.
    # Batters with thin MLB profiles are still eligible for AAA profiles because
    # their MLB sample is insufficient; AAA data may give a better read.
    mlb_full_ids = set()
    for p in PROFILE_DIR.glob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            if not d.get("is_thin_sample", True):
                mlb_full_ids.add(int(p.stem))
        except Exception:
            pass
    aaa_only_ids = [pid for pid in all_ids if int(pid) not in mlb_full_ids]
    n_skipped_mlb = len(all_ids) - len(aaa_only_ids)
    print(
        f"\nAAA profile build: {len(all_ids):,} unique batters  "
        f"| {n_skipped_mlb:,} have full MLB profiles (skipped)  "
        f"| {len(aaa_only_ids):,} candidates (thin-MLB or AAA-only)"
    )

    results: dict[int, HitterProfile] = {}
    n_built = 0
    n_thin  = 0
    n_skip_sample = 0
    n_skip_cached = 0

    for player_id in aaa_only_ids:
        pid = int(player_id)
        out_path = output_dir / f"{pid}.json"

        if out_path.exists() and not force_rebuild:
            try:
                profile = load_profile(pid, output_dir)
                results[pid] = profile
                n_skip_cached += 1
                continue
            except Exception:
                out_path.unlink(missing_ok=True)

        sub = df[df["batter"] == player_id].copy()
        if len(sub) == 0:
            continue

        weighted_pa = _weighted_pa_count(sub)
        if weighted_pa < min_weighted_pa * 0.5:
            # Skip very thin samples (< 100 weighted PAs) outright
            n_skip_sample += 1
            continue

        try:
            profile = build_aaa_profile(pid, df, date_cutoff)
            save_profile(profile, output_dir)
            results[pid] = profile
            n_built += 1
            if profile.is_thin_sample:
                n_thin += 1
        except Exception as e:
            print(f"  WARN: build_aaa_profile({pid}) failed — {e}")

    print(
        f"AAA profiles: {n_built} built ({n_thin} thin-sample blended)  "
        f"| {n_skip_cached} from cache  "
        f"| {n_skip_sample} skipped (< 100 weighted PAs)"
    )
    return results


def list_available_hitters(league: str = "MLB") -> list[str]:
    """
    Return sorted list of hitter names with saved profiles for the given league.

    league="MLB" → data/processed/profiles/*.json
    league="AAA" → data/processed/profiles/aaa/*.json
    """
    profile_dir = AAA_PROFILE_DIR if league.upper() == "AAA" else PROFILE_DIR
    names = []
    for path in profile_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            name = data.get("player_name", "")
            if name:
                names.append(name)
        except Exception:
            continue
    return sorted(names)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_profile(profile: HitterProfile) -> None:
    thin = "(thin — below 200 weighted PAs)" if profile.is_thin_sample else ""
    print(f"\n  {profile.player_name} (id={profile.player_id})")
    print(f"    Sample size : {profile.sample_size:,} pitches {thin}")
    print(f"    Swing rate  : {profile.swing_rate:.3f}")
    print(f"    Chase rate  : {profile.chase_rate:.3f}")
    print(f"    Contact rate: {profile.contact_rate:.3f}")
    print(f"    Hard hit    : {profile.hard_hit_rate:.3f}")
    print(f"    Whiff rate  : {profile.whiff_rate:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build hitter profiles")
    parser.add_argument("--name",  type=str, help="Build profile for a single player by name")
    parser.add_argument("--names", type=str, nargs="+", help="Build profiles for multiple players by name")
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/statcast_processed.parquet")

    if args.names:
        # Targeted multi-player build — does NOT touch any other profiles
        _build_name_cache([])  # pre-warm lookup table once
        targets = args.names
        print(f"\nBuilding profiles for {len(targets)} player(s)...\n")
        for name in targets:
            try:
                pid = get_player_id(name, df)
                _NAME_CACHE[pid] = _NAME_CACHE.get(pid) or name  # fallback if lookup missed
                profile = build_profile(pid, df)
                save_profile(profile)
                _print_profile(profile)
            except Exception as e:
                print(f"  ERROR — {name}: {e}")

    elif args.name:
        _build_name_cache([])
        pid = get_player_id(args.name, df)
        profile = build_profile(pid, df)
        save_profile(profile)
        _print_profile(profile)

    else:
        merged = merge_profiles_into_df(df)
        merged.to_parquet("data/processed/statcast_processed.parquet", index=False)
        print("Saved updated processed file with hitter profile features.")
