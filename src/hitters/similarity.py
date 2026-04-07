"""
similarity.py — Historical pitch similarity search.

Finds the most similar pitches a specific hitter has faced in their Statcast
history using weighted Euclidean distance on z-score-normalized pitch features.

Pitch type is handled as a discrete family-match weight (1.0/0.7/0.3) rather
than a continuous z-score feature, since the PITCH_TYPE_MAP encoding carries
no meaningful numeric distance.

Usage:
    python -m src.hitters.similarity \\
        --hitter "Shohei Ohtani" --pitch-type FF --velo 92 \\
        --plate-x 0.5 --plate-z 3.25
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_PATH = Path("data/processed/statcast_processed.parquet")
PROFILE_DIR    = Path("data/processed/profiles")
TRAIN_CUTOFF   = pd.Timestamp("2025-01-01")

# Only the columns we actually need — avoids loading 100+ engineered feature cols
_LOAD_COLS = [
    "batter", "game_date", "game_pk", "at_bat_number",
    "pitch_type", "release_speed", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "release_spin_rate", "release_extension",
    "description", "launch_speed", "events", "balls", "strikes", "zone",
]

# ---------------------------------------------------------------------------
# Similarity metric configuration
# ---------------------------------------------------------------------------

# Continuous features — z-score normalized before distance calculation
DISTANCE_FEATURES = [
    "release_speed",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "release_extension",
]

FEATURE_WEIGHTS = {
    "release_speed":     3.0,  # velocity is the primary pitch identifier
    "pfx_x":            2.0,  # horizontal movement defines pitch type
    "pfx_z":            2.0,  # vertical movement defines pitch type
    "plate_x":          2.5,  # location is critical to outcome
    "plate_z":          2.5,  # location is critical to outcome
    "release_spin_rate": 1.0,  # secondary quality metric
    "release_extension": 0.5,  # release point context
}

# Pitch-type family matching weights
# Applied as a divisor on effective distance (higher weight → closer match)
PITCH_FAMILY_MAP = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "breaking", "CU": "breaking", "KC": "breaking",
    "SV": "breaking", "CS": "breaking", "ST": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed",
    "KN": "other",    "EP": "other",    "SC": "other",
}

EXACT_MATCH_WEIGHT  = 1.0
FAMILY_MATCH_WEIGHT = 0.7
CROSS_FAMILY_WEIGHT = 0.3

MIN_COUNT_MATCHES = 10  # fall back to cross-count if count slice is thinner than this

# Recency weights — matches profiles.py PROFILE_RECENCY_WEIGHTS
_RECENCY: dict = {2026: 3.0, 2025: 2.5, 2024: 2.0, 2023: 1.5, 2022: 1.0}
_RECENCY_DEFAULT = 0.5

# Outcome description sets — mirrors preprocess.py constants
_SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "foul_bunt", "missed_bunt", "hit_into_play",
    "hit_into_play_no_out", "hit_into_play_score",
}
_CONTACT_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}
_IN_PLAY_DESCRIPTIONS = {
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}
_HIT_EVENTS = {"single", "double", "triple", "home_run"}

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_norm_cache: dict = {}    # str(data_path) -> {"means": ndarray, "stds": ndarray}
_hitter_cache: dict = {}  # (player_id, str(data_path)) -> DataFrame


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimilarityResult:
    n_matches: int          # how many similar pitches found
    avg_similarity: float   # mean similarity score (0–1; 1 = identical)

    # Empirical rates computed from matched pitches (recency-weighted)
    empirical_swing_rate: float
    empirical_contact_rate: float    # contact rate given swing
    empirical_hard_hit_rate: float   # hard hit rate given in-play contact (≥95 mph)
    empirical_hit_rate: float        # hit rate (any hit event) across all matched pitches

    # Outcome distribution (unweighted counts, for readability)
    outcome_counts: dict             # e.g. {'swinging_strike': 12, 'foul': 8, ...}

    # Top N most similar pitches with a similarity_score column appended
    matched_pitches: pd.DataFrame

    # Confidence: high when there are 40+ close matches
    confidence: float                # = min(1.0, n_matches/40) * avg_similarity

    # True if we fell back from count-matched to cross-count search
    count_fallback: bool = False


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def _load_norm_stats(data_path: Path) -> tuple:
    """
    Compute and cache per-feature mean and std from the training set.
    Uses only rows where game_date < TRAIN_CUTOFF to avoid test-set leakage.
    Returns (means, stds) as 1-D arrays aligned to DISTANCE_FEATURES order.
    """
    key = str(data_path)
    if key in _norm_cache:
        return _norm_cache[key]["means"], _norm_cache[key]["stds"]

    print("Computing normalization stats from training set...")
    df = pd.read_parquet(data_path, columns=["game_date"] + DISTANCE_FEATURES)
    df["game_date"] = pd.to_datetime(df["game_date"])
    train = df[df["game_date"] < TRAIN_CUTOFF]

    means = np.array([float(train[f].mean()) for f in DISTANCE_FEATURES])
    stds  = np.array([max(float(train[f].std()), 1e-6) for f in DISTANCE_FEATURES])

    _norm_cache[key] = {"means": means, "stds": stds}
    print(f"  Stats computed from {len(train):,} training rows.")
    return means, stds


# ---------------------------------------------------------------------------
# Hitter pitch loader
# ---------------------------------------------------------------------------

def _load_hitter_pitches(player_id: int, data_path: Path, means: np.ndarray) -> pd.DataFrame:
    """
    Load and cache all pitches seen by this hitter from the processed parquet.
    NaN values in distance features are imputed with the training-set mean.
    Returns a DataFrame with the columns in _LOAD_COLS (game_date as datetime).
    """
    key = (player_id, str(data_path))
    if key in _hitter_cache:
        return _hitter_cache[key]

    df = pd.read_parquet(data_path, columns=_LOAD_COLS)
    df["game_date"] = pd.to_datetime(df["game_date"])
    hitter_df = df[df["batter"] == player_id].copy().reset_index(drop=True)

    # Impute NaN in distance features with the training mean (same imputation used
    # during preprocessing for pitcher-median fields like release_spin_rate)
    for i, col in enumerate(DISTANCE_FEATURES):
        hitter_df[col] = hitter_df[col].fillna(float(means[i]))

    _hitter_cache[key] = hitter_df
    print(f"  Loaded {len(hitter_df):,} pitches for player_id={player_id}.")
    return hitter_df


# ---------------------------------------------------------------------------
# Pitch-type match weight
# ---------------------------------------------------------------------------

def _pitch_type_weights(query_type: str, candidate_types: pd.Series) -> np.ndarray:
    """
    Return a (n,) array of pitch-type match weights:
      1.0  exact type match
      0.7  same pitch family (e.g. FF vs SI — both fastball)
      0.3  cross-family (e.g. FF vs SL)

    These are used as divisors: effective_distance = distance / type_weight,
    so cross-family pitches are pushed further away regardless of physical similarity.
    """
    query_family = PITCH_FAMILY_MAP.get(query_type, "other")
    cand_families = candidate_types.map(lambda t: PITCH_FAMILY_MAP.get(t, "other"))

    exact  = (candidate_types == query_type).values
    family = (~exact) & (cand_families == query_family).values

    return np.where(exact, EXACT_MATCH_WEIGHT,
           np.where(family, FAMILY_MATCH_WEIGHT, CROSS_FAMILY_WEIGHT))


# ---------------------------------------------------------------------------
# Recency weights and empirical rate computation
# ---------------------------------------------------------------------------

def _recency_weights(dates: pd.Series) -> np.ndarray:
    return dates.dt.year.map(lambda y: _RECENCY.get(y, _RECENCY_DEFAULT)).values


def _compute_empirical_rates(
    matched: pd.DataFrame,
) -> tuple:
    """
    Compute recency-weighted empirical rates from the matched pitch rows.
    Returns (swing_rate, contact_rate, hard_hit_rate, hit_rate, outcome_counts).
    """
    w = _recency_weights(matched["game_date"])
    total_w = w.sum()
    if total_w == 0:
        return 0.0, 0.0, 0.0, 0.0, {}

    desc = matched["description"]

    is_swing   = desc.isin(_SWING_DESCRIPTIONS).values
    is_contact = desc.isin(_CONTACT_DESCRIPTIONS).values
    is_in_play = desc.isin(_IN_PLAY_DESCRIPTIONS).values
    is_hard    = is_in_play & (matched["launch_speed"].fillna(0) >= 95).values
    is_hit     = matched["events"].isin(_HIT_EVENTS).values

    swing_rate = float((w * is_swing).sum() / total_w)

    w_swing = w * is_swing
    contact_rate = (
        float((w_swing * is_contact).sum() / w_swing.sum()) if w_swing.sum() > 0 else 0.0
    )

    w_in_play = w * is_in_play
    hard_hit_rate = (
        float((w * is_hard).sum() / w_in_play.sum()) if w_in_play.sum() > 0 else 0.0
    )

    hit_rate = float((w * is_hit).sum() / total_w)

    # Unweighted counts — easier to read in the CLI summary
    outcome_counts = desc.value_counts().to_dict()

    return swing_rate, contact_rate, hard_hit_rate, hit_rate, outcome_counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_similar_pitches(
    pitch: dict,
    hitter_name: str,
    n_matches: int = 50,
    match_count: bool = True,
    data_path: Path = PROCESSED_PATH,
) -> SimilarityResult:
    """
    Find the most similar pitches this hitter has faced in their Statcast history.

    Args:
        pitch:        Pitch dict (same format as predict.py input). Required keys:
                      pitch_type, release_speed, plate_x, plate_z.
                      Optional but improves accuracy: pfx_x, pfx_z,
                      release_spin_rate, release_extension, balls, strikes.
                      Missing values default to the training-set mean.
        hitter_name:  e.g. "Shohei Ohtani"
        n_matches:    Max similar pitches to return (top N by similarity)
        match_count:  If True, restrict candidates to the same count (balls-strikes).
                      Falls back to cross-count if fewer than MIN_COUNT_MATCHES found.
        data_path:    Path to the processed parquet file

    Returns:
        SimilarityResult dataclass
    """
    # Resolve player_id from name
    from src.hitters.profiles import get_player_id
    batter_df = pd.read_parquet(data_path, columns=["batter"])
    player_id = get_player_id(hitter_name, batter_df)

    # Load norm stats first so we can use them for NaN imputation
    means, stds = _load_norm_stats(data_path)

    # Load this hitter's pitch history
    hitter_df = _load_hitter_pitches(player_id, data_path, means)
    if len(hitter_df) == 0:
        raise ValueError(f"No pitches found for '{hitter_name}' (player_id={player_id})")

    # Build z-score normalized, weighted query vector
    mean_map = dict(zip(DISTANCE_FEATURES, means))
    q_raw = np.array([float(pitch.get(f, mean_map[f])) for f in DISTANCE_FEATURES])
    feat_weights = np.array([FEATURE_WEIGHTS[f] for f in DISTANCE_FEATURES])
    q_weighted = ((q_raw - means) / stds) * feat_weights

    def _score(candidates: pd.DataFrame) -> np.ndarray:
        """Return similarity scores (0–1) for the candidate DataFrame."""
        X = candidates[DISTANCE_FEATURES].values          # (n, 7)
        X_weighted = ((X - means) / stds) * feat_weights  # broadcast

        diff = X_weighted - q_weighted                     # (n, 7)
        distances = np.sqrt((diff ** 2).sum(axis=1))       # (n,)

        # Apply pitch-type family weight: cross-family pitches get a distance penalty
        type_w = _pitch_type_weights(
            pitch.get("pitch_type", "FF"), candidates["pitch_type"]
        )
        effective_dist = distances / type_w  # lower weight → higher effective distance

        # exp(-d): d=0 → 1.0 (identical), d≫1 → 0.0 (very different)
        return np.exp(-effective_dist)

    # Count-aware candidate selection
    balls   = int(pitch.get("balls", 0))
    strikes = int(pitch.get("strikes", 0))
    count_fallback = False

    if match_count:
        count_mask  = (hitter_df["balls"] == balls) & (hitter_df["strikes"] == strikes)
        count_cands = hitter_df[count_mask]
        if len(count_cands) >= MIN_COUNT_MATCHES:
            candidates = count_cands
        else:
            if len(count_cands) > 0:
                print(
                    f"  Count {balls}-{strikes}: only {len(count_cands)} pitches found — "
                    "falling back to cross-count search."
                )
            else:
                print(
                    f"  Count {balls}-{strikes}: no pitches found — "
                    "falling back to cross-count search."
                )
            candidates = hitter_df
            count_fallback = True
    else:
        candidates = hitter_df

    # Score all candidates and select top N
    sims = _score(candidates)
    top_idx = np.argsort(-sims)[:n_matches]
    top_df  = candidates.iloc[top_idx].copy()
    top_df["similarity_score"] = sims[top_idx]
    top_df = top_df.reset_index(drop=True)

    n        = len(top_df)
    avg_sim  = float(top_df["similarity_score"].mean()) if n > 0 else 0.0
    confidence = min(1.0, n / 40) * avg_sim

    # Empirical rates from matched pitches
    swing, contact, hard_hit, hit_rate, outcome_counts = _compute_empirical_rates(top_df)

    # Trim matched_pitches to display-relevant columns
    display_cols = [
        "game_date", "pitch_type", "release_speed", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "balls", "strikes",
        "description", "events", "launch_speed", "zone", "similarity_score",
    ]
    display_cols = [c for c in display_cols if c in top_df.columns]

    return SimilarityResult(
        n_matches=n,
        avg_similarity=round(avg_sim, 4),
        empirical_swing_rate=round(swing, 4),
        empirical_contact_rate=round(contact, 4),
        empirical_hard_hit_rate=round(hard_hit, 4),
        empirical_hit_rate=round(hit_rate, 4),
        outcome_counts=outcome_counts,
        matched_pitches=top_df[display_cols],
        confidence=round(confidence, 4),
        count_fallback=count_fallback,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_result(result: SimilarityResult, hitter: str, pitch_type: str, velo: float) -> None:
    print(f"\n{'='*62}")
    print(f"  Similarity: {pitch_type} {velo:.1f} mph  →  {hitter}")
    print(f"{'='*62}")

    count_note = "  (cross-count fallback)" if result.count_fallback else ""
    print(f"  Matches:        {result.n_matches}{count_note}")
    print(f"  Avg similarity: {result.avg_similarity:.4f}")
    print(f"  Confidence:     {result.confidence:.4f}")
    print()
    print("  Empirical rates (recency-weighted):")
    print(f"    Swing rate          : {result.empirical_swing_rate:.3f}")
    print(f"    Contact rate|swing  : {result.empirical_contact_rate:.3f}")
    print(f"    Hard hit rate|contact: {result.empirical_hard_hit_rate:.3f}")
    print(f"    Hit rate (any)      : {result.empirical_hit_rate:.3f}")
    print()

    print("  Outcome distribution:")
    total = sum(result.outcome_counts.values())
    for outcome, cnt in sorted(result.outcome_counts.items(), key=lambda x: -x[1]):
        pct  = cnt / total * 100 if total > 0 else 0
        bar  = "█" * int(pct / 2)
        print(f"    {outcome:<35} {cnt:>4}  {pct:>5.1f}%  {bar}")
    print()

    print("  Top 10 most similar pitches:")
    for _, row in result.matched_pitches.head(10).iterrows():
        date = str(row["game_date"])[:10]
        desc = str(row.get("description", "?"))
        ev   = str(row.get("events", "") or "").strip()
        sim  = float(row.get("similarity_score", 0))
        spd  = float(row.get("release_speed", 0))
        ls   = row.get("launch_speed")
        ls_s = f"  EV={ls:.1f}" if ls and not np.isnan(ls) else ""
        print(
            f"    {date}  {str(row['pitch_type']):>3}  {spd:.1f}mph"
            f"  px={row['plate_x']:.2f} pz={row['plate_z']:.2f}"
            f"  {desc:<28} {ev:<12}{ls_s}  sim={sim:.4f}"
        )
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find similar historical pitches a hitter has faced"
    )
    parser.add_argument("--hitter",      required=True,  type=str,
                        help='e.g. "Shohei Ohtani"')
    parser.add_argument("--pitch-type",  required=True,  type=str, dest="pitch_type",
                        help="Statcast pitch type code: FF, SL, CH, etc.")
    parser.add_argument("--velo",        required=True,  type=float,
                        help="Release speed in mph")
    parser.add_argument("--plate-x",     type=float, default=0.0,  dest="plate_x",
                        help="Horizontal plate location in feet (default 0.0 = center)")
    parser.add_argument("--plate-z",     type=float, default=2.5,  dest="plate_z",
                        help="Vertical plate location in feet (default 2.5 = mid-zone)")
    parser.add_argument("--pfx-x",       type=float, default=None, dest="pfx_x",
                        help="Horizontal movement (inches)")
    parser.add_argument("--pfx-z",       type=float, default=None, dest="pfx_z",
                        help="Vertical movement (inches)")
    parser.add_argument("--spin",        type=float, default=None,
                        help="Spin rate (rpm)")
    parser.add_argument("--balls",       type=int,   default=0,
                        help="Ball count (0-3)")
    parser.add_argument("--strikes",     type=int,   default=0,
                        help="Strike count (0-2)")
    parser.add_argument("--no-count",    action="store_true", dest="no_count",
                        help="Search across all counts (disable count-aware matching)")
    parser.add_argument("--n",           type=int,   default=50,
                        help="Max number of similar pitches to return (default 50)")
    args = parser.parse_args()

    pitch = {
        "pitch_type":    args.pitch_type,
        "release_speed": args.velo,
        "plate_x":       args.plate_x,
        "plate_z":       args.plate_z,
        "balls":         args.balls,
        "strikes":       args.strikes,
    }
    if args.pfx_x is not None:
        pitch["pfx_x"] = args.pfx_x
    if args.pfx_z is not None:
        pitch["pfx_z"] = args.pfx_z
    if args.spin is not None:
        pitch["release_spin_rate"] = args.spin

    result = find_similar_pitches(
        pitch=pitch,
        hitter_name=args.hitter,
        n_matches=args.n,
        match_count=not args.no_count,
    )

    _print_result(result, args.hitter, args.pitch_type, args.velo)
