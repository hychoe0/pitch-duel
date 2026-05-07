"""
pvhi.py — Pitch vs. Hitter Index (PVHI), unified kNN architecture.

Replaces the broken three-component formula (stuff × 0.15 + location × 0.65 +
count × 0.20) which inflated scores because zone_xwoba is on-contact, not
per-pitch, creating a denominator mismatch.

New approach: one kNN lookup across 11 dimensions simultaneously
(7 stuff + 2 location + 2 count) against this hitter's historical pitch data.
The neighbor outcomes are already in per-pitch xwOBA space (0.0 for whiffs/
balls/fouls, positive for batted balls), so dividing by the hitter's overall
per-pitch xwOBA is dimensionally correct.

Scale: 100 = average for this hitter. Range: 0–200.

Usage:
    from src.model.pvhi import compute_pvhi
    result = compute_pvhi(pitch, profile, player_id=int(profile.player_id))

CLI:
    python -m src.model.pvhi --build-indexes [--demo-only]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.hitters.profiles import HitterProfile

# ---------------------------------------------------------------------------
# Dimensions and constants
# ---------------------------------------------------------------------------

PVHI_DIMENSIONS = [
    # Stuff (7) — physical pitch characteristics
    "release_speed",
    "spin_to_velo_ratio",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    # Location (2) — where pitch crossed the plate
    "plate_x",
    "plate_z",
    # Count (2) — game situation at time of pitch
    "balls",
    "strikes",
]
PVHI_N_DIMS      = len(PVHI_DIMENSIONS)  # 11
PVHI_STUFF_DIMS  = 7                     # first 7 columns = stuff only (Level 4 fallback)

PVHI_K_NEIGHBORS   = 30
PVHI_MIN_NEIGHBORS = 10

PVHI_INDEX_DIR = Path("data/processed/pvhi_indexes")
PVHI_STD_PATH  = Path("models/pvhi_standardization.json")

PVHI_LEAGUE_AVG_XWOBA = 0.075  # per-pitch (matches LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH)
PVHI_MAX = 200.0
PVHI_MIN = 0.0

# Blend toward 100 (hitter average) as relaxation level increases
_RELAXATION_CONFIDENCE = {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.40, 5: 0.0}

_HIT_INTO_PLAY_DESCRIPTIONS = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

# Module-level caches (avoid repeated disk reads per request)
_pvhi_index_cache: dict = {}
_pvhi_std_cache:   dict = {}


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def build_pvhi_standardization(df: pd.DataFrame) -> dict:
    """
    Compute mean/std for all 11 PVHI_DIMENSIONS from pre-2023 training data.
    Saves to PVHI_STD_PATH and returns the dict.
    """
    cutoff = pd.Timestamp("2023-01-01")
    train  = df[df["game_date"] < cutoff].copy()

    if "spin_to_velo_ratio" not in train.columns:
        speed = train["release_speed"].replace(0, np.nan)
        train["spin_to_velo_ratio"] = train["release_spin_rate"] / speed

    stats = {}
    for feat in PVHI_DIMENSIONS:
        col = train[feat].dropna()
        stats[feat] = {
            "mean": float(col.mean()),
            "std":  float(max(col.std(), 0.001)),
        }

    PVHI_STD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PVHI_STD_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved PVHI standardization → {PVHI_STD_PATH}")
    for feat, s in stats.items():
        print(f"  {feat:<22}  mean={s['mean']:.4f}  std={s['std']:.4f}")
    return stats


def _load_pvhi_std(path: Path = PVHI_STD_PATH) -> dict:
    key = str(path)
    if key not in _pvhi_std_cache:
        with open(path) as f:
            _pvhi_std_cache[key] = json.load(f)
    return _pvhi_std_cache[key]


def _std_arrays(std: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (means, stds) float32 arrays aligned with PVHI_DIMENSIONS."""
    means = np.array([std[f]["mean"] for f in PVHI_DIMENSIONS], dtype=np.float32)
    stds  = np.array([std[f]["std"]  for f in PVHI_DIMENSIONS], dtype=np.float32)
    stds  = np.where(stds > 0, stds, 1.0)
    return means, stds


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_pvhi_index(
    player_id: int,
    df: pd.DataFrame,
    standardization: dict,
    output_dir: Path = PVHI_INDEX_DIR,
) -> int:
    """
    Build and save a PVHI index for one hitter from pre-2023 data.
    Uses ALL pitches (not just swings) — non-contact outcomes store xwOBA = 0.0.
    Returns n_pitches saved (0 if nothing to save).
    """
    from src.hitters.profiles import PITCH_FAMILY_MAP

    cutoff = pd.Timestamp("2023-01-01")
    sub = df[(df["batter"] == player_id) & (df["game_date"] < cutoff)].copy()

    # Compute spin_to_velo_ratio if not in df
    if "spin_to_velo_ratio" not in sub.columns:
        speed = sub["release_speed"].replace(0, np.nan)
        sub["spin_to_velo_ratio"] = sub["release_spin_rate"] / speed

    # Keep only rows where all 11 dimensions are present
    base_dims = [d for d in PVHI_DIMENSIONS if d != "spin_to_velo_ratio"]
    sub = sub.dropna(subset=base_dims + ["spin_to_velo_ratio"])
    n = len(sub)
    if n == 0:
        return 0

    means, stds = _std_arrays(standardization)
    raw_matrix  = sub[PVHI_DIMENSIONS].values.astype(np.float32)
    pvhi_matrix = (raw_matrix - means) / stds

    # xwOBA per pitch: positive for batted balls, 0.0 for everything else
    is_contact = sub["description"].isin(_HIT_INTO_PLAY_DESCRIPTIONS)
    xwoba_vals = np.where(
        is_contact & sub["estimated_woba_using_speedangle"].notna(),
        sub["estimated_woba_using_speedangle"].fillna(0.0).values,
        0.0,
    ).astype(np.float32)

    pitch_types    = sub["pitch_type"].fillna("FF").values.astype("U4")
    pitch_families = np.array(
        [PITCH_FAMILY_MAP.get(str(pt), "other") for pt in pitch_types], dtype="U10"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{player_id}.npz",
        pvhi_matrix=pvhi_matrix,
        xwoba_vals=xwoba_vals,
        pitch_types=pitch_types,
        pitch_families=pitch_families,
    )
    return n


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_pvhi_index(player_id: int, index_dir: Path) -> dict | None:
    if player_id in _pvhi_index_cache:
        return _pvhi_index_cache[player_id]
    path = index_dir / f"{player_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    entry = {
        "pvhi_matrix":    data["pvhi_matrix"],
        "xwoba_vals":     data["xwoba_vals"],
        "pitch_types":    data["pitch_types"],
        "pitch_families": data["pitch_families"],
    }
    _pvhi_index_cache[player_id] = entry
    return entry


# ---------------------------------------------------------------------------
# Query vector construction
# ---------------------------------------------------------------------------

def _build_query_vector(pitch: dict, standardization: dict) -> np.ndarray:
    """Build standardized 11-dim query vector from a raw pitch dict."""
    spin  = float(pitch.get("release_spin_rate") or 0)
    speed = float(pitch.get("release_speed") or 1)
    spin_velo = spin / speed if speed else 0.0

    values = []
    for dim in PVHI_DIMENSIONS:
        if dim == "spin_to_velo_ratio":
            values.append(spin_velo)
        else:
            values.append(float(pitch.get(dim, 0.0)))

    raw   = np.array(values, dtype=np.float32)
    means, stds = _std_arrays(standardization)
    return (raw - means) / stds


# ---------------------------------------------------------------------------
# Relaxation cascade
# ---------------------------------------------------------------------------

def _find_neighbors(
    query_vector:   np.ndarray,
    pvhi_matrix:    np.ndarray,
    xwoba_vals:     np.ndarray,
    pitch_types:    np.ndarray,
    pitch_families: np.ndarray,
    pitch_type:     str,
    pitch_family:   str,
    k:     int = PVHI_K_NEIGHBORS,
    min_k: int = PVHI_MIN_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return (xwoba_neighbors, distances_k, relaxation_level).

    Level 1: exact pitch type, all 11 dims
    Level 2: pitch family, all 11 dims
    Level 3: no pitch filter, all 11 dims
    Level 4: no pitch filter, 7 stuff dims only
    Level 5: no data → return empty arrays, triggers fallback
    """
    def _knn(qv: np.ndarray, pm: np.ndarray, xv: np.ndarray):
        if len(pm) < min_k:
            return None
        dists = np.linalg.norm(pm - qv, axis=1)
        k_use = min(k, len(dists))
        idx   = np.argpartition(dists, k_use - 1)[:k_use] if k_use < len(dists) else np.arange(len(dists))
        return xv[idx], dists[idx]

    # Level 1: exact pitch type
    mask1 = pitch_types == pitch_type
    r = _knn(query_vector, pvhi_matrix[mask1], xwoba_vals[mask1])
    if r is not None:
        return r[0], r[1], 1

    # Level 2: pitch family
    mask2 = pitch_families == pitch_family
    r = _knn(query_vector, pvhi_matrix[mask2], xwoba_vals[mask2])
    if r is not None:
        return r[0], r[1], 2

    # Level 3: all pitches, full 11 dims
    r = _knn(query_vector, pvhi_matrix, xwoba_vals)
    if r is not None:
        return r[0], r[1], 3

    # Level 4: all pitches, 7 stuff dims only
    q_stuff  = query_vector[:PVHI_STUFF_DIMS]
    pm_stuff = pvhi_matrix[:, :PVHI_STUFF_DIMS]
    r = _knn(q_stuff, pm_stuff, xwoba_vals)
    if r is not None:
        return r[0], r[1], 4

    return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 5


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def _fallback_result(reason: str = "") -> dict:
    return {
        "pvhi":                    100.0,
        "pvhi_interpretation":     interpret_pvhi(100.0),
        "pvhi_n_neighbors":        0,
        "pvhi_relaxation_level":   5,
        "pvhi_similarity_quality": 0.0,
        "debug": {"fallback": True, "reason": reason},
    }


def compute_pvhi(
    pitch: dict,
    profile: "HitterProfile",
    player_id: int,
    index_dir: Path = PVHI_INDEX_DIR,
    standardization_path: Path = PVHI_STD_PATH,
    k: int = PVHI_K_NEIGHBORS,
) -> dict:
    """
    Unified kNN PVHI computation.

    Returns:
      {
        "pvhi":                   float,  # 0–200, 100 = hitter's average
        "pvhi_interpretation":    str,
        "pvhi_n_neighbors":       int,
        "pvhi_relaxation_level":  int,    # 1–5
        "pvhi_similarity_quality": float, # 0–1
        "debug":                  dict,   # all intermediate values
      }
    """
    from src.hitters.profiles import PITCH_FAMILY_MAP, get_overall_xwoba_per_pitch

    index = _load_pvhi_index(player_id, index_dir)
    if index is None:
        return _fallback_result(f"no pvhi index for player {player_id}")

    try:
        std = _load_pvhi_std(standardization_path)
    except FileNotFoundError:
        return _fallback_result("pvhi_standardization.json not found")

    query        = _build_query_vector(pitch, std)
    pitch_type   = pitch.get("pitch_type", "FF")
    pitch_family = PITCH_FAMILY_MAP.get(pitch_type, "other")

    xwoba_neighbors, dists_k, level = _find_neighbors(
        query_vector=query,
        pvhi_matrix=index["pvhi_matrix"],
        xwoba_vals=index["xwoba_vals"],
        pitch_types=index["pitch_types"],
        pitch_families=index["pitch_families"],
        pitch_type=pitch_type,
        pitch_family=pitch_family,
        k=k,
        min_k=PVHI_MIN_NEIGHBORS,
    )

    n_neighbors = len(xwoba_neighbors)
    if n_neighbors == 0:
        return _fallback_result("level 5 — no neighbors found")

    # Denominator: hitter's mean per-pitch xwOBA across all pitches
    denom = get_overall_xwoba_per_pitch(profile)

    neighbor_mean_xwoba = float(xwoba_neighbors.mean())
    raw_ratio           = neighbor_mean_xwoba / max(denom, 0.001)
    pvhi_raw            = raw_ratio * 100.0

    # Blend toward 100 (hitter average) proportionally with relaxation confidence
    confidence   = _RELAXATION_CONFIDENCE.get(level, 0.0)
    pvhi_blended = confidence * pvhi_raw + (1.0 - confidence) * 100.0

    pvhi = float(max(PVHI_MIN, min(PVHI_MAX, pvhi_blended)))

    avg_dist           = float(dists_k.mean()) if len(dists_k) > 0 else 0.0
    similarity_quality = max(0.0, 1.0 - avg_dist / 3.0)

    debug = {
        "pitch_type":   pitch_type,
        "pitch_family": pitch_family,
        # Query vector (unstandardized values for readability)
        "query": {
            "release_speed":     pitch.get("release_speed"),
            "spin_to_velo_ratio": round(float(pitch.get("release_spin_rate", 0) or 0) /
                                        max(float(pitch.get("release_speed", 1) or 1), 0.001), 4),
            "pfx_x":             pitch.get("pfx_x"),
            "pfx_z":             pitch.get("pfx_z"),
            "plate_x":           pitch.get("plate_x"),
            "plate_z":           pitch.get("plate_z"),
            "balls":             pitch.get("balls"),
            "strikes":           pitch.get("strikes"),
        },
        # Relaxation cascade
        "relaxation_level":  level,
        "confidence":        confidence,
        "n_neighbors":       n_neighbors,
        # Computation
        "neighbor_mean_xwoba":         round(neighbor_mean_xwoba, 5),
        "denom_overall_xwoba_per_pitch": round(denom, 5),
        "raw_ratio":          round(raw_ratio, 4),
        "pvhi_raw":           round(pvhi_raw, 2),
        "pvhi_blended":       round(pvhi_blended, 2),
        "pvhi_final":         round(pvhi, 2),
        "avg_dist":           round(avg_dist, 4),
        "similarity_quality": round(similarity_quality, 4),
    }

    return {
        "pvhi":                    round(pvhi, 2),
        "pvhi_interpretation":     interpret_pvhi(pvhi),
        "pvhi_n_neighbors":        n_neighbors,
        "pvhi_relaxation_level":   level,
        "pvhi_similarity_quality": round(similarity_quality, 4),
        "debug":                   debug,
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_pvhi(pvhi: float) -> str:
    if pvhi >= 130:
        return "high_danger"
    if pvhi >= 110:
        return "above_average_danger"
    if pvhi >= 90:
        return "neutral"
    if pvhi >= 70:
        return "safer_than_average"
    return "very_safe"


# ---------------------------------------------------------------------------
# CLI — build indexes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PVHI kNN indexes")
    parser.add_argument("--build-indexes", action="store_true",
                        help="Build per-hitter PVHI indexes")
    parser.add_argument("--demo-only", action="store_true",
                        help="Only build indexes for the 15 DEMO_HITTERS")
    parser.add_argument("--rebuild-std", action="store_true",
                        help="Force-rebuild pvhi_standardization.json even if it exists")
    args = parser.parse_args()

    if not args.build_indexes:
        parser.print_help()
    else:
        from src.hitters.profiles import DEMO_HITTERS

        print("Loading Statcast parquet…")
        df = pd.read_parquet("data/processed/statcast_processed.parquet")

        if "spin_to_velo_ratio" not in df.columns:
            speed = df["release_speed"].replace(0, np.nan)
            df["spin_to_velo_ratio"] = df["release_spin_rate"] / speed

        # Build / load standardization
        if args.rebuild_std or not PVHI_STD_PATH.exists():
            print("\nBuilding PVHI standardization…")
            std = build_pvhi_standardization(df)
        else:
            print(f"Loading PVHI standardization from {PVHI_STD_PATH}")
            with open(PVHI_STD_PATH) as f:
                std = json.load(f)

        # Determine target hitters
        if args.demo_only:
            player_ids = list(DEMO_HITTERS.values())
            name_map   = {v: k for k, v in DEMO_HITTERS.items()}
            print(f"\nBuilding PVHI indexes for {len(player_ids)} demo hitters…")
        else:
            player_ids = sorted(df["batter"].dropna().astype(int).unique().tolist())
            name_map   = {v: k for k, v in DEMO_HITTERS.items()}
            print(f"\nBuilding PVHI indexes for {len(player_ids)} hitters…")

        PVHI_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        total = 0
        for pid in player_ids:
            n    = build_pvhi_index(pid, df, std, PVHI_INDEX_DIR)
            name = name_map.get(pid, str(pid))
            print(f"  {name:<30} {n:>7,} pitches")
            total += 1

        print(f"\nDone — built {total} PVHI indexes in {PVHI_INDEX_DIR}/")
