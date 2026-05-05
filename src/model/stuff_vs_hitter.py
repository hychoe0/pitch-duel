"""
stuff_vs_hitter.py — Hitter-calibrated pitch physics score.

Computes two features that measure pitch *shape* quality against a specific hitter,
using k-nearest-neighbor similarity over historical pitches the hitter has seen:

  stuff_vs_hitter_xwoba  — expected xwOBA from this pitch shape vs this hitter
  stuff_vs_hitter_whiff  — expected whiff rate from this pitch shape vs this hitter

These are pitch-Duel's Stuff+ equivalent, but hitter-calibrated rather than
league-average calibrated.

Usage:
    python -m src.model.stuff_vs_hitter --build-indexes [--demo-only]
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physical pitch characteristics used for similarity — no location, no count, no hitter.
PHYSICS_FEATURES = [
    "release_speed",
    "spin_to_velo_ratio",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
]

K_MIN_NEIGHBORS   = 10    # minimum neighbors before falling back to blend/thin-sample
DIST_NORMALIZATION = 3.0  # distance at which similarity_quality reaches 0

LEAGUE_AVG_STUFF_XWOBA = 0.200  # league avg xwOBA on swings (whiffs/fouls = 0, batted balls ≈ 0.38)
LEAGUE_AVG_STUFF_WHIFF = 0.250

INDEX_DIR            = Path("data/processed/physics_indexes")
STANDARDIZATION_PATH = Path("models/physics_standardization.json")

# Mirror from profiles.py — kept local to avoid circular imports
_SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

_WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
}

# Module-level caches — avoid repeated disk reads within a request or demo run
_index_cache: dict = {}
_standardization_cache: dict = {}


# ---------------------------------------------------------------------------
# Thin-sample / fallback
# ---------------------------------------------------------------------------

def _thin_sample_result() -> dict:
    return {
        "stuff_vs_hitter_xwoba": LEAGUE_AVG_STUFF_XWOBA,
        "stuff_vs_hitter_whiff": LEAGUE_AVG_STUFF_WHIFF,
        "n_neighbors_found":     0,
        "similarity_quality":    0.0,
    }


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def build_physics_standardization(df: pd.DataFrame) -> dict:
    """
    Compute mean and std for PHYSICS_FEATURES over pre-2023 training data.
    Saves result to STANDARDIZATION_PATH; returns the dict.

    Pre-2023 cutoff matches the index-building cutoff — ensures the
    standardization and the index share the same distribution.
    """
    cutoff = pd.Timestamp("2023-01-01")
    train_df = df[df["game_date"] < cutoff]

    stats = {}
    for feat in PHYSICS_FEATURES:
        col = train_df[feat].dropna()
        stats[feat] = {
            "mean": float(col.mean()),
            "std":  float(col.std()),
        }

    STANDARDIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STANDARDIZATION_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved physics standardization → {STANDARDIZATION_PATH}")
    for feat, s in stats.items():
        print(f"  {feat:<22}  mean={s['mean']:.3f}  std={s['std']:.3f}")
    return stats


def _load_standardization(path: Path = STANDARDIZATION_PATH) -> dict:
    """Load and cache the standardization dict."""
    key = str(path)
    if key not in _standardization_cache:
        with open(path) as f:
            _standardization_cache[key] = json.load(f)
    return _standardization_cache[key]


def _make_std_arrays(standardization: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (means, stds) float32 arrays aligned with PHYSICS_FEATURES."""
    means = np.array([standardization[f]["mean"] for f in PHYSICS_FEATURES], dtype=np.float32)
    stds  = np.array([standardization[f]["std"]  for f in PHYSICS_FEATURES], dtype=np.float32)
    stds  = np.where(stds > 0, stds, 1.0)  # guard against zero variance
    return means, stds


def _standardize_dict(physics_dict: dict, standardization: dict) -> np.ndarray:
    """Standardize a pitch_physics dict to a (7,) float32 array."""
    means, stds = _make_std_arrays(standardization)
    raw = np.array([float(physics_dict.get(f, 0.0)) for f in PHYSICS_FEATURES], dtype=np.float32)
    return (raw - means) / stds


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_hitter_physics_index(
    player_id: int,
    df: pd.DataFrame,
    standardization: dict,
    output_dir: Path = INDEX_DIR,
) -> int:
    """
    Build and save a physics index for one hitter.
    Only uses pre-2023 swings (no leakage into test set).
    Returns n_swings saved (0 if nothing to save).
    """
    cutoff = pd.Timestamp("2023-01-01")
    sub = df[
        (df["batter"] == player_id)
        & (df["game_date"] < cutoff)
        & (df["description"].isin(_SWING_DESCRIPTIONS))
    ].copy()

    sub = sub.dropna(subset=PHYSICS_FEATURES)
    n = len(sub)
    if n == 0:
        return 0

    means, stds = _make_std_arrays(standardization)
    physics_raw    = sub[PHYSICS_FEATURES].values.astype(np.float32)
    physics_matrix = (physics_raw - means) / stds

    xwoba_vals  = sub["estimated_woba_using_speedangle"].fillna(0.0).values.astype(np.float32)
    whiff_flags = sub["description"].isin(_WHIFF_DESCRIPTIONS).values.astype(np.uint8)
    pitch_types = sub["pitch_type"].values.astype("U4")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{player_id}.npz",
        physics_matrix=physics_matrix,
        xwoba_vals=xwoba_vals,
        whiff_flags=whiff_flags,
        pitch_types=pitch_types,
    )
    return n


def _load_index(player_id: int, index_dir: Path) -> dict | None:
    """Load and cache a hitter's physics index. Returns None if not found."""
    if player_id in _index_cache:
        return _index_cache[player_id]
    path = index_dir / f"{player_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    entry = {
        "physics_matrix": data["physics_matrix"],
        "xwoba_vals":     data["xwoba_vals"],
        "whiff_flags":    data["whiff_flags"],
        "pitch_types":    data["pitch_types"],
    }
    _index_cache[player_id] = entry
    return entry


# ---------------------------------------------------------------------------
# Pitch-type / family filter helpers
# ---------------------------------------------------------------------------

def _filter_index(
    index: dict,
    pitch_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (physics_matrix, xwoba_vals, whiff_flags) filtered by pitch_type,
    relaxing to pitch family or all types when sample is thin.
    """
    from src.hitters.profiles import PITCH_FAMILY_MAP, PITCH_FAMILIES

    pm = index["physics_matrix"]
    xv = index["xwoba_vals"]
    wf = index["whiff_flags"]
    pt = index["pitch_types"]

    # Exact pitch type match
    mask = pt == pitch_type
    if mask.sum() >= K_MIN_NEIGHBORS:
        return pm[mask], xv[mask], wf[mask]

    # Relax to pitch family
    family = PITCH_FAMILY_MAP.get(pitch_type, "other")
    family_types = list(PITCH_FAMILIES.get(family, [])) + [pitch_type]
    mask_fam = np.isin(pt, family_types)
    if mask_fam.sum() >= K_MIN_NEIGHBORS:
        return pm[mask_fam], xv[mask_fam], wf[mask_fam]

    # Use everything
    return pm, xv, wf


def _blend_result(xwoba: float, whiff: float, n: int) -> tuple[float, float]:
    """Blend toward league averages when n is between K_MIN_NEIGHBORS and 3×."""
    if n >= K_MIN_NEIGHBORS * 3:
        return xwoba, whiff
    w = n / (K_MIN_NEIGHBORS * 3)
    return (
        w * xwoba + (1 - w) * LEAGUE_AVG_STUFF_XWOBA,
        w * whiff  + (1 - w) * LEAGUE_AVG_STUFF_WHIFF,
    )


# ---------------------------------------------------------------------------
# Inference: single-pitch query
# ---------------------------------------------------------------------------

def compute_stuff_vs_hitter(
    pitch_physics: dict,
    player_id: int,
    pitch_type: str,
    index_dir: Path = INDEX_DIR,
    standardization_path: Path = STANDARDIZATION_PATH,
    k_neighbors: int = 50,
    _exclude_dist_zero: bool = False,
) -> dict:
    """
    Returns:
      {
        "stuff_vs_hitter_xwoba":  float,
        "stuff_vs_hitter_whiff":  float,
        "n_neighbors_found":      int,
        "similarity_quality":     float,  # 0-1, 1 = very close matches
      }

    _exclude_dist_zero: set True at training time (LOO — exclude self-matches).
    """
    index = _load_index(player_id, index_dir)
    if index is None:
        return _thin_sample_result()

    pm, xv, wf = _filter_index(index, pitch_type)
    if len(pm) < K_MIN_NEIGHBORS:
        return _thin_sample_result()

    standardization = _load_standardization(standardization_path)
    query = _standardize_dict(pitch_physics, standardization)

    dists = np.linalg.norm(pm - query, axis=1)

    if _exclude_dist_zero:
        valid = dists > 0.0
        dists = dists[valid]
        xv    = xv[valid]
        wf    = wf[valid]

    if len(dists) < K_MIN_NEIGHBORS:
        return _thin_sample_result()

    k = min(k_neighbors, len(dists))
    if k < len(dists):
        k_idx = np.argpartition(dists, k)[:k]
    else:
        k_idx = np.arange(len(dists))

    avg_dist          = float(dists[k_idx].mean())
    similarity_quality = max(0.0, 1.0 - avg_dist / DIST_NORMALIZATION)

    xwoba_c = float(xv[k_idx].mean())
    whiff_c = float(wf[k_idx].mean())
    xwoba_c, whiff_c = _blend_result(xwoba_c, whiff_c, k)

    return {
        "stuff_vs_hitter_xwoba": xwoba_c,
        "stuff_vs_hitter_whiff": whiff_c,
        "n_neighbors_found":     k,
        "similarity_quality":    similarity_quality,
    }


# ---------------------------------------------------------------------------
# Training time: add both columns to a DataFrame
# ---------------------------------------------------------------------------

def add_stuff_vs_hitter_features(
    df: pd.DataFrame,
    index_dir: Path = INDEX_DIR,
    standardization_path: Path = STANDARDIZATION_PATH,
) -> pd.DataFrame:
    """
    Add stuff_vs_hitter_xwoba and stuff_vs_hitter_whiff to df.

    Processes per hitter, batches distance computation per pitch_type.
    Vectorized: distance matrix for each (hitter, pitch_type) computed in one
    matrix-multiply so n_query × n_index distances are evaluated simultaneously.

    LOO: excludes dist == 0.0 (self-matches) so training rows are not
    their own nearest neighbors.
    """
    if not standardization_path.exists():
        raise FileNotFoundError(
            f"Standardization not found at {standardization_path}. "
            "Run: python -m src.model.stuff_vs_hitter --build-indexes"
        )

    df = df.copy()
    df["stuff_vs_hitter_xwoba"] = LEAGUE_AVG_STUFF_XWOBA
    df["stuff_vs_hitter_whiff"] = LEAGUE_AVG_STUFF_WHIFF

    standardization = _load_standardization(standardization_path)
    means, stds = _make_std_arrays(standardization)

    n_processed = 0
    for batter_id, batter_df in df.groupby("batter"):
        index = _load_index(int(batter_id), index_dir)
        if index is None:
            continue

        n_processed += 1

        for pitch_type in batter_df["pitch_type"].dropna().unique():
            pt_mask = batter_df["pitch_type"] == pitch_type
            pt_rows = batter_df[pt_mask]
            if pt_rows.empty:
                continue

            pm, xv, wf = _filter_index(index, pitch_type)
            if len(pm) < K_MIN_NEIGHBORS:
                continue  # leave league-avg defaults

            # Build query matrix — replace NaN with feature mean before standardizing
            raw = pt_rows[PHYSICS_FEATURES].values.astype(np.float32)
            for i in range(raw.shape[1]):
                nan_mask = np.isnan(raw[:, i])
                if nan_mask.any():
                    raw[nan_mask, i] = means[i]
            query_matrix = (raw - means) / stds  # (n_q, 7)

            # Pairwise distance matrix — memory-efficient via identity:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2·a·bᵀ
            q_sq = (query_matrix ** 2).sum(axis=1, keepdims=True)   # (n_q, 1)
            p_sq = (pm ** 2).sum(axis=1)                             # (n_pm,)
            cross = query_matrix @ pm.T                               # (n_q, n_pm)
            dist_sq = np.maximum(q_sq + p_sq - 2 * cross, 0.0)
            dists_matrix = np.sqrt(dist_sq)                          # (n_q, n_pm)

            # LOO: self-matches have dist ≈ 0; set to inf so they're never selected
            dists_matrix = np.where(dists_matrix > 0.0, dists_matrix, np.inf)

            n_q = len(pt_rows)
            xwoba_out = np.full(n_q, LEAGUE_AVG_STUFF_XWOBA, dtype=np.float32)
            whiff_out  = np.full(n_q, LEAGUE_AVG_STUFF_WHIFF,  dtype=np.float32)

            k = min(50, len(pm))

            for row_i in range(n_q):
                row_dists = dists_matrix[row_i]
                finite    = np.isfinite(row_dists)
                n_valid   = int(finite.sum())
                if n_valid < K_MIN_NEIGHBORS:
                    continue

                kk    = min(k, n_valid)
                d_v   = row_dists[finite]
                x_v   = xv[finite]
                w_v   = wf[finite]
                # argpartition requires kth < len(a); use full sort when kk == n_valid
                if kk < n_valid:
                    k_idx = np.argpartition(d_v, kk)[:kk]
                else:
                    k_idx = np.arange(n_valid)

                xwoba_c = float(x_v[k_idx].mean())
                whiff_c = float(w_v[k_idx].mean())
                xwoba_c, whiff_c = _blend_result(xwoba_c, whiff_c, kk)

                xwoba_out[row_i] = xwoba_c
                whiff_out[row_i]  = whiff_c

            df.loc[pt_rows.index, "stuff_vs_hitter_xwoba"] = xwoba_out
            df.loc[pt_rows.index, "stuff_vs_hitter_whiff"]  = whiff_out

    print(f"add_stuff_vs_hitter_features: {n_processed} hitters, {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Build hitter physics indexes")
    parser.add_argument("--build-indexes", action="store_true", required=True)
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only build indexes for the 15 DEMO_HITTERS",
    )
    args = parser.parse_args()

    from src.hitters.profiles import DEMO_HITTERS

    t0 = time.time()
    print("Loading processed parquet...")
    df = pd.read_parquet("data/processed/statcast_processed.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Build (or load) standardization
    if not STANDARDIZATION_PATH.exists():
        print("\nBuilding physics standardization...")
        standardization = build_physics_standardization(df)
    else:
        print(f"\nLoading existing standardization from {STANDARDIZATION_PATH}")
        standardization = _load_standardization(STANDARDIZATION_PATH)
        for feat, s in standardization.items():
            print(f"  {feat:<22}  mean={s['mean']:.3f}  std={s['std']:.3f}")

    # Determine which batters to index
    if args.demo_only:
        batter_ids = list(DEMO_HITTERS.values())
        print(f"\nBuilding indexes for {len(batter_ids)} DEMO_HITTERS...")
    else:
        cutoff = pd.Timestamp("2023-01-01")
        active = set(df[df["game_date"] >= pd.Timestamp("2022-01-01")]["batter"].unique())
        swings = df[
            (df["game_date"] < cutoff)
            & (df["description"].isin(_SWING_DESCRIPTIONS))
            & (df["batter"].isin(active))
        ]
        batter_ids = swings["batter"].unique().tolist()
        print(f"\nBuilding indexes for {len(batter_ids):,} active hitters...")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    name_map = {v: k for k, v in DEMO_HITTERS.items()}
    n_built = 0
    for pid in batter_ids:
        n_swings = build_hitter_physics_index(int(pid), df, standardization, INDEX_DIR)
        if n_swings > 0:
            label = name_map.get(pid, f"player_{pid}")
            print(f"  {label:<28}  n_swings={n_swings:>6,}")
            n_built += 1
        else:
            print(f"  {pid}  — skipped (no pre-2023 swings with complete physics)")

    elapsed = time.time() - t0
    print(f"\nDone: {n_built} indexes built in {elapsed:.1f}s")
    print(f"Index directory: {INDEX_DIR}")
