#!/usr/bin/env python3
"""
Build a compact similarity index for the 15 demo hitters.

Reads:  data/processed/statcast_processed.parquet
Writes:
  data/processed/profiles_demo/similarity_index.parquet   (committable, ~10–30 MB)
  data/processed/profiles_demo/similarity_norm_stats.json (means, stds, name→id)

Run once locally then commit both output files:
    python -m scripts.build_demo_similarity_index
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED   = Path("data/processed/statcast_processed.parquet")
DEMO_DIR    = Path("data/processed/profiles_demo")
OUT_PARQUET = DEMO_DIR / "similarity_index.parquet"
OUT_JSON    = DEMO_DIR / "similarity_index_norm_stats.json"  # companion: {stem}_norm_stats.json

# Exact player_name strings from profiles_demo JSONs → MLBAM batter IDs
DEMO_HITTERS: dict[str, int] = {
    "Judge, Aaron":        592450,
    "Betts, Mookie":       605141,
    "Kiner-Falefa, Isiah": 643396,
    "Ohtani, Shohei":      660271,
    "Raleigh, Cal":        663728,
    "Soto, Juan":          665742,
    "Witt, Bobby":         677951,
    "Duran, Jarren":       680776,
    "Carroll, Corbin":     682998,
    "Henderson, Gunnar":   683002,
    "Volpe, Anthony":      683011,
    "Caissie, Owen":       683357,
    "Winn, Masyn":         691026,
    "Langford, Wyatt":     694671,
    "Lee, Jung Hoo":       808982,
}


def main() -> None:
    from src.hitters.similarity import DISTANCE_FEATURES, TRAIN_CUTOFF, _LOAD_COLS

    if not PROCESSED.exists():
        raise FileNotFoundError(f"Processed parquet not found: {PROCESSED}")

    print(f"Loading {PROCESSED} …")
    df = pd.read_parquet(PROCESSED, columns=_LOAD_COLS)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  {len(df):,} total rows.")

    # Norm stats must be computed from the training set (pre-2025) to avoid leakage
    train = df[df["game_date"] < TRAIN_CUTOFF]
    print(f"  Computing norm stats from {len(train):,} training rows …")
    means = [float(train[f].mean()) for f in DISTANCE_FEATURES]
    stds  = [max(float(train[f].std()), 1e-6) for f in DISTANCE_FEATURES]

    # Filter to demo hitters only
    demo_ids = list(DEMO_HITTERS.values())
    demo_df  = df[df["batter"].isin(demo_ids)].reset_index(drop=True)

    print("\nDemo hitter pitch counts:")
    for name, pid in DEMO_HITTERS.items():
        n = int((demo_df["batter"] == pid).sum())
        print(f"  {name:<25} (id={pid})  {n:,} pitches")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    demo_df.to_parquet(OUT_PARQUET, index=False)
    mb = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(demo_df):,} rows → {OUT_PARQUET}  ({mb:.1f} MB)")

    norm_stats = {
        "features":   DISTANCE_FEATURES,
        "means":      means,
        "stds":       stds,
        "name_to_id": DEMO_HITTERS,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved norm stats + {len(DEMO_HITTERS)} name→id entries → {OUT_JSON}")


if __name__ == "__main__":
    main()
