"""
fetch.py — Download Statcast pitch-by-pitch data via pybaseball.

Fetches by calendar month to avoid HTTP timeouts on large date ranges.
A manifest tracks completed months so reruns are idempotent.

Usage:
    python -m src.data.fetch              # fetch all missing months
    python -m src.data.fetch --force      # re-fetch everything
"""

import calendar
import json
import os
import time
from pathlib import Path

import pandas as pd
import pybaseball
from tqdm import tqdm

pybaseball.cache.enable()

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"

DATA_START_YEAR = 2015
DATA_END_YEAR = 2026
SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]  # April–October

MAX_RETRIES = 3
RETRY_SLEEP = 10  # seconds between retries


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path = MANIFEST_PATH) -> set:
    """Return set of (year, month) tuples already fetched."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path) as f:
        data = json.load(f)
    return {tuple(entry) for entry in data}


def save_manifest(completed: set, manifest_path: Path = MANIFEST_PATH) -> None:
    """Persist the completed set as JSON."""
    with open(manifest_path, "w") as f:
        json.dump(sorted(completed), f)


# ---------------------------------------------------------------------------
# Single-month fetch
# ---------------------------------------------------------------------------

def fetch_month(year: int, month: int, output_dir: Path = RAW_DIR) -> Path:
    """
    Fetch one calendar month of Statcast data and save as CSV.

    Returns path to saved file.
    Raises RuntimeError if pybaseball returns empty DataFrame after MAX_RETRIES.
    """
    _, last_day = calendar.monthrange(year, month)
    date_from = f"{year}-{month:02d}-01"
    date_to = f"{year}-{month:02d}-{last_day:02d}"
    out_path = output_dir / f"statcast_{year}_{month:02d}.csv"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = pybaseball.statcast(start_dt=date_from, end_dt=date_to, parallel=True)

            # Empty DataFrame = no games that month (e.g. 2020 shortened season).
            # Record as completed-but-empty rather than retrying.
            if df is None or len(df) == 0:
                print(f"  {year}-{month:02d}: no data (empty response, recorded as done)")
                df = pd.DataFrame()
                df.to_csv(out_path, index=False)
                return out_path

            df.to_csv(out_path, index=False)
            print(f"  {year}-{month:02d}: {len(df):,} pitches saved to {out_path.name}")
            return out_path

        except Exception as e:
            print(f"  {year}-{month:02d}: attempt {attempt}/{MAX_RETRIES} failed — {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)

    raise RuntimeError(f"Failed to fetch {year}-{month:02d} after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Full fetch pipeline
# ---------------------------------------------------------------------------

def fetch_all(
    raw_dir: Path = RAW_DIR,
    start_year: int = DATA_START_YEAR,
    end_year: int = DATA_END_YEAR,
    force_refetch: bool = False,
) -> list:
    """
    Fetch all season months from start_year through end_year.

    Skips months already in the manifest unless force_refetch=True.
    Updates manifest after each successful month.
    Returns list of all CSV paths (cached + newly fetched).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if force_refetch else load_manifest()

    all_months = [
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in SEASON_MONTHS
    ]

    pending = [ym for ym in all_months if ym not in completed]
    print(f"Fetching {len(pending)} months ({len(all_months) - len(pending)} already cached)")

    for year, month in tqdm(pending, desc="Fetching Statcast"):
        fetch_month(year, month, raw_dir)
        completed.add((year, month))
        save_manifest(completed)

    return sorted(raw_dir.glob("statcast_*.csv"))


# ---------------------------------------------------------------------------
# Load all raw CSVs into one DataFrame
# ---------------------------------------------------------------------------

def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Read all statcast_*.csv files and concatenate into a single DataFrame.
    Deduplicates on (game_pk, at_bat_number, pitch_number).
    """
    csv_files = sorted(raw_dir.glob("statcast_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No statcast CSV files found in {raw_dir}. Run fetch_all() first.")

    print(f"Loading {len(csv_files)} monthly CSVs...")
    chunks = []
    for path in tqdm(csv_files, desc="Loading CSVs"):
        if path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if len(df) > 0:
            chunks.append(df)

    if not chunks:
        raise ValueError("All CSV files were empty.")

    combined = pd.concat(chunks, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    print(f"Loaded {before:,} rows → {len(combined):,} after deduplication")
    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Statcast data")
    parser.add_argument("--force", action="store_true", help="Re-fetch all months")
    parser.add_argument("--start-year", type=int, default=DATA_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DATA_END_YEAR)
    args = parser.parse_args()

    fetch_all(force_refetch=args.force, start_year=args.start_year, end_year=args.end_year)
