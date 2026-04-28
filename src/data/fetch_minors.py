"""
fetch_minors.py — Download Triple-A Statcast data from Baseball Savant.

Uses sport_id=11 to pull AAA pitch-by-pitch data (available from 2023).
Chunks into 5-day windows per month to stay under the Savant 25k row cap.
Manifest tracks completed months so reruns are idempotent.

Usage:
    python -m src.data.fetch_minors              # fetch all missing months
    python -m src.data.fetch_minors --force      # re-fetch everything
    python -m src.data.fetch_minors --year 2024  # single year
"""

import calendar
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

RAW_DIR       = Path("data/raw_minors")
MANIFEST_PATH = RAW_DIR / "manifest.json"

DATA_START_YEAR = 2023   # first year AAA Statcast available
DATA_END_YEAR   = 2026
SEASON_MONTHS   = [4, 5, 6, 7, 8, 9, 10]

CHUNK_DAYS      = 5      # days per request (>5 risks hitting 25k row cap)
REQUEST_TIMEOUT = 90     # seconds — 5-day AAA chunks take ~20s; give headroom
MAX_RETRIES     = 3
RETRY_SLEEP     = 15     # seconds between retries
CHUNK_SLEEP     = 2      # polite pause between sequential chunk requests

# ABS (automated ball/strike) descriptions appear in AAA but not MLB
_AAA_DESC_EXTRA = {"automatic_ball", "automatic_strike"}

AAA_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&hfPT=&hfAB=&hfGT=R%7C&player_type=pitcher"
    "&game_date_gt={start}&game_date_lt={end}"
    "&type=details&sport_id=11"
)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST_PATH) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return {tuple(entry) for entry in data}


def save_manifest(completed: set, path: Path = MANIFEST_PATH) -> None:
    with open(path, "w") as f:
        json.dump(sorted(completed), f)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _month_chunks(year: int, month: int) -> list[tuple[date, date]]:
    """Return (start, end) date pairs covering the month in CHUNK_DAYS windows."""
    _, last_day = calendar.monthrange(year, month)
    today       = date.today()
    m_start     = date(year, month, 1)
    m_end       = min(date(year, month, last_day), today)

    if m_start > today:
        return []

    chunks = []
    cur = m_start
    while cur <= m_end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), m_end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    """Fetch one 5-day window with retries. Returns DataFrame (may be empty)."""
    url = AAA_URL.format(start=start, end=end)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if "error" in df.columns:
                raise RuntimeError(f"Savant error: {df['error'].iloc[0]}")
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                sleep_t = RETRY_SLEEP * attempt
                print(f"    {start}→{end} attempt {attempt} failed ({exc}). Retry in {sleep_t}s...")
                time.sleep(sleep_t)
    raise RuntimeError(f"Failed to fetch {start}→{end} after {MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Single-month fetch
# ---------------------------------------------------------------------------

def fetch_month(year: int, month: int, output_dir: Path = RAW_DIR) -> Path | None:
    """
    Fetch one calendar month of AAA Statcast data via 5-day chunks.
    Returns path to saved CSV, or None if the month hasn't started yet.
    """
    chunks = _month_chunks(year, month)
    if not chunks:
        return None  # future month

    out_path = output_dir / f"minors_aaa_{year}_{month:02d}.csv"
    frames = []

    for start, end in chunks:
        df = _fetch_chunk(start, end)
        if len(df) > 0:
            frames.append(df)
        time.sleep(CHUNK_SLEEP)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"  {year}-{month:02d}: {len(combined):,} pitches  ({len(chunks)} chunks)")
    else:
        print(f"  {year}-{month:02d}: no data (recorded as done)")
        pd.DataFrame().to_csv(out_path, index=False)

    return out_path


# ---------------------------------------------------------------------------
# Full fetch pipeline
# ---------------------------------------------------------------------------

def fetch_all(
    raw_dir: Path = RAW_DIR,
    start_year: int = DATA_START_YEAR,
    end_year: int = DATA_END_YEAR,
    force_refetch: bool = False,
) -> list[Path]:
    """
    Fetch all season months from start_year through end_year.
    Skips months already in the manifest unless force_refetch=True.
    Updates manifest after each successful month.
    Returns list of all CSV paths.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if force_refetch else load_manifest()

    all_months = [
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in SEASON_MONTHS
    ]

    # Skip months that haven't started yet
    today = date.today()
    all_months = [(y, m) for (y, m) in all_months if date(y, m, 1) <= today]

    pending = [ym for ym in all_months if ym not in completed]
    print(f"\nAAA fetch: {len(pending)} months pending  ({len(all_months) - len(pending)} already cached)")

    for year, month in tqdm(pending, desc="Fetching AAA Statcast"):
        try:
            path = fetch_month(year, month, raw_dir)
            if path is not None:
                completed.add((year, month))
                save_manifest(completed)
        except Exception as e:
            print(f"  ERROR {year}-{month:02d}: {e}. Skipping.")

    return sorted(raw_dir.glob("minors_aaa_*.csv"))


# ---------------------------------------------------------------------------
# Load raw CSVs
# ---------------------------------------------------------------------------

def load_raw_minors(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Read all minors_aaa_*.csv files and concatenate into a single DataFrame.
    Deduplicates on (game_pk, at_bat_number, pitch_number).
    """
    csv_files = sorted(raw_dir.glob("minors_aaa_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No AAA CSV files in {raw_dir}. Run fetch_all() first.")

    print(f"Loading {len(csv_files)} monthly AAA CSVs...")
    chunks = []
    for path in tqdm(csv_files, desc="Loading"):
        if path.stat().st_size < 10:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if len(df) > 0:
            chunks.append(df)

    if not chunks:
        raise ValueError("All AAA CSV files were empty.")

    combined = pd.concat(chunks, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    after = len(combined)
    print(f"Loaded {before:,} rows → {after:,} after deduplication")
    return combined


# ---------------------------------------------------------------------------
# Year breakdown report
# ---------------------------------------------------------------------------

def print_summary(raw_dir: Path = RAW_DIR) -> None:
    """Print per-year pitch counts from saved CSVs."""
    csv_files = sorted(raw_dir.glob("minors_aaa_*.csv"))
    year_counts: dict[int, int] = {}
    total = 0
    for path in csv_files:
        if path.stat().st_size < 10:
            continue
        try:
            df = pd.read_csv(path, low_memory=False, usecols=["game_year"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        year = int(path.stem.split("_")[2])
        n = len(df)
        year_counts[year] = year_counts.get(year, 0) + n
        total += n

    print("\n=== AAA Statcast fetch summary ===")
    for yr in sorted(year_counts):
        print(f"  {yr}: {year_counts[yr]:>10,} pitches")
    print(f"  {'TOTAL':>4}: {total:>10,} pitches")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Triple-A Statcast data")
    parser.add_argument("--force",      action="store_true", help="Re-fetch all months")
    parser.add_argument("--start-year", type=int, default=DATA_START_YEAR)
    parser.add_argument("--end-year",   type=int, default=DATA_END_YEAR)
    parser.add_argument("--year",       type=int, default=None, help="Fetch a single year only")
    args = parser.parse_args()

    if args.year:
        args.start_year = args.year
        args.end_year   = args.year

    fetch_all(force_refetch=args.force, start_year=args.start_year, end_year=args.end_year)
    print_summary()
