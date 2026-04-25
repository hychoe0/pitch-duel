"""
run_synthetic_demo.py — Two separated per-league prediction reports against the
synthetic D1 pitcher (50 pitches: 15 FF / 15 FC / 10 SL / 10 CH, seed=42).

Outputs:
  data/synthetic/d1_pitcher_predictions_mlb.csv
  data/synthetic/d1_pitcher_predictions_aaa.csv
  Terminal: MLB report, then a visible divider, then AAA report.

Usage:
    python -m scripts.run_synthetic_demo
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

CSV_PATH    = Path("data/synthetic/d1_pitcher_righty_demo.csv")
OUT_MLB     = Path("data/synthetic/d1_pitcher_predictions_mlb.csv")
OUT_AAA     = Path("data/synthetic/d1_pitcher_predictions_aaa.csv")

PITCHER_DESC = (
    "Synthetic mid-tier D1 RHP — 92-93 mph FF, "
    "30% FF / 30% FC / 20% SL / 20% CH"
)

# MLB hitters — MLBAM IDs kept for reference but name-based lookup is used
MLB_HITTERS = [
    "Shohei Ohtani",
    "Aaron Judge",
    "Francisco Lindor",
    "José Ramírez",
    "Jackson Merrill",
    "Munetaka Murakami",
    "Kevin McGonigle",
]

# AAA hitters — names must match player_name in their AAA profile JSONs
# (pybaseball "Last, First" format, built during Task 2 Phase 4)
AAA_HITTERS = [
    "Domínguez, Jasson",
    "Baldwin, Drake",
    "Dingler, Dillon",
    "Narváez, Carlos",
    "Kurtz, Nick",
]


def _print_rank_comparison(league: str, report: dict) -> None:
    """Print P(hard) rank vs xwOBA rank table. Flag hitters who move 2+ positions."""
    hitters = report["hitters"]
    if not hitters:
        return

    # P(hard) rank is already the sort order of hitters list (sorted by avg_p_hard desc)
    phhard_order = [h["name"] for h in hitters]

    xwoba_hitters = [h for h in hitters if h.get("avg_xwoba") is not None]
    if not xwoba_hitters:
        print(f"\n[{league}] xwOBA not available — model not yet trained.")
        return

    xwoba_order = sorted(xwoba_hitters, key=lambda h: -(h["avg_xwoba"] or 0))
    xwoba_names = [h["name"] for h in xwoba_order]

    print(f"\n{'─' * 64}")
    print(f"  [{league}] P(hard) rank vs xwOBA rank comparison")
    print(f"{'─' * 64}")
    print(f"  {'Hitter':<26}  {'P(hard)':<9}  {'P(hard)#':<9}  {'xwOBA':<8}  {'xwOBA#':<7}  {'Δ'}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*4}")

    for h in hitters:
        name    = h["name"]
        ph_val  = h["avg_p_hard"]
        ph_rank = phhard_order.index(name) + 1 if name in phhard_order else "?"
        xw_val  = h.get("avg_xwoba")
        if xw_val is not None and name in xwoba_names:
            xw_rank = xwoba_names.index(name) + 1
            delta   = xw_rank - ph_rank
            flag    = "  <- MOVED" if abs(delta) >= 2 else ""
            print(
                f"  {name:<26}  {ph_val:<9.3f}  #{ph_rank:<8}  "
                f"{xw_val:<8.3f}  #{xw_rank:<6}  {delta:+d}{flag}"
            )
        else:
            print(f"  {name:<26}  {ph_val:<9.3f}  #{ph_rank:<8}  {'N/A':<8}  {'N/A':<6}  n/a")
    print(f"{'─' * 64}\n")


def run() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run generate_synthetic_d1_pitcher.py first.")
        sys.exit(1)

    from src.data.trackman_ingest import load_trackman_csv, row_to_pitch_dict
    from src.model.report import generate_league_report, render_report

    tm_df  = load_trackman_csv(CSV_PATH)
    pitches = [row_to_pitch_dict(row) for _, row in tm_df.iterrows()]
    n = len(pitches)

    print(f"\nRunning {n} pitches × {len(MLB_HITTERS)} MLB hitters = {n * len(MLB_HITTERS)} predictions...")
    mlb_report = generate_league_report(
        pitches, MLB_HITTERS, league="MLB", pitcher_description=PITCHER_DESC
    )

    print(f"Running {n} pitches × {len(AAA_HITTERS)} AAA hitters = {n * len(AAA_HITTERS)} predictions...")
    aaa_report = generate_league_report(
        pitches, AAA_HITTERS, league="AAA", pitcher_description=PITCHER_DESC
    )

    # Save separate CSVs
    OUT_MLB.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mlb_report["records"]).to_csv(OUT_MLB, index=False)
    pd.DataFrame(aaa_report["records"]).to_csv(OUT_AAA, index=False)
    print(f"\nSaved → {OUT_MLB}  ({len(mlb_report['records'])} rows)")
    print(f"Saved → {OUT_AAA}  ({len(aaa_report['records'])} rows)")

    # Print separated reports — no combined table, no cross-tier summary
    print()
    print(render_report(mlb_report))
    print()
    print("=" * 64)
    print()
    print(render_report(aaa_report))
    print()

    # Rank comparison: P(hard) vs xwOBA for each league
    _print_rank_comparison("MLB", mlb_report)
    _print_rank_comparison("AAA", aaa_report)


if __name__ == "__main__":
    run()
