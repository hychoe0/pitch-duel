"""
demo_battery.py — V5 model validation against DEMO_HITTERS battery.

Runs 50 synthetic D1 RHP pitches (15 FF / 15 FC / 10 SL / 10 CH, seed=42)
against all 15 DEMO_HITTERS and computes avg P(hard) per hitter.

Outputs:
  validation/v5_demo_battery.csv
  validation/v5_differentiation.md

Usage:
    python -m validation.demo_battery
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

CSV_PATH  = Path("data/synthetic/d1_pitcher_righty_demo.csv")
OUT_CSV   = Path("validation/v5_demo_battery.csv")
OUT_DIFF  = Path("validation/v5_differentiation.md")

# V4 benchmarks from CLAUDE.md validation table (seed=42 synthetic D1 RHP)
V4_BENCHMARKS = {
    "Shohei Ohtani":    0.310,
    "Jackson Merrill":  0.295,
    "Aaron Judge":      0.289,
    "Francisco Lindor": 0.259,
    "Kevin McGonigle":  0.226,
    "Munetaka Murakami": 0.225,
    "José Ramírez":     0.206,
}

DEMO_HITTERS = [
    "Judge, Aaron",
    "Betts, Mookie",
    "Kiner-Falefa, Isiah",
    "Ohtani, Shohei",
    "Raleigh, Cal",
    "Soto, Juan",
    "Witt, Bobby",
    "Duran, Jarren",
    "Carroll, Corbin",
    "Henderson, Gunnar",
    "Volpe, Anthony",
    "Caissie, Owen",
    "Winn, Masyn",
    "Langford, Wyatt",
    "Lee, Jung Hoo",
]

# V4 benchmark name mapping (evaluate_expanded uses "First Last" format)
_NAME_MAP_EVAL = {
    "Judge, Aaron":      "Aaron Judge",
    "Ohtani, Shohei":    "Shohei Ohtani",
    "Lee, Jung Hoo":     "Lee Jung Hoo",
}


def run() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    from src.data.trackman_ingest import load_trackman_csv, row_to_pitch_dict
    from src.model.report import generate_league_report

    tm_df   = load_trackman_csv(CSV_PATH)
    pitches = [row_to_pitch_dict(row) for _, row in tm_df.iterrows()]

    print(f"\nRunning {len(pitches)} pitches × {len(DEMO_HITTERS)} hitters...")
    report = generate_league_report(
        pitches, DEMO_HITTERS, league="MLB",
        pitcher_description="Synthetic mid-tier D1 RHP — seed=42",
    )

    hitters = report["hitters"]
    rows = []
    for h in hitters:
        rows.append({
            "hitter":         h["name"],
            "avg_p_swing":    h.get("avg_p_swing"),
            "avg_p_contact":  h.get("avg_p_contact"),
            "avg_p_hard":     h.get("avg_p_hard"),
            "avg_xwoba":      h.get("avg_xwoba"),
            "is_thin_sample": h.get("is_thin_sample"),
        })

    df = pd.DataFrame(rows).sort_values("avg_p_hard", ascending=False).reset_index(drop=True)
    df["rank_v5"] = df.index + 1
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}\n")

    # Print table
    print(f"{'Rank':<5} {'Hitter':<26} {'P(hard)':<9} {'xwOBA':<8} {'thin?'}")
    print(f"{'─'*5} {'─'*26} {'─'*9} {'─'*8} {'─'*5}")
    for _, row in df.iterrows():
        thin = "thin" if row["is_thin_sample"] else ""
        print(
            f"  #{int(row['rank_v5']):<3} {row['hitter']:<26} "
            f"{row['avg_p_hard']:.3f}    {(row['avg_xwoba'] or 0.0):.3f}    {thin}"
        )

    # Differentiation metrics
    p_hard_vals = df["avg_p_hard"].dropna()
    spread      = p_hard_vals.max() - p_hard_vals.min()
    std_dev     = p_hard_vals.std()

    print(f"\nDifferentiation: spread={spread:.3f}  std={std_dev:.3f}")
    print(f"Min P(hard): {p_hard_vals.min():.3f}  Max P(hard): {p_hard_vals.max():.3f}")

    # Write differentiation report
    with open(OUT_DIFF, "w") as f:
        f.write("# V5 Hitter Differentiation\n\n")
        f.write(f"## Battery: 50 synthetic D1 RHP pitches (seed=42)\n\n")
        f.write(f"| Rank | Hitter | P(hard) | xwOBA | Thin? |\n")
        f.write(f"|------|--------|---------|-------|-------|\n")
        for _, row in df.iterrows():
            thin = "yes" if row["is_thin_sample"] else ""
            f.write(
                f"| #{int(row['rank_v5'])} | {row['hitter']} | {row['avg_p_hard']:.3f} "
                f"| {(row['avg_xwoba'] or 0.0):.3f} | {thin} |\n"
            )
        f.write(f"\n## Differentiation Metrics\n\n")
        f.write(f"- **Spread** (max − min): {spread:.3f}\n")
        f.write(f"- **Std dev**: {std_dev:.3f}\n")
        f.write(f"- **Min P(hard)**: {p_hard_vals.min():.3f}\n")
        f.write(f"- **Max P(hard)**: {p_hard_vals.max():.3f}\n")

    print(f"\nSaved → {OUT_DIFF}")


if __name__ == "__main__":
    run()
