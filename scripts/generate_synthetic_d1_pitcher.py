"""
generate_synthetic_d1_pitcher.py — Synthetic Trackman CSV for a mid-tier D1 RHP.

50 pitches: 15 FF + 15 FC + 10 SL + 10 CH.
Grouped into ~15 mini plate appearances (2–5 pitches each).
Output: data/synthetic/d1_pitcher_righty_demo.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR  = Path("data/synthetic")
OUTPUT_FILE = OUTPUT_DIR / "d1_pitcher_righty_demo.csv"

# ── Pitch specs (Trackman units: inches for IVB/HorzBreak) ───────────────────
# HorzBreak positive = arm-side for RHP (toward first base)
# IVB positive = upward induced break

PITCH_SPECS = {
    "FF": {
        "count": 15,
        "tagged": "Fastball",
        "velo":   (92.5, 1.0),
        "spin":   (2150, 100),
        "ivb":    (17.0, 1.0),
        "hb":     (10.0, 1.5),
    },
    "FC": {
        "count": 15,
        "tagged": "Cutter",
        "velo":   (90.0, 1.0),
        "spin":   (2200, 100),
        "ivb":    ( 8.0, 1.0),
        "hb":     (-2.0, 1.0),
    },
    "SL": {
        "count": 10,
        "tagged": "Slider",
        "velo":   (84.0, 1.0),
        "spin":   (2250, 100),
        "ivb":    ( 2.0, 1.0),
        "hb":     (-6.0, 1.5),
    },
    "CH": {
        "count": 10,
        "tagged": "Changeup",
        "velo":   (84.0, 1.0),
        "spin":   (1900, 100),
        "ivb":    ( 8.0, 1.0),
        "hb":     (12.0, 1.5),
    },
}

# Release parameters (consistent single pitcher)
REL_HEIGHT = (6.0,  0.10)   # RelHeight feet
REL_SIDE   = (-1.8, 0.05)   # RelSide feet, negative = RHP
EXTENSION  = (6.3,  0.10)   # Extension feet


def _sample_plate_location(pt: str) -> tuple[float, float]:
    """Sample realistic (PlateLocSide, PlateLocHeight) for pitch type."""
    if pt == "FF":
        x = rng.normal(0.10, 0.40)
        z = rng.normal(2.80, 0.50)
    elif pt == "FC":
        x = rng.normal(-0.20, 0.35)
        z = rng.normal(2.40, 0.45)
    elif pt == "SL":
        x = rng.normal(-0.50, 0.45)
        z = rng.normal(1.90, 0.50)
    else:  # CH
        x = rng.normal(0.30, 0.40)
        z = rng.normal(2.00, 0.50)
    return float(np.clip(x, -2.0, 2.0)), float(np.clip(z, 0.5, 5.0))


def _build_pitch_pool() -> list[dict]:
    """Generate 50 pitch rows (unsequenced). Stores _pt for internal use."""
    rows = []
    for pt, spec in PITCH_SPECS.items():
        n = spec["count"]
        velos  = rng.normal(*spec["velo"], size=n)
        spins  = rng.normal(*spec["spin"], size=n)
        ivbs   = rng.normal(*spec["ivb"],  size=n)
        hbs    = rng.normal(*spec["hb"],   size=n)
        rels_z = rng.normal(*REL_HEIGHT,   size=n)
        rels_x = rng.normal(*REL_SIDE,     size=n)
        exts   = rng.normal(*EXTENSION,    size=n)
        for i in range(n):
            px, pz = _sample_plate_location(pt)
            rows.append({
                "_pt":             pt,
                "TaggedPitchType": spec["tagged"],
                "AutoPitchType":   spec["tagged"],
                "PitcherThrows":   "Right",
                "Pitcher":         "Demo, RHP",
                "Batter":          "Lineup, Avg",
                "RelSpeed":        round(float(velos[i]), 1),
                "SpinRate":        int(round(float(spins[i]))),
                "InducedVertBreak": round(float(ivbs[i]), 1),
                "HorzBreak":        round(float(hbs[i]),  1),
                "RelHeight":       round(float(rels_z[i]), 2),
                "RelSide":         round(float(rels_x[i]), 2),
                "Extension":       round(float(exts[i]),   2),
                "PlateLocSide":    round(px, 3),
                "PlateLocHeight":  round(pz, 3),
            })
    return rows


def _assign_sequences(rows: list[dict]) -> pd.DataFrame:
    """
    Shuffle rows, group into ~15 PAs, assign Inning/PAofInning/Balls/Strikes/
    PitchNo/PitchCall. Balls and Strikes record the count BEFORE the pitch.
    """
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    rows = [rows[i] for i in idx]

    # Build PA sizes summing to exactly len(rows)
    pa_sizes = []
    remaining = len(rows)
    while remaining > 0:
        s = int(rng.integers(2, 6))
        s = min(s, remaining)
        pa_sizes.append(s)
        remaining -= s

    records = []
    pitch_no = 1
    inning   = 1
    pa_of_inning = 1
    pitch_idx = 0

    for pa_i, pa_size in enumerate(pa_sizes):
        if pa_i > 0 and pa_i % 5 == 0:
            inning += 1
            pa_of_inning = 1

        balls   = 0
        strikes = 0

        for p_in_pa in range(pa_size):
            row = rows[pitch_idx].copy()
            pitch_idx += 1
            is_last = (p_in_pa == pa_size - 1)

            # Save count BEFORE the pitch
            count_balls   = balls
            count_strikes = strikes

            # Choose PitchCall
            if is_last:
                if strikes == 2 and balls == 3:
                    call = rng.choice(
                        ["StrikeSwinging", "StrikeCalled", "InPlay", "BallCalled"],
                        p=[0.30, 0.15, 0.35, 0.20])
                elif strikes == 2:
                    call = rng.choice(
                        ["StrikeSwinging", "StrikeCalled", "FoulBall", "InPlay", "BallCalled"],
                        p=[0.30, 0.15, 0.00, 0.30, 0.25])
                elif balls == 3:
                    call = rng.choice(
                        ["BallCalled", "InPlay", "StrikeSwinging"],
                        p=[0.40, 0.35, 0.25])
                else:
                    call = rng.choice(
                        ["StrikeCalled", "StrikeSwinging", "FoulBall", "BallCalled", "InPlay"],
                        p=[0.15, 0.20, 0.15, 0.25, 0.25])
            else:
                if strikes < 2:
                    call = rng.choice(
                        ["StrikeCalled", "StrikeSwinging", "FoulBall", "BallCalled"],
                        p=[0.20, 0.20, 0.20, 0.40])
                else:  # 2 strikes — foul balls keep PA alive
                    call = rng.choice(
                        ["FoulBall", "BallCalled", "StrikeSwinging"],
                        p=[0.35, 0.40, 0.25])

            # Update count for NEXT pitch in this PA
            if call in ("StrikeCalled", "StrikeSwinging"):
                strikes += 1
            elif call == "BallCalled":
                balls += 1
            elif call == "FoulBall" and strikes < 2:
                strikes += 1

            row.update({
                "Inning":       inning,
                "PAofInning":   pa_of_inning,
                "Balls":        count_balls,
                "Strikes":      count_strikes,
                "PitchNo":      pitch_no,
                "PitchCall":    call,
                "HomeScore":    0,
                "AwayScore":    0,
                "Top/Bottom":   "Top",
                "Runner1B":     None,
                "Runner2B":     None,
                "Runner3B":     None,
            })
            records.append(row)
            pitch_no += 1

        pa_of_inning += 1

    df = pd.DataFrame(records)
    df.drop(columns=["_pt"], inplace=True, errors="ignore")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = _build_pitch_pool()
    df   = _assign_sequences(pool)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nGenerated {len(df)} pitches → {OUTPUT_FILE}")
    print("\nPitch type distribution:")
    print(df["TaggedPitchType"].value_counts().to_string())
    print("\nCount distribution (Balls-Strikes):")
    df["count"] = df["Balls"].astype(str) + "-" + df["Strikes"].astype(str)
    print(df["count"].value_counts().sort_index().to_string())
    print("\nSample rows:")
    cols = ["TaggedPitchType", "RelSpeed", "SpinRate", "Balls", "Strikes", "PitchCall",
            "PlateLocSide", "PlateLocHeight"]
    print(df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
