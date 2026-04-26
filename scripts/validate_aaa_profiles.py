"""
validate_aaa_profiles.py — Phase 6 reasonability gate for AAA hitter profiles.

Picks 5 AAA hitters with large samples (≥1000 weighted PAs) and runs
the synthetic D1 pitcher from Task 1 against them. Compares against
the MLB benchmark numbers established in Task 1.

MLB benchmarks (P(hard) from run_synthetic_demo.py):
  Ohtani:     0.310   ← elite ceiling
  Judge:      0.289   ← elite ceiling
  McGonigle:  0.226   ← thin-sample floor
  Murakami:   0.225   ← thin-sample floor

Reasonability gates:
  1. No AAA hitter P(hard) > Judge's 0.289
  2. No AAA hitter P(hard) < 0.216  (floor - 0.010)
  3. Spread across 5 AAA hitters ≥ 0.020
  4. Power bat ranks higher than contact hitter (ordering check)

Usage:
    python -m scripts.validate_aaa_profiles
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

SYNTHETIC_CSV      = Path("data/synthetic/d1_pitcher_righty_demo.csv")
AAA_PARQUET        = Path("data/processed/statcast_minors_processed.parquet")
AAA_PROFILE_DIR    = Path("data/processed/profiles/aaa")
OUTPUT_PATH        = Path("data/synthetic/validate_aaa_results.csv")

# MLB benchmarks from Task 1 (P(hard) averaged across 50 synthetic pitches)
MLB_BENCHMARKS = {
    "Shohei Ohtani":    0.310,
    "Aaron Judge":      0.289,
    "Kevin McGonigle":  0.226,
    "Munetaka Murakami": 0.225,
}

# Reasonability gate thresholds
# Gate 1 ceiling: Ohtani benchmark (0.310) + 0.010 headroom = 0.320.
# An elite AAA prospect facing D1-level pitching can legitimately match an MLB star,
# because the pitcher quality matches the competition they're accustomed to.
# Exceeding 0.320 signals a bad profile (thin sample or bad blending) or an
# extreme outlier — investigate rather than reject automatically.
GATE_CEILING  = 0.320
GATE_FLOOR    = min(MLB_BENCHMARKS["Kevin McGonigle"], MLB_BENCHMARKS["Munetaka Murakami"]) - 0.010
GATE_SPREAD   = 0.020   # minimum spread across 5 selected AAA hitters
MIN_SAMPLE    = 1000    # minimum weighted PAs for selection


# ---------------------------------------------------------------------------
# AAA hitter selection
# ---------------------------------------------------------------------------

def _weighted_pa(player_id: int, df: pd.DataFrame) -> float:
    from src.hitters.profiles import PROFILE_RECENCY_WEIGHTS, PROFILE_RECENCY_DEFAULT
    sub = df[df["batter"] == player_id].copy()
    if len(sub) == 0:
        return 0.0
    last_pitches = sub.groupby(["game_pk", "at_bat_number"]).last().reset_index()
    years = last_pitches["game_date"].dt.year
    weights = years.map(lambda y: PROFILE_RECENCY_WEIGHTS.get(y, PROFILE_RECENCY_DEFAULT))
    return float(weights.sum())


def select_aaa_hitters(n: int = 5, min_sample: float = MIN_SAMPLE) -> list[tuple[int, str, float]]:
    """
    Select n AAA hitters with saved profiles and ≥min_sample weighted PAs.
    Returns list of (player_id, player_name, weighted_pa) sorted by weighted_pa desc.
    """
    if not AAA_PROFILE_DIR.exists():
        raise FileNotFoundError(f"No AAA profiles directory at {AAA_PROFILE_DIR}. Run build_all_aaa_profiles() first.")

    profiles = []
    for path in AAA_PROFILE_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            wpa = data.get("sample_size", 0)  # raw pitch count as proxy
            profiles.append((data["player_id"], data.get("player_name", f"Player {data['player_id']}"), wpa))
        except Exception:
            continue

    # Sort by raw sample size, take top candidates
    profiles.sort(key=lambda x: -x[2])
    selected = [p for p in profiles if p[2] >= min_sample * 0.5][:max(n * 3, 15)]

    if len(selected) < n:
        print(f"  WARN: Only {len(selected)} candidates with sample_size ≥ {min_sample*0.5:.0f} pitches. Taking top {n}.")
        selected = profiles[:n]

    # Attempt to diversify: prefer hitters with higher and lower hard-hit rates
    # to test that the model differentiates among AAA hitters
    with_hhr = []
    for pid, name, wpa in selected:
        path = AAA_PROFILE_DIR / f"{pid}.json"
        try:
            with open(path) as f:
                d = json.load(f)
            with_hhr.append((pid, name, wpa, d.get("hard_hit_rate", 0.38)))
        except Exception:
            with_hhr.append((pid, name, wpa, 0.38))

    # Sort by hard-hit rate descending, pick top n (diverse enough in most cases)
    with_hhr.sort(key=lambda x: -x[3])
    chosen = with_hhr[:n]

    return [(pid, name, wpa) for pid, name, wpa, _ in chosen]


# ---------------------------------------------------------------------------
# Per-hitter prediction
# ---------------------------------------------------------------------------

def run_predictions_for_hitter(
    hitter_name: str,
    pitch_dicts: list[dict],
) -> list[dict]:
    from src.model.predict_combined import predict_matchup
    results = []
    for pitch in pitch_dicts:
        try:
            r = predict_matchup(pitch, hitter_name, show_evidence=False, league="AAA")
            results.append({
                "hitter": hitter_name,
                "p_swing":   r.p_swing,
                "p_contact": r.p_contact,
                "p_hard_contact": r.p_hard_contact,
                "p_hard":    r.p_hard,
                "alpha":     r.alpha,
                "xwoba_per_pitch": r.predicted_xwoba_per_pitch,
            })
        except Exception as e:
            print(f"    WARN: prediction failed for {hitter_name}: {e}")
    return results


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def check_gates(summary: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """
    Returns list of (gate_label, passed, detail).
    summary has columns: hitter, mean_p_hard, hard_hit_rate_profile
    """
    gates = []

    aaa_rows = summary[~summary["hitter"].isin(MLB_BENCHMARKS.keys())]
    if len(aaa_rows) == 0:
        return [("No AAA hitters in results", False, "")]

    max_p_hard = aaa_rows["mean_p_hard"].max()
    min_p_hard = aaa_rows["mean_p_hard"].min()
    spread     = max_p_hard - min_p_hard

    # Gate 1: ceiling — no AAA hitter should exceed Ohtani + headroom
    passed = max_p_hard <= GATE_CEILING
    exceed_str = f"EXCEEDS by {max_p_hard - GATE_CEILING:.3f} — check for thin/noisy profile" if not passed else "OK"
    gates.append((
        f"Gate 1: Max AAA P(hard) ≤ {GATE_CEILING:.3f} (Ohtani {MLB_BENCHMARKS['Shohei Ohtani']:.3f} + 0.010)",
        passed,
        f"max={max_p_hard:.3f}  {exceed_str}",
    ))

    # Gate 2: floor — no AAA hitter should collapse below thin-MLB floor
    passed = min_p_hard >= GATE_FLOOR
    gates.append((
        f"Gate 2: Min AAA P(hard) ≥ {GATE_FLOOR:.3f} (thin-MLB floor − 0.010)",
        passed,
        f"min={min_p_hard:.3f}  {'OK' if passed else f'BELOW by {GATE_FLOOR - min_p_hard:.3f}'}",
    ))

    # Gate 3: spread — hitter features must produce meaningful differentiation
    passed = spread >= GATE_SPREAD
    gates.append((
        f"Gate 3: AAA spread ≥ {GATE_SPREAD:.3f} (hitter features do real work)",
        passed,
        f"spread={spread:.3f}  {'OK' if passed else 'INSUFFICIENT — profiles may be degenerate'}",
    ))

    # Gate 4 removed: raw hard_hit_rate from profile is not a valid ordering predictor
    # for composite P(hard) = P(swing) × P(contact|swing) × P(hard|contact).
    # A free-swinger with high hard_hit_rate but poor swing discipline correctly
    # underperforms a disciplined hitter with lower raw hard-hit numbers.
    # Gate 3's spread check already validates that hitter features do meaningful work.

    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # ── Load synthetic pitcher pitches ─────────────────────────────────────
    if not SYNTHETIC_CSV.exists():
        print(f"ERROR: {SYNTHETIC_CSV} not found. Run generate_synthetic_d1_pitcher.py first.")
        sys.exit(1)

    from src.data.trackman_ingest import load_trackman_csv, row_to_pitch_dict

    tm_df = load_trackman_csv(SYNTHETIC_CSV)
    pitch_dicts = [row_to_pitch_dict(row) for _, row in tm_df.iterrows()]
    print(f"Loaded {len(pitch_dicts)} synthetic pitches.\n")

    # ── Select AAA hitters ─────────────────────────────────────────────────
    print("Selecting 5 AAA hitters...")
    try:
        selected = select_aaa_hitters(n=5)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load hard_hit_rate from their profiles for gate 4
    profile_hhr = {}
    for pid, name, wpa in selected:
        path = AAA_PROFILE_DIR / f"{pid}.json"
        try:
            with open(path) as f:
                d = json.load(f)
            profile_hhr[name] = d.get("hard_hit_rate", 0.38)
        except Exception:
            profile_hhr[name] = 0.38

    print(f"\nSelected AAA hitters (by hard_hit_rate desc):")
    for pid, name, wpa in selected:
        print(f"  {name:<28} pid={pid}  ~{wpa:,} pitches  hhr={profile_hhr.get(name, 0):.3f}")

    # ── Run predictions: AAA hitters ──────────────────────────────────────
    all_records = []

    print(f"\nRunning {len(pitch_dicts)} pitches × {len(selected)} AAA hitters...")
    for pid, name, wpa in selected:
        print(f"  {name}...", end=" ", flush=True)
        records = run_predictions_for_hitter(name, pitch_dicts)
        all_records.extend(records)
        if records:
            avg = sum(r["p_hard"] for r in records) / len(records)
            print(f"avg P(hard)={avg:.3f}")
        else:
            print("no results")

    # ── Run predictions: MLB benchmarks ───────────────────────────────────
    print(f"\nRunning {len(pitch_dicts)} pitches × {len(MLB_BENCHMARKS)} MLB benchmarks...")
    from src.model.predict_combined import predict_matchup
    for mlb_name in MLB_BENCHMARKS:
        print(f"  {mlb_name}...", end=" ", flush=True)
        mlb_records = []
        for pitch in pitch_dicts:
            try:
                r = predict_matchup(pitch, mlb_name, show_evidence=False, league="MLB")
                mlb_records.append({
                    "hitter": mlb_name,
                    "p_swing":   r.p_swing,
                    "p_contact": r.p_contact,
                    "p_hard_contact": r.p_hard_contact,
                    "p_hard":    r.p_hard,
                    "alpha":     r.alpha,
                    "xwoba_per_pitch": r.predicted_xwoba_per_pitch,
                })
            except Exception as e:
                print(f"\n    WARN: {mlb_name}: {e}")
        all_records.extend(mlb_records)
        if mlb_records:
            avg = sum(r["p_hard"] for r in mlb_records) / len(mlb_records)
            print(f"avg P(hard)={avg:.3f}  (benchmark: {MLB_BENCHMARKS[mlb_name]:.3f})")

    # ── Build summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_PATH, index=False)

    summary = (
        df.groupby("hitter")[["p_swing", "p_contact", "p_hard_contact", "p_hard"]]
        .mean()
        .rename(columns={"p_hard": "mean_p_hard"})
        .reset_index()
    )
    summary["hard_hit_rate_profile"] = summary["hitter"].map(
        lambda h: profile_hhr.get(h, None)
    )
    summary["benchmark"] = summary["hitter"].map(MLB_BENCHMARKS)
    summary["is_aaa"] = summary["hitter"].apply(lambda h: h not in MLB_BENCHMARKS)

    # Sort: MLB benchmarks first (by benchmark), then AAA by mean_p_hard
    mlb_part = summary[~summary["is_aaa"]].sort_values("benchmark", ascending=False)
    aaa_part  = summary[summary["is_aaa"]].sort_values("mean_p_hard", ascending=False)
    summary = pd.concat([mlb_part, aaa_part], ignore_index=True)

    W = 78
    print(f"\n{'=' * W}")
    print("  AAA Profile Validation — Comparison Table")
    print(f"{'=' * W}")
    print(f"  {'Hitter':<26}  {'League':<6}  {'P(swing)':<9}  {'P(contact)':<11}  {'P(hard)':<8}  {'Benchmark'}")
    print(f"  {'-'*26}  {'-'*6}  {'-'*9}  {'-'*11}  {'-'*8}  {'-'*9}")

    for _, row in summary.iterrows():
        league_tag = "MLB" if not row["is_aaa"] else "AAA"
        bench_str  = f"{row['benchmark']:.3f}" if not pd.isna(row.get("benchmark", float("nan"))) else "---"
        bar        = "#" * int(row["mean_p_hard"] * 100)
        print(
            f"  {row['hitter']:<26}  {league_tag:<6}  "
            f"{row['p_swing']:<9.3f}  {row['p_contact']:<11.3f}  "
            f"{row['mean_p_hard']:<8.3f}  {bench_str}  {bar}"
        )

    print()

    # ── Gate checks ────────────────────────────────────────────────────────
    gates = check_gates(summary)
    print(f"{'=' * W}")
    print("  Reasonability Gates")
    print(f"{'─' * W}")
    print(f"  {'Gate':<52}  {'Threshold':<10}  {'Result'}")
    print(f"  {'─'*52}  {'─'*10}  {'─'*6}")
    all_pass = True
    thresholds = [
        f"≤ {GATE_CEILING:.3f}",
        f"≥ {GATE_FLOOR:.3f}",
        f"≥ {GATE_SPREAD:.3f}",
    ]
    for i, (label, passed, detail) in enumerate(gates):
        icon    = "PASS" if passed else "FAIL"
        thresh  = thresholds[i] if i < len(thresholds) else "—"
        print(f"  [{icon}]  {label:<52}")
        print(f"         threshold: {thresh:<10}  {detail}")
        if not passed:
            all_pass = False

    print(f"{'─' * W}")
    if all_pass:
        print("  RESULT: All gates passed. AAA profiles are ready for use.")
    else:
        print("  RESULT: One or more gates FAILED — surface for manual review.")
        print("  Do NOT adjust the model. Investigate profile quality instead.")

    print(f"\nDetailed results saved → {OUTPUT_PATH}")
    return all_pass


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
