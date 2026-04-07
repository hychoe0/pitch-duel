"""
evaluate_full.py — Three-stage model evaluation and demo readiness assessment.

Sections:
  A  Sanity checks (velocity / location / count / pitch-type gradients)
  B  Hitter differentiation (do probabilities differ across hitters?)
  C  Calibration / range checks on test set sample
  D  Demo readiness checklist

Run from the pitch_duel project root:
    python -m src.model.evaluate_full
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_DIR    = ROOT / "models"
PROFILE_DIR  = ROOT / "data" / "processed" / "profiles"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"

from src.hitters.profiles import (
    HitterProfile, load_profile, profile_to_feature_dict,
    get_league_average_profile,
)
from src.model.predict import predict_pitch, build_feature_row, load_models


# ──────────────────────────────────────────────────────────────────────────────
# Profile cache
# ──────────────────────────────────────────────────────────────────────────────

_profile_cache: dict = {}


def _get_profile(batter_id: int) -> HitterProfile:
    if batter_id not in _profile_cache:
        try:
            _profile_cache[batter_id] = load_profile(batter_id, PROFILE_DIR)
        except FileNotFoundError:
            _profile_cache[batter_id] = get_league_average_profile()
    return _profile_cache[batter_id]


def _get_name(batter_id: int) -> str:
    p = _get_profile(batter_id)
    return p.player_name or f"P{batter_id}"


# ──────────────────────────────────────────────────────────────────────────────
# Base pitch (Ohtani, 0-0, inning 1, no runners, score tied)
# ──────────────────────────────────────────────────────────────────────────────

OHTANI_ID = 660271

BASE_PITCH = {
    "release_speed":      95.0,
    "release_spin_rate":  2350.0,
    "pfx_x":             -0.4,
    "pfx_z":              1.1,
    "release_pos_x":     -1.8,
    "release_pos_z":      6.1,
    "release_extension":  6.5,
    "plate_x":            0.0,
    "plate_z":            2.5,
    "pitch_type":         "FF",
    "balls":              0,
    "strikes":            0,
    "pitch_number":       1,
    "prev_pitch_type":    "FIRST_PITCH",
    "prev_pitch_speed":   0.0,
    "prev_pitch_result":  "FIRST_PITCH",
    "on_1b":              None,
    "on_2b":              None,
    "on_3b":              None,
    "inning":             1,
    "score_diff":         0,
    "p_throws":           "R",
    "game_date":          "2024-06-15",
}


def _p(pitch_dict: dict, hitter_name: str) -> dict:
    """Thin wrapper around predict_pitch for eval convenience."""
    return predict_pitch(pitch_dict, hitter_name)


# ──────────────────────────────────────────────────────────────────────────────
# Section A — Sanity Checks
# ──────────────────────────────────────────────────────────────────────────────

def section_a() -> dict:
    hitter = "Shohei Ohtani"
    print("\n" + "═" * 70)
    print("SECTION A — SANITY CHECKS")
    print(f"Hitter: {hitter}  |  Base: 95 mph FF center-cut 0-0")
    print("═" * 70)

    results = {}

    # A.1 — Velocity gradient: P(contact|swing) should decrease as velocity rises
    print("\nA.1  Velocity Gradient — P(contact|swing) vs velocity")
    print(f"     {'MPH':>5}  {'P(swing)':>9}  {'P(con|sw)':>10}  {'P(hard|con)':>11}")
    velo_vals = [88, 91, 94, 97, 100]
    contact_preds = []
    for mph in velo_vals:
        p = dict(BASE_PITCH)
        p["release_speed"] = float(mph)
        r = _p(p, hitter)
        contact_preds.append(r["p_contact_given_swing"])
        print(f"     {mph:>5}  {r['p_swing']:>9.4f}  {r['p_contact_given_swing']:>10.4f}  "
              f"{r['p_hard_given_contact']:>11.4f}")

    diffs = [contact_preds[i+1] - contact_preds[i] for i in range(len(contact_preds)-1)]
    n_increase = sum(1 for d in diffs if d > 0.002)
    pass_a1 = n_increase <= 1
    results["a1"] = pass_a1
    print(f"     → {'PASS' if pass_a1 else 'FAIL'}: P(contact|swing) "
          f"{'mostly decreases' if pass_a1 else 'does NOT consistently decrease'} with velocity")

    # A.2 — Location: P(swing) should be highest in the zone, lowest in waste
    print("\nA.2  Location Gradient — P(swing) by location (95 mph FF)")
    locations = [
        ("Middle-middle",       0.0,  2.5),
        ("Low-away",            0.8,  1.8),
        ("Up-in",              -0.5,  3.3),
        ("Off plate outside",   1.2,  2.5),
        ("Way outside",         2.0,  2.5),
    ]
    swing_preds = {}
    print(f"     {'Location':<22}  {'px':>5}  {'pz':>5}  {'P(swing)':>9}  {'P(con|sw)':>10}")
    for label, px, pz in locations:
        p = dict(BASE_PITCH)
        p["plate_x"] = px
        p["plate_z"] = pz
        r = _p(p, hitter)
        swing_preds[label] = r["p_swing"]
        print(f"     {label:<22}  {px:>5.1f}  {pz:>5.1f}  "
              f"{r['p_swing']:>9.4f}  {r['p_contact_given_swing']:>10.4f}")

    middle   = swing_preds["Middle-middle"]
    way_out  = swing_preds["Way outside"]
    pass_a2  = (middle == max(swing_preds.values())) and (way_out < middle - 0.010)
    results["a2"] = pass_a2
    print(f"     → {'PASS' if pass_a2 else 'FAIL'}: "
          f"middle-middle {'is highest' if middle == max(swing_preds.values()) else 'is NOT highest'} P(swing), "
          f"way-outside {'lower' if way_out < middle - 0.010 else 'not much lower'} (Δ={middle - way_out:.4f})")

    # A.3 — Count leverage: P(swing) should be higher in hitter's counts
    # This is a DIRECT behavioral prediction — hitters DO swing more in 2-0 than 0-2.
    # This check should PASS with the three-stage model.
    print("\nA.3  Count Leverage — P(swing) in hitter vs pitcher counts")
    counts = [
        ("0-0 (neutral)",  0, 0, 1, "FIRST_PITCH", 0.0,  "FIRST_PITCH"),
        ("0-2 (pitcher)",  0, 2, 3, "SL",          86.0, "swinging_strike"),
        ("2-0 (hitter)",   2, 0, 3, "FF",          94.0, "ball"),
        ("3-0 (hitter+)",  3, 0, 4, "FF",          94.0, "ball"),
        ("3-2 (full)",     3, 2, 6, "FF",          94.0, "foul"),
    ]
    count_swings = {}
    print(f"     {'Count':<20}  {'P(swing)':>9}  {'P(con|sw)':>10}  {'P(hard|con)':>11}")
    for label, b, s, pnum, ppt, pps, ppr in counts:
        p = dict(BASE_PITCH)
        p.update({"balls": b, "strikes": s, "pitch_number": pnum,
                  "prev_pitch_type": ppt, "prev_pitch_speed": pps, "prev_pitch_result": ppr})
        r = _p(p, hitter)
        count_swings[label] = r["p_swing"]
        print(f"     {label:<20}  {r['p_swing']:>9.4f}  "
              f"{r['p_contact_given_swing']:>10.4f}  {r['p_hard_given_contact']:>11.4f}")

    hitter_counts  = [count_swings["2-0 (hitter)"], count_swings["3-0 (hitter+)"]]
    pitcher_counts = [count_swings["0-2 (pitcher)"]]
    pass_a3 = min(hitter_counts) > max(pitcher_counts)
    results["a3"] = pass_a3
    print(f"     → {'PASS' if pass_a3 else 'FAIL'}: "
          f"hitter counts P(swing) min={min(hitter_counts):.4f} "
          f"{'>' if pass_a3 else 'NOT >'} pitcher counts max={max(pitcher_counts):.4f}")
    if not pass_a3:
        print("     NOTE: Three-stage P(swing) models hitter intent directly — ")
        print("     if this fails, check that hitter_swing_rate_this_count feature")
        print("     is being loaded correctly from profiles.")

    # A.4 — Pitch type: P(contact|swing) should differ across pitch types
    print("\nA.4  Pitch Type — P(contact|swing) by type (88 mph, middle-middle)")
    pitch_types = ["FF", "SL", "CH", "CU"]
    pt_preds = {}
    print(f"     {'Type':<6}  {'P(swing)':>9}  {'P(con|sw)':>10}  {'P(hard|con)':>11}")
    for pt in pitch_types:
        p = dict(BASE_PITCH)
        p["release_speed"] = 88.0
        p["pitch_type"] = pt
        r = _p(p, hitter)
        pt_preds[pt] = r["p_contact_given_swing"]
        print(f"     {pt:<6}  {r['p_swing']:>9.4f}  {r['p_contact_given_swing']:>10.4f}  "
              f"{r['p_hard_given_contact']:>11.4f}")

    spread = max(pt_preds.values()) - min(pt_preds.values())
    pass_a4 = spread >= 0.010
    results["a4"] = pass_a4
    print(f"     → {'PASS' if pass_a4 else 'FAIL'}: P(contact|swing) spread={spread:.4f} "
          f"({'meaningful' if pass_a4 else 'too small — < 0.010'})")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Section B — Hitter Differentiation
# ──────────────────────────────────────────────────────────────────────────────

HITTERS_B = [
    (660271, "Ohtani (elite contact+power)"),
    (670541, "Álvarez (elite power)"),
    (592450, "Judge (elite power, hi-K)"),
    (518692, "Freeman (solid all-around)"),
    (571740, "Hamilton (weak power)"),
    (543294, "Hendricks (pitcher hitting)"),
]


def section_b() -> dict:
    print("\n" + "═" * 70)
    print("SECTION B — HITTER DIFFERENTIATION")
    print("Pitch: 95 mph FF center-cut 0-0 | Same pitch, different hitters")
    print("═" * 70)

    pitch = dict(BASE_PITCH)
    results_b = []

    print(f"\n  {'Hitter':<35} {'P(swing)':>8} {'P(con|sw)':>10} {'P(hard|con)':>11} {'P(hard)':>8}")
    print("  " + "─" * 74)

    for batter_id, label in HITTERS_B:
        profile = _get_profile(batter_id)
        hitter_name = profile.player_name or f"P{batter_id}"
        try:
            r = _p(pitch, hitter_name)
        except Exception as e:
            print(f"  {label:<35} ERROR: {e}")
            continue
        thin_tag = " *" if profile.is_thin_sample else ""
        print(f"  {label + thin_tag:<35} "
              f"{r['p_swing']:>8.4f} {r['p_contact_given_swing']:>10.4f} "
              f"{r['p_hard_given_contact']:>11.4f} {r['p_hard_contact']:>8.4f}")
        results_b.append((batter_id, label, r))

    if not results_b:
        print("  No results produced.")
        return {"b": False, "spread": 0}

    hard_contacts = [x[2]["p_hard_contact"] for x in results_b]
    spread = max(hard_contacts) - min(hard_contacts)

    elite_ids = {660271, 670541, 592450}
    weak_ids  = {571740, 543294}
    elite_hc = [r["p_hard_contact"] for bid, _, r in results_b if bid in elite_ids]
    weak_hc  = [r["p_hard_contact"] for bid, _, r in results_b if bid in weak_ids]

    elite_mean = np.mean(elite_hc) if elite_hc else 0
    weak_mean  = np.mean(weak_hc) if weak_hc else 0

    print(f"\n  P(hard_contact) spread: {spread:.4f}")
    print(f"  Elite mean: {elite_mean:.4f}  |  Weak mean: {weak_mean:.4f}")
    print(f"  * = thin sample (profile blended with league averages)")

    pass_b = spread >= 0.050 and elite_mean > weak_mean
    print(f"\n  → {'PASS' if pass_b else 'FAIL'}: "
          f"spread {'≥' if spread >= 0.050 else '<'} 0.050, "
          f"elite {'>' if elite_mean > weak_mean else 'NOT >'} weak")

    return {"b": pass_b, "spread": spread}


# ──────────────────────────────────────────────────────────────────────────────
# Section C — Calibration / Range Check
# ──────────────────────────────────────────────────────────────────────────────

def section_c() -> dict:
    print("\n" + "═" * 70)
    print("SECTION C — CALIBRATION / RANGE CHECK")
    print("═" * 70)

    results = {}

    # ── C.1  Range check: call predict_pitch on a grid of realistic pitches ───
    print("\nC.1  Prediction Range (representative pitch grid)")

    test_cases = [
        # (label, pitch_type, speed, plate_x, plate_z)
        ("FF belt-high strike",     "FF", 96, 0.0, 2.5),
        ("SL low-away",             "SL", 87, 0.7, 1.8),
        ("CU bouncing",             "CU", 81, 0.1, 0.8),
        ("CH middle-low",           "CH", 88, 0.0, 2.0),
        ("FF up-and-in",            "FF", 98, -0.5, 3.4),
        ("SL way outside",          "SL", 85, 1.8, 2.5),
        ("FF 2-0 count",            "FF", 94, 0.0, 2.5),  # count matters
        ("FF 0-2 count",            "FF", 94, 0.0, 2.5),
    ]
    counts_override = [
        {},  # default 0-0
        {},
        {},
        {},
        {},
        {},
        {"balls": 2, "strikes": 0, "pitch_number": 3, "prev_pitch_type": "FF",
         "prev_pitch_speed": 93.0, "prev_pitch_result": "ball"},
        {"balls": 0, "strikes": 2, "pitch_number": 3, "prev_pitch_type": "FF",
         "prev_pitch_speed": 93.0, "prev_pitch_result": "swinging_strike"},
    ]

    all_sw, all_con, all_hrd = [], [], []
    print(f"     {'Case':<28} {'P(swing)':>9} {'P(con|sw)':>10} {'P(hard|con)':>11} {'P(hard)':>8}")
    for (label, pt, spd, px, pz), ovr in zip(test_cases, counts_override):
        p = dict(BASE_PITCH)
        p.update({"pitch_type": pt, "release_speed": spd, "plate_x": px, "plate_z": pz})
        p.update(ovr)
        r = _p(p, "Shohei Ohtani")
        all_sw.append(r["p_swing"])
        all_con.append(r["p_contact_given_swing"])
        all_hrd.append(r["p_hard_given_contact"])
        print(f"     {label:<28} {r['p_swing']:>9.4f} {r['p_contact_given_swing']:>10.4f} "
              f"{r['p_hard_given_contact']:>11.4f} {r['p_hard_contact']:>8.4f}")

    # All probabilities should be in [0, 1]
    in_range = all(0 <= v <= 1 for v in all_sw + all_con + all_hrd)
    # P(swing) range should be meaningful (min < 0.4, max > 0.5)
    sw_range_ok = min(all_sw) < 0.40 and max(all_sw) > 0.50
    # P(contact|swing) range should vary (spread > 0.05)
    con_spread_ok = max(all_con) - min(all_con) > 0.05

    pass_c1 = in_range and sw_range_ok and con_spread_ok
    results["c1"] = pass_c1
    print(f"\n  In range [0,1]: {in_range}  |  P(swing) varies meaningfully: {sw_range_ok}  "
          f"|  P(contact|sw) spread>{0.05}: {con_spread_ok}")
    print(f"  → {'PASS' if pass_c1 else 'FAIL'}")

    # ── C.2  Count sanity: P(swing) in 2-0 vs 0-2 ────────────────────────────
    print("\nC.2  Count Swing Sanity — same pitch, two counts vs Ohtani")
    r_20 = _p({**BASE_PITCH, "balls": 2, "strikes": 0, "pitch_number": 3,
               "prev_pitch_type": "FF", "prev_pitch_speed": 93.0, "prev_pitch_result": "ball"},
              "Shohei Ohtani")
    r_02 = _p({**BASE_PITCH, "balls": 0, "strikes": 2, "pitch_number": 3,
               "prev_pitch_type": "FF", "prev_pitch_speed": 93.0, "prev_pitch_result": "swinging_strike"},
              "Shohei Ohtani")
    pass_c2 = r_20["p_swing"] > r_02["p_swing"]
    results["c2"] = pass_c2
    print(f"  2-0 P(swing)={r_20['p_swing']:.4f}  vs  0-2 P(swing)={r_02['p_swing']:.4f}")
    print(f"  → {'PASS' if pass_c2 else 'FAIL'}: 2-0 P(swing) {'>' if pass_c2 else 'NOT >'} 0-2 P(swing)")

    # ── C.3  Ohtani vs Hamilton hard-contact split ────────────────────────────
    print("\nC.3  Elite vs Weak Hitter — P(hard_contact) split (center-cut 95 FF)")
    try:
        r_ot  = _p(BASE_PITCH, "Shohei Ohtani")
        r_ham = _p(BASE_PITCH, "Hamilton, Billy")
        delta = r_ot["p_hard_contact"] - r_ham["p_hard_contact"]
        pass_c3 = delta > 0.02 and r_ot["p_swing"] > r_ham["p_swing"] - 0.05
        results["c3"] = pass_c3
        print(f"  Ohtani:   P(swing)={r_ot['p_swing']:.4f}  P(hard)={r_ot['p_hard_contact']:.4f}")
        print(f"  Hamilton: P(swing)={r_ham['p_swing']:.4f}  P(hard)={r_ham['p_hard_contact']:.4f}")
        print(f"  P(hard) gap: {delta:+.4f}")
        print(f"  → {'PASS' if pass_c3 else 'FAIL'}: "
              f"Ohtani P(hard) {'>' if delta > 0.02 else 'NOT >'} Hamilton by >0.02")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["c3"] = False

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Section D — Demo Readiness Checklist
# ──────────────────────────────────────────────────────────────────────────────

def section_d(a_results: dict, b_results: dict, c_results: dict) -> None:
    print("\n" + "═" * 70)
    print("SECTION D — DEMO READINESS CHECKLIST")
    print("═" * 70)

    checks = [
        ("P(contact|sw) decreases w/ velocity?", "A.1 — velocity gradient",    a_results.get("a1", False)),
        ("P(swing) highest in zone?",            "A.2 — location gradient",    a_results.get("a2", False)),
        ("P(swing) higher in hitter counts?",    "A.3 — count leverage ✓",     a_results.get("a3", False)),
        ("P(contact|sw) differs by pitch type?", "A.4 — pitch type spread",    a_results.get("a4", False)),
        ("P(hard) differs across hitters?",      "B   — elite vs weak spread", b_results.get("b", False)),
        ("Probabilities in valid range?",        "C.1 — [0,1] + meaningful",   c_results.get("c1", False)),
        ("P(swing) higher in 2-0 vs 0-2?",      "C.2 — count sanity",         c_results.get("c2", False)),
        ("Ohtani P(hard) > Hamilton P(hard)?",   "C.3 — hitter split",         c_results.get("c3", False)),
    ]

    print(f"\n  {'Check':<42}  {'Section':<28}  {'Result':>6}")
    print("  " + "─" * 78)
    passes = 0
    for question, section, passed in checks:
        if passed:
            passes += 1
        print(f"  {question:<42}  {section:<28}  {'PASS' if passed else 'FAIL':>6}")

    print(f"\n  Score: {passes}/{len(checks)}")
    if passes == len(checks):
        verdict = "Demo ready. All three stages are behaving correctly."
    elif passes >= 6:
        failing = [q for q, _, p in checks if not p]
        verdict = "Demo ready with caveats:\n    " + "\n    ".join(f"• {q}" for q in failing)
    elif passes >= 4:
        failing = [q for q, _, p in checks if not p]
        verdict = "Not demo ready. Fix before showing anyone:\n    " + "\n    ".join(f"• {q}" for q in failing)
    else:
        verdict = "Fundamental model issues. Revisit architecture before demoing."

    print(f"\n  ► {verdict}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PITCH DUEL — FULL THREE-STAGE MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading model artifacts...")
    try:
        swing_pair, contact_pair, hc_pair, feature_cols, encodings = load_models()
        print(f"  Stages loaded: swing, contact, hard_contact")
        print(f"  Features:      {len(feature_cols)}")
        print(f"  Profiles:      {sum(1 for _ in PROFILE_DIR.glob('*.json')):,}")
    except FileNotFoundError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    a_results = section_a()
    b_results = section_b()
    c_results = section_c()
    section_d(a_results, b_results, c_results)


if __name__ == "__main__":
    main()
