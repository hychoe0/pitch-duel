"""
at_bat.py — Narrative at-bat simulator for Pitch Duel.

Simulates a full plate appearance pitch by pitch, filling in contextual
features automatically (count, pitch number, prev pitch tracking) and
printing a coach-readable interpretation after each pitch.

Usage:
    python -m src.demo.at_bat
"""

import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.predict import predict_pitch, interpret_pitch
from src.utils.pitch_helpers import PITCH_DEFAULTS, location_tag


# ──────────────────────────────────────────────────────────────────────────────
# Strike zone check
# ──────────────────────────────────────────────────────────────────────────────

_SZ_X = 0.85   # half-width of strike zone in feet
_SZ_Z_LO = 1.5
_SZ_Z_HI = 3.5


def _is_in_zone(plate_x: float, plate_z: float) -> bool:
    return abs(plate_x) < _SZ_X and _SZ_Z_LO < plate_z < _SZ_Z_HI


# ──────────────────────────────────────────────────────────────────────────────
# Outcome simulation (stochastic, with count advancement)
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_outcome(
    p_swing: float,
    p_contact_given_swing: float,
    p_hard_given_contact: float,
    plate_x: float,
    plate_z: float,
    balls: int,
    strikes: int,
) -> tuple:
    """
    Stochastically simulate a pitch outcome and advance the count.
    Returns (result_label, new_balls, new_strikes, at_bat_over).
    """
    if random.random() >= p_swing:
        # Hitter takes — called strike or ball depends on location
        if _is_in_zone(plate_x, plate_z):
            s = strikes + 1
            return "called_strike", balls, s, s >= 3
        else:
            b = balls + 1
            return "ball", b, strikes, b >= 4

    # Hitter swings
    if random.random() >= p_contact_given_swing:
        # Whiff
        s = strikes + 1
        return "swinging_strike", balls, s, s >= 3

    # Contact made — hard or soft?
    if random.random() < p_hard_given_contact:
        return "hit_into_play", balls, strikes, True

    # Soft contact → foul (cannot advance past 2 strikes)
    s = min(strikes + 1, 2)
    return "foul", balls, s, False


# ──────────────────────────────────────────────────────────────────────────────
# Core simulator
# ──────────────────────────────────────────────────────────────────────────────


def simulate_at_bat(
    pitcher_arsenal: list,
    hitter_name: str,
    p_throws: str = "R",
    on_1b=None,
    on_2b=None,
    on_3b=None,
    inning: int = 1,
    score_diff: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Simulate a full plate appearance and return a summary dict.

    Args:
        pitcher_arsenal: ordered list of pitch dicts, each containing at minimum:
            release_speed, pitch_type, plate_x, plate_z
            Optional: release_spin_rate, pfx_x, pfx_z, release_pos_x/z,
                      release_extension (defaults filled from PITCH_DEFAULTS)
        hitter_name:  player name string (e.g. "Shohei Ohtani")
        p_throws:     pitcher handedness "R" or "L"
        on_1b/2b/3b:  baserunner flags (None = empty)
        inning:       current inning
        score_diff:   runs: positive = batting team ahead
        verbose:      print pitch-by-pitch output

    Returns:
        {
          'pitches': list of per-pitch result dicts,
          'avg_p_hard_contact': float,
          'best_pitch': dict,   # lowest p_hard_contact
          'worst_pitch': dict,  # highest p_hard_contact
          'verdict': str,
          'hitter': str,
          'final_count': str,
        }
    """
    balls, strikes = 0, 0
    prev_pitch_type   = "FIRST_PITCH"
    prev_pitch_speed  = 0.0
    prev_pitch_result = "FIRST_PITCH"

    pitch_results = []

    if verbose:
        print(f"  Hitter: {hitter_name}")
        print(f"  {'#':<4} {'Count':<6} {'Type':<5} {'MPH':>5}  {'Location':<18} "
              f"{'P(sw)':>6} {'P(con)':>6} {'P(hrd)':>6}  Quality")
        print(f"  {'─'*4} {'─'*6} {'─'*5} {'─'*5}  {'─'*18} "
              f"{'─'*6} {'─'*6} {'─'*6}  {'─'*35}")

    for i, raw_pitch in enumerate(pitcher_arsenal):
        pitch_num = i + 1

        pitch = {**PITCH_DEFAULTS, **raw_pitch}
        pitch.update({
            "balls":             balls,
            "strikes":           strikes,
            "pitch_number":      pitch_num,
            "prev_pitch_type":   prev_pitch_type,
            "prev_pitch_speed":  prev_pitch_speed,
            "prev_pitch_result": prev_pitch_result,
            "on_1b":             on_1b,
            "on_2b":             on_2b,
            "on_3b":             on_3b,
            "inning":            inning,
            "score_diff":        score_diff,
            "p_throws":          p_throws,
            "game_date":         date.today().isoformat(),
        })

        result = predict_pitch(pitch, hitter_name)
        p_swing            = result["p_swing"]
        p_contact_given_sw = result["p_contact_given_swing"]
        p_hard_given_con   = result["p_hard_given_contact"]
        p_hard_contact     = result["p_hard_contact"]

        px = raw_pitch.get("plate_x", pitch["plate_x"])
        pz = raw_pitch.get("plate_z", pitch["plate_z"])
        loc_tag   = location_tag(px, pz)
        quality   = result["pitch_quality"]
        count_str = f"{balls}-{strikes}"

        if verbose:
            print(
                f"  {pitch_num:<4} {count_str:<6} "
                f"{pitch['pitch_type']:<5} {pitch['release_speed']:>5.1f}  "
                f"{loc_tag:<18} "
                f"{p_swing:>6.3f} {p_contact_given_sw:>6.3f} {p_hard_given_con:>6.3f}  "
                f"{quality}"
            )

        outcome, balls, strikes, ab_over = _simulate_outcome(
            p_swing, p_contact_given_sw, p_hard_given_con, px, pz, balls, strikes
        )

        prev_pitch_type   = pitch["pitch_type"]
        prev_pitch_speed  = pitch["release_speed"]
        prev_pitch_result = outcome

        pitch_results.append({
            "pitch_num":             pitch_num,
            "count_before":          count_str,
            "pitch_type":            pitch["pitch_type"],
            "speed":                 pitch["release_speed"],
            "plate_x":               px,
            "plate_z":               pz,
            "location":              loc_tag,
            "p_swing":               p_swing,
            "p_contact_given_swing": p_contact_given_sw,
            "p_hard_given_contact":  p_hard_given_con,
            "p_contact":             result["p_contact"],
            "p_hard_contact":        p_hard_contact,
            "outcome":               outcome,
            "pitch_quality":         quality,
            # legacy key for backward compat with server.py at-bat endpoint
            "xwoba":                 p_hard_contact,
            "interpretation":        quality,
        })

        if ab_over:
            if verbose:
                if strikes == 3:
                    print(f"  {'':4} {'':6} → Strikeout")
                elif balls == 4:
                    print(f"  {'':4} {'':6} → Walk")
                else:
                    print(f"  {'':4} {'':6} → Ball in play")
            break

    avg_hard = sum(p["p_hard_contact"] for p in pitch_results) / len(pitch_results)
    best_pitch  = min(pitch_results, key=lambda p: p["p_hard_contact"])
    worst_pitch = max(pitch_results, key=lambda p: p["p_hard_contact"])

    verdict = (
        f"This sequence kept {hitter_name} off balance"
        if avg_hard < 0.12
        else f"This sequence gave {hitter_name} too many hard-contact looks"
    )

    final_count = f"{balls}-{strikes}"

    if verbose:
        print(f"\n  Avg P(hard_contact): {avg_hard:.3f}")
        print(f"  Best pitch:  #{best_pitch['pitch_num']} "
              f"{best_pitch['pitch_type']} {best_pitch['speed']:.0f} mph "
              f"{best_pitch['location']} → P(hard)={best_pitch['p_hard_contact']:.3f}")
        print(f"  Worst pitch: #{worst_pitch['pitch_num']} "
              f"{worst_pitch['pitch_type']} {worst_pitch['speed']:.0f} mph "
              f"{worst_pitch['location']} → P(hard)={worst_pitch['p_hard_contact']:.3f}")
        print(f"  Verdict: {verdict}")

    return {
        "pitches":           pitch_results,
        "avg_p_hard_contact": round(avg_hard, 4),
        # legacy keys for server.py at-bat endpoint
        "avg_xwoba":         round(avg_hard, 4),
        "best_pitch":        best_pitch,
        "worst_pitch":       worst_pitch,
        "verdict":           verdict,
        "hitter":            hitter_name,
        "final_count":       final_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# __main__ demo
# ──────────────────────────────────────────────────────────────────────────────

_ARSENAL = [
    {"pitch_type": "FF", "release_speed": 96.0,
     "plate_x": -0.4, "plate_z": 3.3,
     "pfx_x": -0.5, "pfx_z": 1.2, "release_spin_rate": 2400,
     "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3},
    {"pitch_type": "SL", "release_speed": 87.0,
     "plate_x": 0.7, "plate_z": 1.8,
     "pfx_x": 0.8, "pfx_z": 0.3, "release_spin_rate": 2600,
     "release_pos_x": -1.6, "release_pos_z": 5.9, "release_extension": 6.2},
    {"pitch_type": "FF", "release_speed": 97.0,
     "plate_x": -0.5, "plate_z": 3.4,
     "pfx_x": -0.5, "pfx_z": 1.2, "release_spin_rate": 2400,
     "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3},
    {"pitch_type": "SL", "release_speed": 86.0,
     "plate_x": -0.8, "plate_z": 2.0,
     "pfx_x": 0.8, "pfx_z": 0.3, "release_spin_rate": 2600,
     "release_pos_x": -1.6, "release_pos_z": 5.9, "release_extension": 6.2},
    {"pitch_type": "FF", "release_speed": 97.0,
     "plate_x": 0.1, "plate_z": 3.6,
     "pfx_x": -0.5, "pfx_z": 1.2, "release_spin_rate": 2400,
     "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3},
]

if __name__ == "__main__":
    random.seed(42)
    SEP = "═" * 70

    print(SEP)
    print("PITCH DUEL — Narrative At-Bat Simulator (three-stage)")
    print(SEP)

    print("\n[AT-BAT 1]  Power FB/SL combo vs Shohei Ohtani")
    print("─" * 70)
    ab1 = simulate_at_bat(_ARSENAL, "Shohei Ohtani")

    print(f"\n[AT-BAT 2]  Same arsenal vs Billy Hamilton (weakest by hard-hit rate)")
    print("─" * 70)
    ab2 = simulate_at_bat(_ARSENAL, "Hamilton, Billy")

    print(f"\n{SEP}")
    print("COMPARISON — Same Pitching, Different Hitter Caliber")
    print(SEP)
    print(f"  {'Metric':<35} {'vs Ohtani':>10}  {'vs Hamilton':>12}")
    print(f"  {'─'*35} {'─'*10}  {'─'*12}")
    print(f"  {'Avg P(hard_contact)':<35} {ab1['avg_p_hard_contact']:>10.3f}  "
          f"{ab2['avg_p_hard_contact']:>12.3f}")

    for i, (p1, p2) in enumerate(zip(ab1["pitches"], ab2["pitches"]), 1):
        print(f"  Pitch {i} P(hard) ({p1['pitch_type']} {p1['speed']:.0f}mph){'':<10} "
              f"{p1['p_hard_contact']:>10.3f}  {p2['p_hard_contact']:>12.3f}")

    spread = ab1["avg_p_hard_contact"] - ab2["avg_p_hard_contact"]
    print(f"\n  P(hard_contact) gap (Ohtani - Hamilton): {spread:+.3f}")
    print(f"  Ohtani verdict:   {ab1['verdict']}")
    print(f"  Hamilton verdict: {ab2['verdict']}")
    print()
