"""
at_bat.py — Narrative at-bat simulator for Pitch Duel.

Simulates a full plate appearance pitch by pitch, filling in contextual
features automatically (count, pitch number, prev pitch tracking) and
printing a coach-readable interpretation after each pitch.

Usage:
    python -m src.demo.at_bat
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.predict import predict_hit_probability

# ──────────────────────────────────────────────────────────────────────────────
# Interpretation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _interpret(xwoba: float) -> str:
    if xwoba < 0.04:
        return "Swing and miss territory"
    if xwoba < 0.08:
        return "Good pitch — hitter is uncomfortable"
    if xwoba < 0.15:
        return "Competitive — could go either way"
    if xwoba < 0.25:
        return "Hitter is sitting on this — danger zone"
    return "Mistake pitch — hitter punishes this"


def _location_tag(plate_x: float, plate_z: float) -> str:
    """Human-readable zone label from catcher's perspective."""
    # Vertical
    if plate_z > 3.2:
        vert = "elevated"
    elif plate_z > 2.6:
        vert = "mid"
    elif plate_z > 1.8:
        vert = "low"
    else:
        vert = "below zone"

    # Horizontal (from catcher's view: positive plate_x = arm side = right side of plate)
    if plate_x > 0.85:
        horiz = "off plate away"
    elif plate_x > 0.4:
        horiz = "away"
    elif plate_x > -0.4:
        horiz = "middle"
    elif plate_x > -0.85:
        horiz = "in"
    else:
        horiz = "off plate in"

    if horiz == "middle":
        return vert
    return f"{vert}-{horiz}"


# ──────────────────────────────────────────────────────────────────────────────
# Outcome simulation (rough — for count progression only)
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_outcome(xwoba: float, balls: int, strikes: int) -> tuple[str, int, int, bool]:
    """
    Return (result_label, new_balls, new_strikes, at_bat_over).
    Thresholds differ by whether the hitter is already at 2 strikes.
    """
    if strikes < 2:
        if xwoba < 0.05:
            return "swinging_strike", balls, strikes + 1, False
        elif xwoba < 0.10:
            return "called_strike", balls, strikes + 1, False
        elif xwoba < 0.18:
            return "ball", balls + 1, strikes, balls + 1 == 4  # walk if 4th ball
        else:
            return "foul", balls, strikes + 1, False
    else:  # strikes == 2
        if xwoba < 0.05:
            return "swinging_strike", balls, 3, True   # strikeout
        elif xwoba < 0.10:
            return "foul", balls, 2, False             # foul, count holds
        elif xwoba < 0.18:
            return "ball", balls + 1, 2, balls + 1 == 4  # walk if 4th ball
        else:
            return "hit_into_play", balls, 2, True     # contact, end at-bat


# ──────────────────────────────────────────────────────────────────────────────
# Core simulator
# ──────────────────────────────────────────────────────────────────────────────

_PITCH_DEFAULTS = {
    "release_spin_rate": 2400.0,
    "pfx_x":             -0.5,
    "pfx_z":              1.2,
    "release_pos_x":     -1.5,
    "release_pos_z":      6.0,
    "release_extension":  6.3,
}


def simulate_at_bat(
    pitcher_arsenal: list[dict],
    hitter_name: str,
    pitcher_id: int = None,
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
        pitcher_id:   MLBAM pitcher ID for feature lookup; None → medians
        p_throws:     pitcher handedness "R" or "L"
        on_1b/2b/3b:  baserunner flags (None = empty)
        inning:       current inning
        score_diff:   runs: positive = batting team ahead
        verbose:      print pitch-by-pitch output

    Returns:
        {
          'pitches': list of per-pitch result dicts,
          'avg_xwoba': float,
          'best_pitch': dict,   # lowest xwOBA
          'worst_pitch': dict,  # highest xwOBA
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
        print(f"  {'#':<4} {'Count':<6} {'Type':<5} {'MPH':>5}  {'Location':<18} {'xwOBA':>7}  Interpretation")
        print(f"  {'─'*4} {'─'*6} {'─'*5} {'─'*5}  {'─'*18} {'─'*7}  {'─'*35}")

    for i, raw_pitch in enumerate(pitcher_arsenal):
        pitch_num = i + 1

        # Fill defaults for missing mechanical features
        pitch = {**_PITCH_DEFAULTS, **raw_pitch}
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

        result = predict_hit_probability(pitch, hitter_name, pitcher_id=pitcher_id)
        xw = result["xwoba_prediction"]

        px = raw_pitch.get("plate_x", pitch["plate_x"])
        pz = raw_pitch.get("plate_z", pitch["plate_z"])
        loc_tag  = _location_tag(px, pz)
        interp   = _interpret(xw)
        count_str = f"{balls}-{strikes}"

        if verbose:
            print(
                f"  {pitch_num:<4} {count_str:<6} "
                f"{pitch['pitch_type']:<5} {pitch['release_speed']:>5.1f}  "
                f"{loc_tag:<18} {xw:>7.3f}  {interp}"
            )

        # Determine outcome and advance count
        outcome, balls, strikes, ab_over = _simulate_outcome(xw, balls, strikes)

        prev_pitch_type   = pitch["pitch_type"]
        prev_pitch_speed  = pitch["release_speed"]
        prev_pitch_result = outcome

        pitch_results.append({
            "pitch_num":    pitch_num,
            "count_before": count_str,
            "pitch_type":   pitch["pitch_type"],
            "speed":        pitch["release_speed"],
            "plate_x":      px,
            "plate_z":      pz,
            "location":     loc_tag,
            "xwoba":        xw,
            "outcome":      outcome,
            "interpretation": interp,
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

    avg_xw     = sum(p["xwoba"] for p in pitch_results) / len(pitch_results)
    best_pitch = min(pitch_results, key=lambda p: p["xwoba"])
    worst_pitch = max(pitch_results, key=lambda p: p["xwoba"])

    if avg_xw < 0.10:
        verdict = f"This sequence kept {hitter_name} off balance"
    else:
        verdict = f"This sequence gave {hitter_name} too many good looks"

    final_count = f"{balls}-{strikes}"

    if verbose:
        print(f"\n  Average xwOBA: {avg_xw:.3f}")
        print(f"  Best pitch:  #{best_pitch['pitch_num']} "
              f"{best_pitch['pitch_type']} {best_pitch['speed']:.0f} mph "
              f"{best_pitch['location']} → {best_pitch['xwoba']:.3f}")
        print(f"  Worst pitch: #{worst_pitch['pitch_num']} "
              f"{worst_pitch['pitch_type']} {worst_pitch['speed']:.0f} mph "
              f"{worst_pitch['location']} → {worst_pitch['xwoba']:.3f}")
        print(f"  Verdict: {verdict}")

    return {
        "pitches":      pitch_results,
        "avg_xwoba":    round(avg_xw, 4),
        "best_pitch":   best_pitch,
        "worst_pitch":  worst_pitch,
        "verdict":      verdict,
        "hitter":       hitter_name,
        "final_count":  final_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# __main__ demo
# ──────────────────────────────────────────────────────────────────────────────

# Hardcoded 5-pitch sequence: power FB-slider combo
_ARSENAL = [
    # FF up and in
    {"pitch_type": "FF", "release_speed": 96.0,
     "plate_x": -0.4, "plate_z": 3.3,
     "pfx_x": -0.5, "pfx_z": 1.2, "release_spin_rate": 2400,
     "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3},
    # SL low and away
    {"pitch_type": "SL", "release_speed": 87.0,
     "plate_x": 0.7, "plate_z": 1.8,
     "pfx_x": 0.8, "pfx_z": 0.3, "release_spin_rate": 2600,
     "release_pos_x": -1.6, "release_pos_z": 5.9, "release_extension": 6.2},
    # FF up and in again (tunnel off first pitch)
    {"pitch_type": "FF", "release_speed": 97.0,
     "plate_x": -0.5, "plate_z": 3.4,
     "pfx_x": -0.5, "pfx_z": 1.2, "release_spin_rate": 2400,
     "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3},
    # SL back-door (sweeps toward the opposite corner)
    {"pitch_type": "SL", "release_speed": 86.0,
     "plate_x": -0.8, "plate_z": 2.0,
     "pfx_x": 0.8, "pfx_z": 0.3, "release_spin_rate": 2600,
     "release_pos_x": -1.6, "release_pos_z": 5.9, "release_extension": 6.2},
    # FF elevated — chase pitch
    {"pitch_type": "FF", "release_speed": 97.0,
     "plate_x": 0.1, "plate_z": 3.6,
     "pfx_x": -0.5, "pfx_z": 1.2, "release_spin_rate": 2400,
     "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3},
]

if __name__ == "__main__":
    SEP = "═" * 64

    print(SEP)
    print("PITCH DUEL — Narrative At-Bat Simulator")
    print(SEP)

    # ── At-bat 1: vs Ohtani ──────────────────────────────────────────────────
    print("\n[AT-BAT 1]  Power FB/SL combo vs Shohei Ohtani")
    print("─" * 64)
    ab1 = simulate_at_bat(_ARSENAL, "Shohei Ohtani")

    # ── At-bat 2: vs Billy Hamilton (weakest hitter) ─────────────────────────
    print(f"\n[AT-BAT 2]  Same arsenal vs Billy Hamilton (weakest by hard-hit rate)")
    print("─" * 64)
    ab2 = simulate_at_bat(_ARSENAL, "Hamilton, Billy")

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("COMPARISON — Same Pitching, Different Hitter Caliber")
    print(SEP)
    print(f"  {'Metric':<30} {'vs Ohtani':>10}  {'vs Hamilton':>12}")
    print(f"  {'─'*30} {'─'*10}  {'─'*12}")
    print(f"  {'Avg xwOBA':<30} {ab1['avg_xwoba']:>10.3f}  {ab2['avg_xwoba']:>12.3f}")

    for i, (p1, p2) in enumerate(zip(ab1["pitches"], ab2["pitches"]), 1):
        print(f"  Pitch {i} xwOBA ({p1['pitch_type']} {p1['speed']:.0f}mph){'':<8} "
              f"{p1['xwoba']:>10.3f}  {p2['xwoba']:>12.3f}")

    spread = ab1["avg_xwoba"] - ab2["avg_xwoba"]
    print(f"\n  Avg xwOBA gap (Ohtani - Hamilton): {spread:+.3f}")
    print(f"  Ohtani verdict:   {ab1['verdict']}")
    print(f"  Hamilton verdict: {ab2['verdict']}")
    print()
