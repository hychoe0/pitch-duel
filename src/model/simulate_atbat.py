"""
simulate_atbat.py — Pitch-by-pitch at-bat simulation with blended predictions.

Runs a sequence of pitches as a simulated plate appearance, automatically
updating count, pitch_number, and prev_pitch context after each pitch.
Uses predict_combined (model + historical similarity blend) for each pitch.

Two modes:
  Prediction mode: model decides simulated outcome from probabilities
  Replay mode:     user provides actual outcomes, model shows predicted vs actual

Usage:
    python -m src.model.simulate_atbat \\
        --hitter "Shohei Ohtani" \\
        --pitches '[{"pitch_type":"FF","release_speed":95.2,"plate_x":0.1,"plate_z":2.8},...]'
"""

import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from src.utils.pitch_helpers import PITCH_DEFAULTS, location_tag

MODEL_DIR = Path("models")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PitchResult:
    pitch_num: int
    count_before: str           # e.g. "0-0"
    pitch_type: str
    speed: float
    plate_x: float
    plate_z: float
    location: str               # human-readable zone label

    # Blended predictions
    p_swing: float
    p_contact: float            # P(contact | swing)
    p_hard_contact: float       # P(hard | contact)
    p_hard: float               # composite

    # Blend metadata
    alpha: float
    n_similar: int

    # Model-only predictions (for comparison)
    model_p_hard: float

    # Outcome
    outcome: str                # simulated or actual
    danger_level: str           # HIGH / MODERATE / LOW


@dataclass
class AtBatResult:
    hitter: str
    pitches: list               # list of PitchResult
    final_count: str
    avg_p_hard: float
    most_dangerous: PitchResult
    safest: PitchResult
    verdict: str




def _danger_level(p_hard: float) -> str:
    if p_hard >= 0.35:
        return "HIGH DANGER"
    elif p_hard >= 0.15:
        return "MODERATE"
    else:
        return "LOW DANGER"


# ---------------------------------------------------------------------------
# Strike zone check
# ---------------------------------------------------------------------------

_SZ_X = 0.85   # half-width of strike zone in feet
_SZ_Z_LO = 1.5
_SZ_Z_HI = 3.5


def _is_in_zone(plate_x: float, plate_z: float) -> bool:
    return abs(plate_x) < _SZ_X and _SZ_Z_LO < plate_z < _SZ_Z_HI


# ---------------------------------------------------------------------------
# Outcome simulation (stochastic)
# ---------------------------------------------------------------------------

def _simulate_outcome(
    p_swing: float,
    p_contact: float,
    p_hard_contact: float,
    plate_x: float,
    plate_z: float,
) -> str:
    """
    Stochastically simulate a pitch outcome by sampling from model probabilities.

    Branches:
      1. Swing/take — sampled from P(swing)
      2. If take → called_strike (in zone) or ball (out of zone)
      3. If swing → contact/whiff sampled from P(contact|swing)
      4. If contact → hard hit sampled from P(hard|contact)
         hard hit → hit_into_play; soft contact → foul
    """
    if random.random() >= p_swing:
        # Hitter takes — called strike or ball depends on location
        return "called_strike" if _is_in_zone(plate_x, plate_z) else "ball"

    # Hitter swings
    if random.random() >= p_contact:
        return "swinging_strike"

    # Contact made — hard or soft?
    if random.random() < p_hard_contact:
        return "hit_into_play"

    return "foul"


def _advance_count(outcome: str, balls: int, strikes: int) -> tuple:
    """
    Advance balls/strikes from the outcome label.
    Returns (new_balls, new_strikes, at_bat_over).
    """
    if outcome in ("ball", "blocked_ball"):
        b = balls + 1
        return b, strikes, b >= 4
    if outcome in ("called_strike", "swinging_strike", "swinging_strike_blocked"):
        s = strikes + 1
        return balls, s, s >= 3
    if outcome == "foul":
        s = min(strikes + 1, 2)
        return balls, s, False
    if outcome in ("hit_into_play", "hit_into_play_no_out", "hit_into_play_score"):
        return balls, strikes, True
    # Foul tip / foul bunt — same as foul
    if outcome in ("foul_tip", "foul_bunt"):
        s = min(strikes + 1, 2)
        return balls, s, False
    # Default: count unchanged, not over
    return balls, strikes, False


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_at_bat(
    pitches: list,
    hitter_name: str,
    outcomes: list = None,
    p_throws: str = "R",
    on_1b=None, on_2b=None, on_3b=None,
    inning: int = 1,
    score_diff: int = 0,
) -> AtBatResult:
    """
    Simulate a plate appearance pitch by pitch using blended predictions.

    Args:
        pitches:    List of pitch dicts. Required keys per pitch: pitch_type,
                    release_speed, plate_x, plate_z. Optional keys are filled
                    from PITCH_DEFAULTS.
        hitter_name: e.g. "Shohei Ohtani"
        outcomes:   If provided (replay mode), list of outcome strings matching
                    each pitch (e.g. ["called_strike", "ball", "swinging_strike"]).
                    If None (prediction mode), outcomes are simulated from probabilities.
        p_throws:   Pitcher handedness
        on_1b/2b/3b: Baserunner flags
        inning:     Current inning
        score_diff: Runs from batting team's perspective

    Returns:
        AtBatResult with per-pitch breakdowns and summary.
    """
    from src.model.predict_combined import predict_matchup

    replay_mode = outcomes is not None
    if replay_mode and len(outcomes) != len(pitches):
        raise ValueError(
            f"outcomes length ({len(outcomes)}) must match pitches length ({len(pitches)})"
        )

    balls, strikes = 0, 0
    prev_pitch_type   = "FIRST_PITCH"
    prev_pitch_speed  = 0.0
    prev_pitch_result = "FIRST_PITCH"

    pitch_results = []

    for i, raw_pitch in enumerate(pitches):
        pitch_num = i + 1
        count_str = f"{balls}-{strikes}"

        # Build full pitch dict with defaults and context
        pitch = {**PITCH_DEFAULTS, **raw_pitch}
        pitch.update({
            "balls":            balls,
            "strikes":          strikes,
            "pitch_number":     pitch_num,
            "prev_pitch_type":  prev_pitch_type,
            "prev_pitch_speed": prev_pitch_speed,
            "prev_pitch_result": prev_pitch_result,
            "on_1b":            on_1b,
            "on_2b":            on_2b,
            "on_3b":            on_3b,
            "inning":           inning,
            "score_diff":       score_diff,
            "p_throws":         p_throws,
            "game_date":        date.today().isoformat(),
        })

        mr = predict_matchup(pitch, hitter_name, show_evidence=False)

        px = float(pitch["plate_x"])
        pz = float(pitch["plate_z"])
        loc = location_tag(px, pz)
        danger = _danger_level(mr.p_hard)

        # Determine outcome
        if replay_mode:
            outcome = outcomes[i]
        else:
            outcome = _simulate_outcome(
                mr.p_swing, mr.p_contact, mr.p_hard_contact, px, pz,
            )

        pr = PitchResult(
            pitch_num=pitch_num,
            count_before=count_str,
            pitch_type=pitch["pitch_type"],
            speed=pitch["release_speed"],
            plate_x=px,
            plate_z=pz,
            location=loc,
            p_swing=mr.p_swing,
            p_contact=mr.p_contact,
            p_hard_contact=mr.p_hard_contact,
            p_hard=mr.p_hard,
            alpha=mr.alpha,
            n_similar=mr.n_similar_pitches,
            model_p_hard=mr.model_p_hard,
            outcome=outcome,
            danger_level=danger,
        )
        pitch_results.append(pr)

        # Advance context for next pitch
        prev_pitch_type  = pitch["pitch_type"]
        prev_pitch_speed = pitch["release_speed"]
        prev_pitch_result = outcome

        balls, strikes, ab_over = _advance_count(outcome, balls, strikes)
        if ab_over:
            break

    final_count = f"{balls}-{strikes}"
    avg_p_hard = sum(p.p_hard for p in pitch_results) / len(pitch_results)
    most_dangerous = max(pitch_results, key=lambda p: p.p_hard)
    safest = min(pitch_results, key=lambda p: p.p_hard)

    if avg_p_hard < 0.12:
        verdict = f"Sequence kept {hitter_name} off balance — low damage throughout."
    elif avg_p_hard < 0.30:
        verdict = f"Competitive at-bat vs {hitter_name} — mixed danger levels."
    else:
        verdict = f"Sequence gave {hitter_name} too many hard-contact opportunities."

    return AtBatResult(
        hitter=hitter_name,
        pitches=pitch_results,
        final_count=final_count,
        avg_p_hard=round(avg_p_hard, 4),
        most_dangerous=most_dangerous,
        safest=safest,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _print_at_bat(result: AtBatResult) -> None:
    print(f"\n{'='*62}")
    print(f"  At-bat: vs {result.hitter}")
    print(f"{'='*62}")

    for p in result.pitches:
        print(
            f"  Pitch {p.pitch_num}: {p.speed:.1f} mph {p.pitch_type} ({p.count_before})  "
            f"{p.location}"
        )
        print(
            f"    P(swing)={p.p_swing:.2f}  P(hard)={p.p_hard:.2f}  "
            f"alpha={p.alpha:.2f}  [{p.danger_level}]"
        )
        print(
            f"    outcome: {p.outcome}  "
            f"(model P(hard)={p.model_p_hard:.3f}, {p.n_similar} similar pitches)"
        )

    print(f"\n  {'─'*56}")
    print(f"  At-bat summary:")
    print(f"    Avg P(hard):          {result.avg_p_hard:.3f}")
    print(f"    Most dangerous pitch: #{result.most_dangerous.pitch_num} "
          f"({result.most_dangerous.pitch_type} {result.most_dangerous.location} "
          f"at {result.most_dangerous.count_before})")
    print(f"    Safest pitch:         #{result.safest.pitch_num} "
          f"({result.safest.pitch_type} {result.safest.location} "
          f"at {result.safest.count_before})")
    print(f"    Final count:          {result.final_count}")
    print(f"    Verdict:              {result.verdict}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Simulate an at-bat pitch by pitch with blended predictions"
    )
    parser.add_argument("--hitter",   required=True, type=str)
    parser.add_argument("--pitches",  required=True, type=str,
                        help="JSON array of pitch dicts")
    parser.add_argument("--outcomes", type=str, default=None,
                        help="JSON array of outcome strings (replay mode)")
    parser.add_argument("--p-throws", type=str, default="R", dest="p_throws")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed for reproducible simulation")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    pitches = json.loads(args.pitches)
    outcomes = json.loads(args.outcomes) if args.outcomes else None

    result = simulate_at_bat(
        pitches=pitches,
        hitter_name=args.hitter,
        outcomes=outcomes,
        p_throws=args.p_throws,
    )
    _print_at_bat(result)
