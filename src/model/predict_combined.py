"""
predict_combined.py — Blended pitch outcome prediction.

Runs two paths in parallel and blends them:
  Path A: Three-stage XGBoost model (predict.py)
  Path B: Historical similarity lookup (similarity.py)

The blend weight alpha comes from the similarity confidence score:
  alpha = 0 → pure model (no historical evidence)
  alpha = 1 → pure history (perfect evidence, never happens in practice)

For each stage independently:
  blended = alpha * historical + (1 - alpha) * model

Usage:
    from src.model.predict_combined import predict_matchup

    result = predict_matchup(
        pitch={'release_speed': 92.1, 'pitch_type': 'FF', ...},
        hitter_name="Shohei Ohtani",
    )
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_DIR = Path("models")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchupResult:
    # Final blended predictions
    p_swing: float
    p_contact: float           # P(contact | swing)
    p_hard_contact: float      # P(hard_contact | contact)
    p_hard: float              # composite: swing x contact x hard

    # Path A: model predictions
    model_p_swing: float
    model_p_contact: float
    model_p_hard_contact: float
    model_p_hard: float

    # Path B: historical empirical rates
    historical_swing_rate: float
    historical_contact_rate: float
    historical_hard_hit_rate: float
    historical_p_hard: float

    # Blend metadata
    alpha: float               # blend weight (0=all model, 1=all history)
    n_similar_pitches: int
    confidence: float

    # Context
    hitter: str
    pitch_type: str
    count: str

    # Evidence (optional, for display)
    top_similar_pitches: list = field(default_factory=list)
    outcome_distribution: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Blend logic
# ---------------------------------------------------------------------------

def _blend(model_val: float, hist_val: float, alpha: float) -> float:
    return alpha * hist_val + (1.0 - alpha) * model_val


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_matchup(
    pitch: dict,
    hitter_name: str,
    model_dir: Path = MODEL_DIR,
    show_evidence: bool = True,
) -> MatchupResult:
    """
    Run both model prediction and historical similarity lookup,
    then blend into a single MatchupResult.

    If the hitter has no Statcast history (KBO/international) or
    the similarity search finds 0 matches, alpha = 0 (pure model).
    """
    from src.model.predict import predict_pitch
    from src.hitters.similarity import find_similar_pitches

    # -- Path A: model prediction --
    model_result = predict_pitch(pitch, hitter_name, model_dir)
    m_swing   = model_result["p_swing"]
    m_contact = model_result["p_contact_given_swing"]
    m_hard    = model_result["p_hard_given_contact"]
    m_p_hard  = model_result["p_hard_contact"]

    # -- Path B: historical similarity lookup --
    h_swing = 0.0
    h_contact = 0.0
    h_hard = 0.0
    h_p_hard = 0.0
    alpha = 0.0
    n_similar = 0
    confidence = 0.0
    top_pitches = []
    outcome_dist = {}

    try:
        sim_result = find_similar_pitches(
            pitch=pitch,
            hitter_name=hitter_name,
            n_matches=50,
            match_count=True,
        )
        n_similar  = sim_result.n_matches
        confidence = sim_result.confidence

        if n_similar > 0:
            alpha     = confidence
            h_swing   = sim_result.empirical_swing_rate
            h_contact = sim_result.empirical_contact_rate
            h_hard    = sim_result.empirical_hard_hit_rate
            h_p_hard  = h_swing * h_contact * h_hard
            outcome_dist = sim_result.outcome_counts

            if show_evidence:
                for _, row in sim_result.matched_pitches.head(5).iterrows():
                    date = str(row.get("game_date", ""))[:10]
                    velo = float(row.get("release_speed", 0))
                    pt   = str(row.get("pitch_type", "?"))
                    px   = float(row.get("plate_x", 0))
                    pz   = float(row.get("plate_z", 0))
                    desc = str(row.get("description", "?"))
                    ev   = str(row.get("events", "") or "").strip()
                    ls   = row.get("launch_speed")
                    sim  = float(row.get("similarity_score", 0))
                    top_pitches.append({
                        "date": date, "velo": velo, "pitch_type": pt,
                        "plate_x": px, "plate_z": pz,
                        "description": desc, "events": ev,
                        "launch_speed": float(ls) if ls and not np.isnan(ls) else None,
                        "similarity": sim,
                    })

    except (ValueError, FileNotFoundError) as e:
        # No Statcast history for this hitter — pure model prediction
        warnings.warn(
            f"Similarity lookup failed for '{hitter_name}': {e}. Using model only.",
            stacklevel=2,
        )

    # -- Blend per stage (for interpretable display of each stage) --
    b_swing   = _blend(m_swing, h_swing, alpha)
    b_contact = _blend(m_contact, h_contact, alpha)
    b_hard    = _blend(m_hard, h_hard, alpha)

    # Canonical p_hard uses blend-of-composites, not product-of-blends.
    # Product-of-blends (b_swing * b_contact * b_hard) underestimates because
    # the three stages are correlated within each path. Blend-of-composites is
    # the correct weighted expectation: E[hard] = alpha * hist + (1-alpha) * model.
    b_p_hard = _blend(m_p_hard, h_p_hard, alpha)

    count_str = f"{pitch.get('balls', 0)}-{pitch.get('strikes', 0)}"

    return MatchupResult(
        p_swing=round(b_swing, 4),
        p_contact=round(b_contact, 4),
        p_hard_contact=round(b_hard, 4),
        p_hard=round(b_p_hard, 4),

        model_p_swing=m_swing,
        model_p_contact=m_contact,
        model_p_hard_contact=m_hard,
        model_p_hard=m_p_hard,

        historical_swing_rate=h_swing,
        historical_contact_rate=h_contact,
        historical_hard_hit_rate=h_hard,
        historical_p_hard=round(h_p_hard, 4),

        alpha=round(alpha, 4),
        n_similar_pitches=n_similar,
        confidence=round(confidence, 4),

        hitter=hitter_name,
        pitch_type=pitch.get("pitch_type", "?"),
        count=count_str,

        top_similar_pitches=top_pitches,
        outcome_distribution=outcome_dist,
    )


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _print_matchup(r: MatchupResult, velo: float) -> None:
    print()
    print(f"{'='*60}")
    print(f"  Matchup: {velo:.1f} mph {r.pitch_type} vs {r.hitter} ({r.count})")
    print(f"{'='*60}")

    n = r.n_similar_pitches
    n_str = f"{n} matches" if n > 0 else "no matches"

    def _row(label, blended, model, hist):
        print(
            f"  {label:<20} {blended:.2f}  "
            f"(model: {model:.2f} | history: {hist:.2f} | {n_str})"
        )

    _row("P(swing)",         r.p_swing,        r.model_p_swing,        r.historical_swing_rate)
    _row("P(contact|swing)", r.p_contact,       r.model_p_contact,      r.historical_contact_rate)
    _row("P(hard|contact)",  r.p_hard_contact,  r.model_p_hard_contact, r.historical_hard_hit_rate)
    print(f"  {'─'*56}")
    print(
        f"  {'P(hard contact)':<20} {r.p_hard:.2f}  "
        f"(blend weight alpha={r.alpha:.2f})"
    )
    print()

    if r.top_similar_pitches:
        print(f"  Top similar pitches {r.hitter} has faced:")
        for p in r.top_similar_pitches:
            ev_str = ""
            if p["launch_speed"] is not None:
                ev_str = f" ({p['launch_speed']:.1f} mph EV)"
            outcome = p["events"] if p["events"] else p["description"]
            print(
                f"    {p['date']}  {p['velo']:.1f} mph {p['pitch_type']}  "
                f"plate({p['plate_x']:.2f}, {p['plate_z']:.2f})  "
                f"-> {outcome}{ev_str}  sim={p['similarity']:.3f}"
            )
        print()

    if r.outcome_distribution:
        total = sum(r.outcome_distribution.values())
        print("  Outcome distribution from similar pitches:")
        for outcome, cnt in sorted(r.outcome_distribution.items(), key=lambda x: -x[1])[:8]:
            pct = cnt / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {outcome:<32} {cnt:>4}  {pct:>5.1f}%  {bar}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Blended pitch prediction: model + historical similarity"
    )
    parser.add_argument("--hitter",     required=True, type=str)
    parser.add_argument("--pitch-type", required=True, type=str, dest="pitch_type")
    parser.add_argument("--velo",       required=True, type=float)
    parser.add_argument("--plate-x",    type=float, default=0.0,  dest="plate_x")
    parser.add_argument("--plate-z",    type=float, default=2.5,  dest="plate_z")
    parser.add_argument("--pfx-x",      type=float, default=None, dest="pfx_x")
    parser.add_argument("--pfx-z",      type=float, default=None, dest="pfx_z")
    parser.add_argument("--spin",       type=float, default=None)
    parser.add_argument("--ext",        type=float, default=None,
                        help="Release extension (feet)")
    parser.add_argument("--count",      type=str,   default="0-0",
                        help="Ball-strike count, e.g. '1-2'")
    parser.add_argument("--p-throws",   type=str,   default="R", dest="p_throws",
                        help="Pitcher handedness: R or L")
    args = parser.parse_args()

    balls, strikes = (int(x) for x in args.count.split("-"))

    pitch = {
        "pitch_type":    args.pitch_type,
        "release_speed": args.velo,
        "plate_x":       args.plate_x,
        "plate_z":       args.plate_z,
        "balls":         balls,
        "strikes":       strikes,
        "pitch_number":  1,
        "prev_pitch_type":   "FIRST_PITCH",
        "prev_pitch_speed":  0.0,
        "prev_pitch_result": "FIRST_PITCH",
        "on_1b": None, "on_2b": None, "on_3b": None,
        "inning": 1, "score_diff": 0,
        "p_throws": args.p_throws,
        "release_spin_rate": args.spin or 0,
    }
    if args.pfx_x is not None:
        pitch["pfx_x"] = args.pfx_x
    if args.pfx_z is not None:
        pitch["pfx_z"] = args.pfx_z
    if args.ext is not None:
        pitch["release_extension"] = args.ext

    result = predict_matchup(pitch=pitch, hitter_name=args.hitter)
    _print_matchup(result, args.velo)
