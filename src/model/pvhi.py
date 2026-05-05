"""
pvhi.py — Pitch vs. Hitter Index (PVHI).

Derived metric on top of V5 model outputs. Does NOT retrain anything.
Scale: 100 = average danger for this specific hitter.

Components:
  Stuff    (weight 0.15): physics-based run value via KNN similarity
  Location (weight 0.65): zone-specific damage potential
  Count    (weight 0.20): count aggressiveness multiplier

Usage:
    from src.model.pvhi import compute_pvhi
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.hitters.profiles import HitterProfile

PVHI_WEIGHTS = {
    "stuff":    0.15,
    "location": 0.65,
    "count":    0.20,
}

LEAGUE_AVG_STUFF_RUN_VALUE    = 0.075
LEAGUE_AVG_LOCATION_RUN_VALUE = 0.075
LEAGUE_AVG_COUNT_RUN_VALUE    = 0.075

_PVHI_MAX = 250.0
_PVHI_MIN = 0.0


def _clip(v: float) -> float:
    return float(max(_PVHI_MIN, min(_PVHI_MAX, v)))


def compute_pvhi(
    pitch: dict,
    profile: "HitterProfile",
    stuff_features: dict,
    hitter_context_features: dict,
) -> dict:
    """
    Compute Pitch vs. Hitter Index and its three components.

    Args:
        pitch: raw pitch dict (plate_x, plate_z, balls, strikes, pitch_type, …)
        profile: HitterProfile for the batter
        stuff_features: output from compute_stuff_vs_hitter()
        hitter_context_features: output from resolve_hitter_features_for_pitch()

    Returns:
        {
            "pvhi":          float,  # 0–250 scale, 100 = avg for this hitter
            "pvhi_stuff":    float,
            "pvhi_location": float,
            "pvhi_count":    float,
            "denominator":   float,  # hitter's overall_xwoba_per_pitch used
        }
    """
    from src.hitters.profiles import get_overall_xwoba_per_pitch

    denom = get_overall_xwoba_per_pitch(profile)

    # ── Stuff component ────────────────────────────────────────────────────────
    # stuff_vs_hitter_xwoba = mean xwOBA on swings from physics-similar pitches.
    # Multiply by P(swing) to convert swing-conditional to per-pitch run value.
    p_swing_est = (
        hitter_context_features["hitter_swing_rate_this_count"]
        + hitter_context_features["hitter_zone_swing_rate"]
    ) / 2.0
    stuff_run_value = stuff_features["stuff_vs_hitter_xwoba"] * p_swing_est
    pvhi_stuff = _clip(100.0 * stuff_run_value / denom)

    # ── Location component ─────────────────────────────────────────────────────
    # Expected per-pitch run value = P(swing) × P(contact|swing) × xwOBA_on_contact
    # zone_xwoba represents xwOBA on contact in this zone.
    zone_xwoba    = hitter_context_features["hitter_zone_xwoba"]
    p_zone_swing  = hitter_context_features["hitter_zone_swing_rate"]
    p_zone_contact = 1.0 - hitter_context_features["hitter_zone_whiff_rate"]
    location_run_value = p_zone_swing * p_zone_contact * zone_xwoba
    pvhi_location = _clip(100.0 * location_run_value / denom)

    # ── Count component ────────────────────────────────────────────────────────
    # How much more/less aggressive is this hitter in the current count?
    p_swing_this_count = hitter_context_features["hitter_swing_rate_this_count"]
    p_swing_overall    = max(profile.swing_rate, 0.01)
    count_multiplier   = p_swing_this_count / p_swing_overall
    pvhi_count = _clip(100.0 * count_multiplier)

    # ── Composite ──────────────────────────────────────────────────────────────
    pvhi = _clip(
        PVHI_WEIGHTS["stuff"]    * pvhi_stuff
        + PVHI_WEIGHTS["location"] * pvhi_location
        + PVHI_WEIGHTS["count"]    * pvhi_count
    )

    return {
        "pvhi":          round(pvhi, 2),
        "pvhi_stuff":    round(pvhi_stuff, 2),
        "pvhi_location": round(pvhi_location, 2),
        "pvhi_count":    round(pvhi_count, 2),
        "denominator":   round(denom, 5),
    }


def interpret_pvhi(pvhi: float) -> str:
    if pvhi >= 130:
        return "high_danger"
    if pvhi >= 110:
        return "above_average_danger"
    if pvhi >= 90:
        return "neutral"
    if pvhi >= 70:
        return "safer_than_average"
    return "very_safe"
