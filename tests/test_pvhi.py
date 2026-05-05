"""Tests for pvhi.py and the overall_xwoba_per_pitch profile field."""
import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(**overrides):
    from src.hitters.profiles import HitterProfile
    defaults = dict(
        player_id=1, player_name="Test", stand="R",
        swing_rate=0.47, chase_rate=0.30, contact_rate=0.78,
        hard_hit_rate=0.38, whiff_rate=0.25,
        overall_xwoba_per_pitch=0.085,
    )
    defaults.update(overrides)
    return HitterProfile(**defaults)


def _make_stuff(xwoba=0.200, whiff=0.250):
    return {"stuff_vs_hitter_xwoba": xwoba, "stuff_vs_hitter_whiff": whiff,
            "n_neighbors_found": 30, "similarity_quality": 0.8}


def _make_context(
    swing_this_count=0.47,
    zone_swing=0.50,
    zone_whiff=0.25,
    zone_xwoba=0.380,
    zone_hard_hit=0.38,
    contact_this_pitch=0.78,
    family_swing=0.47,
    family_whiff=0.25,
):
    return {
        "hitter_swing_rate_this_count":   swing_this_count,
        "hitter_zone_swing_rate":         zone_swing,
        "hitter_zone_whiff_rate":         zone_whiff,
        "hitter_zone_xwoba":              zone_xwoba,
        "hitter_zone_hard_hit_rate":      zone_hard_hit,
        "hitter_contact_rate_this_pitch": contact_this_pitch,
        "hitter_family_swing_rate":       family_swing,
        "hitter_family_whiff_rate":       family_whiff,
    }


def _make_pitch(**overrides):
    p = dict(
        pitch_type="FF", release_speed=94.0, balls=1, strikes=2,
        plate_x=0.0, plate_z=2.5, p_throws="R",
    )
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# Test 1 — compute_pvhi returns all 5 required keys
# ---------------------------------------------------------------------------

def test_compute_pvhi_returns_all_keys():
    from src.model.pvhi import compute_pvhi
    r = compute_pvhi(_make_pitch(), _make_profile(), _make_stuff(), _make_context())
    assert set(r) == {"pvhi", "pvhi_stuff", "pvhi_location", "pvhi_count", "denominator"}


# ---------------------------------------------------------------------------
# Test 2 — denominator fallback: 0.0 overall_xwoba_per_pitch → league avg
# ---------------------------------------------------------------------------

def test_denominator_falls_back_when_zero():
    from src.model.pvhi import compute_pvhi
    from src.hitters.profiles import LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH
    profile = _make_profile(overall_xwoba_per_pitch=0.0)
    r = compute_pvhi(_make_pitch(), profile, _make_stuff(), _make_context())
    assert r["denominator"] == pytest.approx(LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH)
    assert r["pvhi"] > 0  # no divide-by-zero


# ---------------------------------------------------------------------------
# Test 3 — interpretation thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pvhi,expected", [
    (130.0,  "high_danger"),
    (129.9,  "above_average_danger"),
    (110.0,  "above_average_danger"),
    (109.9,  "neutral"),
    (90.0,   "neutral"),
    (89.9,   "safer_than_average"),
    (70.0,   "safer_than_average"),
    (69.9,   "very_safe"),
    (0.0,    "very_safe"),
])
def test_interpretation_thresholds(pvhi, expected):
    from src.model.pvhi import interpret_pvhi
    assert interpret_pvhi(pvhi) == expected


# ---------------------------------------------------------------------------
# Test 4 — clipping: extreme inputs capped at [0, 250]
# ---------------------------------------------------------------------------

def test_extreme_high_inputs_clipped():
    from src.model.pvhi import compute_pvhi
    # zone_xwoba=2.0 would normally blow up the location component
    r = compute_pvhi(
        _make_pitch(),
        _make_profile(),
        _make_stuff(xwoba=2.0),
        _make_context(zone_xwoba=2.0, zone_swing=1.0),
    )
    assert r["pvhi"]          <= 250.0
    assert r["pvhi_stuff"]    <= 250.0
    assert r["pvhi_location"] <= 250.0
    assert r["pvhi_count"]    <= 250.0


def test_extreme_low_inputs_clipped():
    from src.model.pvhi import compute_pvhi
    r = compute_pvhi(
        _make_pitch(),
        _make_profile(),
        _make_stuff(xwoba=0.0),
        _make_context(zone_xwoba=0.0, zone_swing=0.0),
    )
    assert r["pvhi"]          >= 0.0
    assert r["pvhi_stuff"]    >= 0.0
    assert r["pvhi_location"] >= 0.0
    assert r["pvhi_count"]    >= 0.0


# ---------------------------------------------------------------------------
# Test 5 — Two different pitches against same hitter produce different PVHI
# ---------------------------------------------------------------------------

def test_different_pitches_produce_different_pvhi():
    from src.model.pvhi import compute_pvhi
    profile = _make_profile()
    stuff   = _make_stuff()

    # Power zone: zone 1 (high-in), 3-1 count
    ctx_hot = _make_context(zone_xwoba=0.80, zone_swing=0.75, zone_whiff=0.10,
                            swing_this_count=0.65)
    # Chase: zone 14 (low-away), 1-2 count
    ctx_chase = _make_context(zone_xwoba=0.22, zone_swing=0.25, zone_whiff=0.45,
                              swing_this_count=0.38)

    r_hot   = compute_pvhi(_make_pitch(balls=3, strikes=1), profile, stuff, ctx_hot)
    r_chase = compute_pvhi(_make_pitch(balls=1, strikes=2), profile, stuff, ctx_chase)

    assert r_hot["pvhi"] != pytest.approx(r_chase["pvhi"], abs=1e-3)
    assert r_hot["pvhi"] > r_chase["pvhi"]


# ---------------------------------------------------------------------------
# Test 6 — Same pitch, different hitters produce different PVHI
# ---------------------------------------------------------------------------

def test_different_hitters_produce_different_pvhi():
    from src.model.pvhi import compute_pvhi
    stuff = _make_stuff()
    ctx   = _make_context()

    # Power hitter: high overall xwOBA per pitch, aggressive zone swing
    power_profile = _make_profile(
        overall_xwoba_per_pitch=0.11, swing_rate=0.50,
        zone_xwoba_rates={"1": 0.80},
        zone_swing_rates={1: 0.70},
    )
    # Contact hitter: low overall xwOBA, passive in zone
    contact_profile = _make_profile(
        overall_xwoba_per_pitch=0.06, swing_rate=0.42,
        zone_xwoba_rates={"1": 0.35},
        zone_swing_rates={1: 0.45},
    )

    # Use context with same zone values — PVHI difference comes from denominator
    r_power   = compute_pvhi(_make_pitch(), power_profile,   stuff, ctx)
    r_contact = compute_pvhi(_make_pitch(), contact_profile, stuff, ctx)

    assert r_power["pvhi"] != pytest.approx(r_contact["pvhi"], abs=1e-3)


# ---------------------------------------------------------------------------
# Test 7 — overall_xwoba_per_pitch backward compat (legacy JSON without field)
# ---------------------------------------------------------------------------

def test_legacy_profile_loads_without_overall_xwoba(tmp_path):
    from src.hitters.profiles import HitterProfile
    legacy = {
        "player_id": 42, "player_name": "Old Player", "stand": "R",
        "swing_rate": 0.47, "chase_rate": 0.30,
        "contact_rate": 0.78, "hard_hit_rate": 0.38, "whiff_rate": 0.25,
        # overall_xwoba_per_pitch intentionally absent
        "sample_size": 1000, "is_thin_sample": False,
    }
    path = tmp_path / "42.json"
    path.write_text(json.dumps(legacy))
    data = json.loads(path.read_text())
    profile = HitterProfile(**data)
    assert profile.overall_xwoba_per_pitch == 0.0


# ---------------------------------------------------------------------------
# Integration tests (require built profiles)
# ---------------------------------------------------------------------------

_JUDGE_PROFILE = Path("data/processed/profiles/592450.json")
_KIF_PROFILE   = Path("data/processed/profiles/643396.json")


@pytest.mark.skipif(not _JUDGE_PROFILE.exists(), reason="Profiles not built")
def test_judge_overall_xwoba_above_league_avg():
    from src.hitters.profiles import (
        HitterProfile, LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH, get_overall_xwoba_per_pitch
    )
    profile = HitterProfile(**json.loads(_JUDGE_PROFILE.read_text()))
    val = get_overall_xwoba_per_pitch(profile)
    assert val > LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH, (
        f"Judge overall_xwoba_per_pitch={val:.4f} should exceed league avg "
        f"{LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH}"
    )


@pytest.mark.skipif(not _KIF_PROFILE.exists(), reason="Profiles not built")
def test_kif_overall_xwoba_below_judge():
    from src.hitters.profiles import HitterProfile, get_overall_xwoba_per_pitch
    judge  = HitterProfile(**json.loads(_JUDGE_PROFILE.read_text()))
    kif    = HitterProfile(**json.loads(_KIF_PROFILE.read_text()))
    assert get_overall_xwoba_per_pitch(kif) < get_overall_xwoba_per_pitch(judge), (
        "Kiner-Falefa's per-pitch xwOBA should be below Judge's"
    )


@pytest.mark.skipif(not _KIF_PROFILE.exists(), reason="Profiles not built")
def test_thin_sample_profile_gets_league_avg_fallback():
    from src.hitters.profiles import HitterProfile, LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH
    profile = HitterProfile(
        player_id=9999, player_name="Thin", stand="R",
        swing_rate=0.47, chase_rate=0.30, contact_rate=0.78,
        hard_hit_rate=0.38, whiff_rate=0.25,
        overall_xwoba_per_pitch=0.0,  # thin-sample gets 0.0
        sample_size=0, is_thin_sample=True,
    )
    from src.hitters.profiles import get_overall_xwoba_per_pitch
    assert get_overall_xwoba_per_pitch(profile) == pytest.approx(LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH)
