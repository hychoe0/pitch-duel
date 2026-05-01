"""Tests for profiles.py — hitter profile building and lookup."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legacy_profile_dict() -> dict:
    """Profile dict as saved before zone_xwoba_rates / zone_hard_hit_rates existed."""
    return {
        "player_id": 592450,
        "player_name": "Judge, Aaron",
        "stand": "R",
        "swing_rate": 0.47,
        "chase_rate": 0.30,
        "contact_rate": 0.78,
        "hard_hit_rate": 0.61,
        "whiff_rate": 0.25,
        "swing_rate_by_count": {},
        "contact_rate_by_pitch_type": {},
        "zone_swing_rates": {},
        "zone_whiff_rates": {},
        "family_swing_rates": {},
        "family_whiff_rates": {},
        "sample_size": 1000,
        "is_thin_sample": False,
        "league": "MLB",
        # NOTE: intentionally missing zone_xwoba_rates and zone_hard_hit_rates
    }


def _make_batted_ball_df(zone_counts: dict) -> tuple:
    """
    Build a synthetic batted-ball DataFrame with given per-zone row counts.
    All rows: description=hit_into_play, xwoba=0.50, launch_speed=98.0.
    Returns (df, w) where w is a uniform weight array.
    """
    rows = []
    for zone_id, count in zone_counts.items():
        rows.extend([{
            "description": "hit_into_play",
            "zone": float(zone_id),
            "estimated_woba_using_speedangle": 0.50,
            "launch_speed": 98.0,
            "game_date": pd.Timestamp("2024-06-01"),
        }] * count)
    df = pd.DataFrame(rows)
    w = np.ones(len(df))
    return df, w


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_profile_loads_without_new_fields():
    from src.hitters.profiles import HitterProfile
    profile = HitterProfile(**_legacy_profile_dict())
    assert profile.zone_xwoba_rates == {}
    assert profile.zone_hard_hit_rates == {}


# ---------------------------------------------------------------------------
# resolve_hitter_features_for_pitch
# ---------------------------------------------------------------------------

def _minimal_profile(**overrides):
    from src.hitters.profiles import HitterProfile
    defaults = dict(
        player_id=1, player_name="Test", stand="R",
        swing_rate=0.47, chase_rate=0.30, contact_rate=0.78,
        hard_hit_rate=0.40, whiff_rate=0.25,
    )
    defaults.update(overrides)
    return HitterProfile(**defaults)


def test_resolver_falls_back_to_league_avg_xwoba_when_zone_missing():
    from src.hitters.profiles import resolve_hitter_features_for_pitch, LEAGUE_AVG
    profile = _minimal_profile()  # zone_xwoba_rates = {} (default)
    result = resolve_hitter_features_for_pitch(profile, zone=5)
    assert result["hitter_zone_xwoba"] == pytest.approx(LEAGUE_AVG["zone_xwoba"])


def test_resolver_falls_back_to_profile_hard_hit_rate_when_zone_missing():
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(hard_hit_rate=0.55)
    result = resolve_hitter_features_for_pitch(profile, zone=5)
    assert result["hitter_zone_hard_hit_rate"] == pytest.approx(0.55)


def test_resolver_uses_zone_data_when_present():
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(
        zone_xwoba_rates={"5": 0.72},
        zone_hard_hit_rates={"5": 0.65},
    )
    result = resolve_hitter_features_for_pitch(profile, zone=5)
    assert result["hitter_zone_xwoba"] == pytest.approx(0.72)
    assert result["hitter_zone_hard_hit_rate"] == pytest.approx(0.65)


def test_resolver_returns_all_five_global_rates():
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile()
    result = resolve_hitter_features_for_pitch(profile, zone=5)
    for key in ("hitter_swing_rate", "hitter_chase_rate", "hitter_contact_rate",
                "hitter_hard_hit_rate", "hitter_whiff_rate"):
        assert key in result
