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


# ---------------------------------------------------------------------------
# compute_zone_xwoba_rates
# ---------------------------------------------------------------------------

def test_zone_xwoba_omits_thin_zone():
    from src.hitters.profiles import compute_zone_xwoba_rates
    # zone 1: 5 batted balls (below MIN_ZONE_BATTED_BALLS=10) — should be omitted
    # zone 5: 15 batted balls — should be present
    df, w = _make_batted_ball_df({1: 5, 5: 15})
    result = compute_zone_xwoba_rates(df, w)
    assert "1" not in result, "Thin zone should be omitted, not filled"
    assert "5" in result
    assert result["5"] == pytest.approx(0.50)


def test_zone_xwoba_weighted_mean():
    from src.hitters.profiles import compute_zone_xwoba_rates
    # 10 rows with xwoba=0.80, 10 rows with xwoba=0.20 → weighted mean 0.50
    rows = (
        [{"description": "hit_into_play", "zone": 5.0,
          "estimated_woba_using_speedangle": 0.80,
          "launch_speed": 98.0, "game_date": pd.Timestamp("2024-06-01")}] * 10
        + [{"description": "hit_into_play", "zone": 5.0,
            "estimated_woba_using_speedangle": 0.20,
            "launch_speed": 85.0, "game_date": pd.Timestamp("2024-06-01")}] * 10
    )
    df = pd.DataFrame(rows)
    w = np.ones(len(df))
    result = compute_zone_xwoba_rates(df, w)
    assert result["5"] == pytest.approx(0.50)


def test_zone_xwoba_excludes_non_batted_ball_descriptions():
    from src.hitters.profiles import compute_zone_xwoba_rates
    rows = [
        # These should be counted
        {"description": "hit_into_play", "zone": 5.0,
         "estimated_woba_using_speedangle": 0.80,
         "launch_speed": 98.0, "game_date": pd.Timestamp("2024-06-01")},
    ] * 15 + [
        # Swinging strike — should NOT be counted toward batted balls
        {"description": "swinging_strike", "zone": 5.0,
         "estimated_woba_using_speedangle": None,
         "launch_speed": None, "game_date": pd.Timestamp("2024-06-01")},
    ] * 100
    df = pd.DataFrame(rows)
    w = np.ones(len(df))
    result = compute_zone_xwoba_rates(df, w)
    # Only the 15 batted balls count; result should reflect xwoba=0.80
    assert result["5"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# compute_zone_hard_hit_rates
# ---------------------------------------------------------------------------

def test_zone_hard_hit_omits_thin_zone():
    from src.hitters.profiles import compute_zone_hard_hit_rates
    # zone 1: 3 batted balls (thin) — should be omitted
    # zone 5: 15 batted balls — should be present
    df, w = _make_batted_ball_df({1: 3, 5: 15})
    result = compute_zone_hard_hit_rates(df, w)
    assert "1" not in result
    assert "5" in result
    assert result["5"] == pytest.approx(1.0)  # _make_batted_ball_df uses launch_speed=98 (>=95)


def test_zone_hard_hit_rate_calculation():
    from src.hitters.profiles import compute_zone_hard_hit_rates
    # zone 5: 10 hard-hit (>=95) + 5 soft-hit (<95) → rate = 10/15
    rows = (
        [{"description": "hit_into_play", "zone": 5.0,
          "launch_speed": 98.0,
          "game_date": pd.Timestamp("2024-06-01")}] * 10
        + [{"description": "hit_into_play", "zone": 5.0,
            "launch_speed": 85.0,
            "game_date": pd.Timestamp("2024-06-01")}] * 5
    )
    df = pd.DataFrame(rows)
    w = np.ones(len(df))
    result = compute_zone_hard_hit_rates(df, w)
    assert result["5"] == pytest.approx(10 / 15)


def test_zone_hard_hit_excludes_non_batted_balls():
    from src.hitters.profiles import compute_zone_hard_hit_rates
    rows = (
        [{"description": "hit_into_play", "zone": 5.0,
          "launch_speed": 98.0, "game_date": pd.Timestamp("2024-06-01")}] * 15
        + [{"description": "swinging_strike", "zone": 5.0,
            "launch_speed": None, "game_date": pd.Timestamp("2024-06-01")}] * 50
    )
    df = pd.DataFrame(rows)
    w = np.ones(len(df))
    result = compute_zone_hard_hit_rates(df, w)
    assert result["5"] == pytest.approx(1.0)  # only the 15 batted balls, all hard-hit
