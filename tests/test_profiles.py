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


def test_resolver_returns_exactly_13_keys():
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile()
    result = resolve_hitter_features_for_pitch(
        profile, zone=5, balls=1, strikes=2, pitch_type="FF"
    )
    assert len(result) == 13, f"Expected 13 keys, got {len(result)}: {sorted(result)}"
    expected_keys = {
        "hitter_swing_rate", "hitter_chase_rate", "hitter_contact_rate",
        "hitter_hard_hit_rate", "hitter_whiff_rate",
        "hitter_swing_rate_this_count",
        "hitter_zone_swing_rate", "hitter_zone_whiff_rate",
        "hitter_zone_xwoba", "hitter_zone_hard_hit_rate",
        "hitter_contact_rate_this_pitch",
        "hitter_family_swing_rate", "hitter_family_whiff_rate",
    }
    assert set(result) == expected_keys


def test_resolver_count_fallback_invalid_count():
    """balls=9, strikes=9 is not a real count — must return profile.swing_rate, no KeyError."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(swing_rate=0.42)
    result = resolve_hitter_features_for_pitch(
        profile, zone=5, balls=9, strikes=9, pitch_type="FF"
    )
    assert result["hitter_swing_rate_this_count"] == pytest.approx(0.42)


def test_resolver_count_uses_profile_data_when_available():
    """Valid count key from swing_rate_by_count takes priority over global rate."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(
        swing_rate=0.42,
        swing_rate_by_count={"0-0": 0.325},
    )
    result = resolve_hitter_features_for_pitch(
        profile, zone=5, balls=0, strikes=0, pitch_type="FF"
    )
    assert result["hitter_swing_rate_this_count"] == pytest.approx(0.325)


def test_resolver_zone_fallback_invalid_zone():
    """zone=99 is invalid — all zone features return their defined fallbacks, no exception."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch, LEAGUE_AVG
    profile = _minimal_profile(swing_rate=0.42, whiff_rate=0.28, hard_hit_rate=0.50)
    result = resolve_hitter_features_for_pitch(
        profile, zone=99, balls=0, strikes=0, pitch_type="FF"
    )
    assert result["hitter_zone_swing_rate"]    == pytest.approx(0.42)   # falls back to profile.swing_rate
    assert result["hitter_zone_whiff_rate"]    == pytest.approx(0.28)   # falls back to profile.whiff_rate
    assert result["hitter_zone_xwoba"]         == pytest.approx(LEAGUE_AVG["zone_xwoba"])
    assert result["hitter_zone_hard_hit_rate"] == pytest.approx(0.50)   # falls back to profile.hard_hit_rate


def test_resolver_zone_uses_str_keys():
    """Profile zone dicts stored with str keys (JSON round-trip) work correctly."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(
        zone_swing_rates={"5": 0.61},   # str keys as loaded from JSON
        zone_whiff_rates={"5": 0.33},
    )
    result = resolve_hitter_features_for_pitch(profile, zone=5)
    assert result["hitter_zone_swing_rate"] == pytest.approx(0.61)
    assert result["hitter_zone_whiff_rate"] == pytest.approx(0.33)


def test_resolver_zone_uses_int_keys():
    """Profile zone dicts stored with int keys (fresh build) work correctly."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(
        zone_swing_rates={5: 0.61},   # int keys as built in-memory
        zone_whiff_rates={5: 0.33},
    )
    result = resolve_hitter_features_for_pitch(profile, zone=5)
    assert result["hitter_zone_swing_rate"] == pytest.approx(0.61)
    assert result["hitter_zone_whiff_rate"] == pytest.approx(0.33)


def test_resolver_pitch_type_fallback_unknown():
    """pitch_type='XX' is unknown — contact_rate_this_pitch falls back to profile.contact_rate."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(
        contact_rate=0.78,
        swing_rate=0.42,
        whiff_rate=0.28,
    )
    result = resolve_hitter_features_for_pitch(
        profile, zone=5, balls=0, strikes=0, pitch_type="XX"
    )
    assert result["hitter_contact_rate_this_pitch"] == pytest.approx(0.78)
    # Unknown pitch_type maps to "other" family — falls back to global rates
    assert result["hitter_family_swing_rate"] == pytest.approx(0.42)
    assert result["hitter_family_whiff_rate"] == pytest.approx(0.28)


def test_resolver_pitch_type_uses_profile_data():
    """Known pitch_type with profile data uses the specific rate."""
    from src.hitters.profiles import resolve_hitter_features_for_pitch
    profile = _minimal_profile(
        contact_rate=0.78,
        contact_rate_by_pitch_type={"SL": 0.62},
        family_swing_rates={"breaking": 0.51},
        family_whiff_rates={"breaking": 0.38},
    )
    result = resolve_hitter_features_for_pitch(
        profile, zone=5, balls=1, strikes=2, pitch_type="SL"
    )
    assert result["hitter_contact_rate_this_pitch"] == pytest.approx(0.62)
    assert result["hitter_family_swing_rate"]       == pytest.approx(0.51)
    assert result["hitter_family_whiff_rate"]       == pytest.approx(0.38)


# ---------------------------------------------------------------------------
# PITCH_FAMILY_MAP coverage
# ---------------------------------------------------------------------------

def test_pitch_family_map_ff_is_fastball():
    from src.hitters.profiles import PITCH_FAMILY_MAP
    assert PITCH_FAMILY_MAP["FF"] == "fastball"
    assert PITCH_FAMILY_MAP["SI"] == "fastball"
    assert PITCH_FAMILY_MAP["FC"] == "fastball"


def test_pitch_family_map_sl_is_breaking():
    from src.hitters.profiles import PITCH_FAMILY_MAP
    assert PITCH_FAMILY_MAP["SL"] == "breaking"
    assert PITCH_FAMILY_MAP["ST"] == "breaking"
    assert PITCH_FAMILY_MAP["CU"] == "breaking"


def test_pitch_family_map_ch_is_offspeed():
    from src.hitters.profiles import PITCH_FAMILY_MAP
    assert PITCH_FAMILY_MAP["CH"] == "offspeed"
    assert PITCH_FAMILY_MAP["FS"] == "offspeed"
    assert PITCH_FAMILY_MAP["FO"] == "offspeed"


def test_pitch_family_map_unknown_returns_other_via_get():
    from src.hitters.profiles import PITCH_FAMILY_MAP
    assert PITCH_FAMILY_MAP.get("XX", "other") == "other"


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


# ---------------------------------------------------------------------------
# Integration: Judge zone xwOBA validation (requires --rebuild-demo to have run)
# ---------------------------------------------------------------------------

_JUDGE_PROFILE_PATH = Path("data/processed/profiles/592450.json")


@pytest.mark.skipif(
    not _JUDGE_PROFILE_PATH.exists(),
    reason="Run 'python -m src.hitters.profiles --rebuild-demo' first",
)
def test_judge_zone_xwoba_heart_of_plate_above_threshold():
    from src.hitters.profiles import load_profile
    profile = load_profile(592450)
    assert "5" in profile.zone_xwoba_rates, (
        "Zone 5 (heart of plate) should have enough batted balls for Judge"
    )
    assert profile.zone_xwoba_rates["5"] > 0.40, (
        f"Expected xwOBA > 0.40 in zone 5 for Judge, got {profile.zone_xwoba_rates['5']:.3f}"
    )


@pytest.mark.skipif(
    not _JUDGE_PROFILE_PATH.exists(),
    reason="Run 'python -m src.hitters.profiles --rebuild-demo' first",
)
def test_judge_zone_xwoba_chase_lower_than_heart():
    from src.hitters.profiles import load_profile
    profile = load_profile(592450)
    if "14" not in profile.zone_xwoba_rates:
        pytest.skip("Zone 14 below MIN_ZONE_BATTED_BALLS threshold for Judge — skip comparison")
    assert profile.zone_xwoba_rates["14"] < profile.zone_xwoba_rates["5"], (
        "Chase zone (14) xwOBA should be lower than heart-of-plate (5) for Judge"
    )
