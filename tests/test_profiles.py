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
