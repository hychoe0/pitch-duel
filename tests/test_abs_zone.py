"""Tests for abs_zone.py — ABS strike zone computation."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_compute_abs_zone_6ft():
    from src.hitters.abs_zone import compute_abs_zone
    z = compute_abs_zone(72)   # 6'0"
    assert z["top_ft"]    == pytest.approx(3.21, abs=0.01)
    assert z["bottom_ft"] == pytest.approx(1.62, abs=0.01)
    assert z["left_ft"]   == -0.708
    assert z["right_ft"]  ==  0.708


def test_compute_abs_zone_6ft4():
    from src.hitters.abs_zone import compute_abs_zone
    z = compute_abs_zone(76)   # 6'4"
    assert z["top_ft"]    == pytest.approx(3.39, abs=0.01)
    assert z["bottom_ft"] == pytest.approx(1.71, abs=0.01)


def test_compute_abs_zone_5ft9():
    from src.hitters.abs_zone import compute_abs_zone
    z = compute_abs_zone(69)   # 5'9"
    assert z["top_ft"]    == pytest.approx(3.07, abs=0.01)
    assert z["bottom_ft"] == pytest.approx(1.55, abs=0.01)


def test_parse_height_str_valid():
    from src.hitters.abs_zone import _parse_height_str
    assert _parse_height_str("6' 4\"") == 76
    assert _parse_height_str("5' 9\"") == 69
    assert _parse_height_str("6'0\"")  == 72
    assert _parse_height_str("6'7\"")  == 79


def test_parse_height_str_invalid():
    from src.hitters.abs_zone import _parse_height_str
    assert _parse_height_str("") is None
    assert _parse_height_str("tall") is None


def test_get_height_inches_from_cache():
    from src.hitters.abs_zone import get_height_inches, HEIGHT_CACHE_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "hitter_heights.json"
        cache_path.write_text(json.dumps({"660271": 76, "592450": 79}))
        with patch("src.hitters.abs_zone.HEIGHT_CACHE_PATH", cache_path):
            assert get_height_inches(660271) == 76
            assert get_height_inches(592450) == 79


def test_get_height_inches_cache_miss_returns_none():
    from src.hitters.abs_zone import get_height_inches, HEIGHT_CACHE_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "hitter_heights.json"
        cache_path.write_text(json.dumps({}))
        # Patch urlopen to raise so we test the fallback
        with patch("src.hitters.abs_zone.HEIGHT_CACHE_PATH", cache_path), \
             patch("urllib.request.urlopen", side_effect=OSError("network unavailable")):
            result = get_height_inches(999999)
            assert result is None


def test_get_abs_zone_from_cache():
    from src.hitters.abs_zone import get_abs_zone, HEIGHT_CACHE_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "hitter_heights.json"
        cache_path.write_text(json.dumps({"660271": 76}))
        with patch("src.hitters.abs_zone.HEIGHT_CACHE_PATH", cache_path):
            z = get_abs_zone(660271)
            assert z is not None
            assert z["top_ft"] == pytest.approx(3.39, abs=0.01)


def test_get_abs_zone_unknown_returns_none():
    from src.hitters.abs_zone import get_abs_zone, HEIGHT_CACHE_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "hitter_heights.json"
        cache_path.write_text(json.dumps({}))
        with patch("src.hitters.abs_zone.HEIGHT_CACHE_PATH", cache_path), \
             patch("urllib.request.urlopen", side_effect=OSError("network unavailable")):
            assert get_abs_zone(999999) is None
