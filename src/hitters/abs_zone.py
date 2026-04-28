"""
abs_zone.py — ABS strike zone bounds for a hitter of given height.

2026 MLB ABS specs:
  Top:    53.5% of height
  Bottom: 27.0% of height
  Width:  17 inches (±0.708 ft) centered on plate

All values in feet to match plate_x / plate_z units in Statcast.
"""

import json
import re
import urllib.request
from pathlib import Path

HEIGHT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "hitter_heights.json"


def compute_abs_zone(height_inches: int) -> dict:
    return {
        "top_ft":    round((height_inches * 0.535) / 12.0, 3),
        "bottom_ft": round((height_inches * 0.270) / 12.0, 3),
        "left_ft":  -0.708,
        "right_ft":  0.708,
    }


def _parse_height_str(s: str) -> int | None:
    """Parse '6\' 4\"' → 76. Returns None on failure."""
    m = re.match(r"(\d+)'\s*(\d+)", s)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def _load_cache() -> dict:
    if HEIGHT_CACHE_PATH.exists():
        return json.loads(HEIGHT_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    HEIGHT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEIGHT_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_height_inches(player_id: int, timeout: float = 2.0) -> int | None:
    """
    Return height in inches for player_id.

    Checks cache first. On miss, queries statsapi.mlb.com (2-second timeout).
    Caches the result if successful. Returns None if not in cache and API fails.
    """
    cache = _load_cache()
    key = str(player_id)
    if key in cache:
        return cache[key]

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        height_str = data["people"][0].get("height", "")
        inches = _parse_height_str(height_str)
        # Cache both successful and "API responded but no parseable height" outcomes
        # so we don't repeatedly hit the network for the same unknown player.
        # Network failures are NOT cached — allow retry on transient errors.
        cache[key] = inches
        _save_cache(cache)
        return inches
    except Exception:
        return None


def get_abs_zone(player_id: int) -> dict | None:
    """Return ABS zone dict for player_id, or None if height unknown."""
    h = get_height_inches(player_id)
    if h is None:
        return None
    return compute_abs_zone(h)
