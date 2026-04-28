# Task 8: ABS Strike Zone + UX Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-hitter ABS strike zone to all pitch-location SVGs, relabel "xwOBA per pitch" to "Expected damage (per-pitch)", and fix Path B outcomes showing literal "nan".

**Architecture:** Three independent parts: (1) new `abs_zone.py` helper + height cache JSON, wired into `server.py` profile/prediction endpoints, consumed by `index.html` zone SVG functions; (2) text-only relabeling in `index.html`; (3) server-side post-processing for Path B pitch outcomes.

**Tech Stack:** Python 3.11, Flask, XGBoost (unchanged), vanilla JS/SVG in index.html

---

## File Inventory

| File | Action | What changes |
|------|--------|--------------|
| `src/hitters/abs_zone.py` | **Create** | `compute_abs_zone()`, `_parse_height_str()`, `get_height_inches()`, `get_abs_zone()` |
| `data/processed/hitter_heights.json` | **Create** | 12 demo hitters: player_id → height_inches |
| `tests/test_abs_zone.py` | **Create** | Unit tests for `compute_abs_zone()` and cache-based `get_height_inches()` |
| `src/demo/server.py` | **Modify** | (a) import abs_zone helpers; (b) augment `_load_hitter_profile_for_display()`; (c) add `abs_zone`/`hitter_height_inches` to predict response; (d) add `_EVENT_LABELS`, `_DESCRIPTION_LABELS`, `_resolve_pitch_outcome()`; (e) post-process `top_pitches` |
| `src/demo/templates/index.html` | **Modify** | (a) add `_absZone_demo`/`_absZone_atbat` state vars; (b) update `loadHitterCard()` to store zone and redraw; (c) update `buildPickerZoneSvg()` signature and body; (d) update `buildZoneSvg()` to accept `opts.absZone`; (e) update all 4 affected call sites; (f) relabel xwOBA in 2 labels + 2 tooltips + About panel |

---

## Task 1: `abs_zone.py` — ABS zone computations

**Files:**
- Create: `src/hitters/abs_zone.py`
- Create: `tests/test_abs_zone.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_abs_zone.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect all FAIL with ImportError**

```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -m pytest tests/test_abs_zone.py -v 2>&1 | head -30
```

Expected: 8 errors all with `ModuleNotFoundError: No module named 'src.hitters.abs_zone'`

- [ ] **Step 3: Create `src/hitters/abs_zone.py`**

```python
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

HEIGHT_CACHE_PATH = Path("data/processed/hitter_heights.json")


def compute_abs_zone(height_inches: int) -> dict:
    return {
        "top_ft":    round((height_inches * 0.535) / 12.0, 3),
        "bottom_ft": round((height_inches * 0.270) / 12.0, 3),
        "left_ft":  -0.708,
        "right_ft":  0.708,
    }


def _parse_height_str(s: str) -> int | None:
    """Parse '6\\' 4\\"' → 76. Returns None on failure."""
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
        if inches is not None:
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
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -m pytest tests/test_abs_zone.py -v
```

Expected output:
```
tests/test_abs_zone.py::test_compute_abs_zone_6ft PASSED
tests/test_abs_zone.py::test_compute_abs_zone_6ft4 PASSED
tests/test_abs_zone.py::test_compute_abs_zone_5ft9 PASSED
tests/test_abs_zone.py::test_parse_height_str_valid PASSED
tests/test_abs_zone.py::test_parse_height_str_invalid PASSED
tests/test_abs_zone.py::test_get_height_inches_from_cache PASSED
tests/test_abs_zone.py::test_get_height_inches_cache_miss_returns_none PASSED
tests/test_abs_zone.py::test_get_abs_zone_from_cache PASSED
tests/test_abs_zone.py::test_get_abs_zone_unknown_returns_none PASSED
9 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/hitters/abs_zone.py tests/test_abs_zone.py
git commit -m "feat: add abs_zone.py — compute ABS strike zone bounds from hitter height"
```

---

## Task 2: Height cache JSON

**Files:**
- Create: `data/processed/hitter_heights.json`

- [ ] **Step 1: Create `data/processed/hitter_heights.json`**

Heights sourced from MLB stats API for 9 IDs; 3 hardcoded (682998/Merrill, 119534/Murakami, 605196/McGonigle — API returns wrong player for these IDs):

```json
{
  "119534": 74,
  "592450": 79,
  "596019": 70,
  "605196": 71,
  "608070": 68,
  "660271": 76,
  "665966": 71,
  "682998": 75,
  "686948": 72,
  "691176": 69,
  "693307": 73,
  "701762": 77
}
```

- [ ] **Step 2: Verify the cache is readable by `abs_zone.py`**

```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -c "
from src.hitters.abs_zone import get_height_inches, get_abs_zone
print('Judge (660271):', get_height_inches(660271), 'in')
z = get_abs_zone(660271)
print('  top:', z['top_ft'], 'bottom:', z['bottom_ft'])
print('Altuve (608070):', get_height_inches(608070), 'in')
z2 = get_abs_zone(608070)
print('  top:', z2['top_ft'], 'bottom:', z2['bottom_ft'])
"
```

Expected:
```
Judge (660271): 76 in
  top: 3.389 bottom: 1.71
Altuve (608070): 68 in
  top: 3.032 bottom: 1.53
```

- [ ] **Step 3: Commit**

```bash
git add data/processed/hitter_heights.json
git commit -m "data: add hitter_heights.json — 12 demo hitters for ABS zone display"
```

---

## Task 3: `server.py` Path B fix — clean "nan" outcomes

**Files:**
- Modify: `src/demo/server.py`

This is independent of the ABS zone work. The root cause: `float('nan')` is truthy in Python, so `str(row.get("events", "") or "")` evaluates to `"nan"` for NaN events. Fix: post-process `mr.top_similar_pitches` before serialization to add a clean `"outcome"` string.

- [ ] **Step 1: Add label dicts and `_resolve_pitch_outcome()` to `server.py`**

Open `src/demo/server.py`. Insert after the `ZONE_X_MIN, ZONE_X_MAX = ...` block (after line 88), before `def classify_pitch_zone`:

```python
# ──────────────────────────────────────────────────────────────────────────────
# Path B outcome labels
# ──────────────────────────────────────────────────────────────────────────────

_EVENT_LABELS = {
    "single": "Single", "double": "Double", "triple": "Triple",
    "home_run": "Home Run", "strikeout": "Strikeout",
    "walk": "Walk", "intent_walk": "Walk (IBB)", "hit_by_pitch": "HBP",
    "field_out": "Field Out", "grounded_into_double_play": "GIDP",
    "force_out": "Force Out", "double_play": "Double Play",
    "sac_fly": "Sac Fly", "fielders_choice": "Fielder's Choice",
}
_DESCRIPTION_LABELS = {
    "ball": "Ball", "blocked_ball": "Ball (blocked)",
    "called_strike": "Called Strike",
    "swinging_strike": "Swinging Strike",
    "swinging_strike_blocked": "Swinging Strike (blocked)",
    "foul": "Foul", "foul_tip": "Foul Tip", "foul_bunt": "Foul Bunt",
    "hit_into_play": "Ball in Play",
    "missed_bunt": "Missed Bunt", "pitchout": "Pitchout",
}


def _resolve_pitch_outcome(p: dict) -> str:
    ev   = p.get("events",      "")
    desc = p.get("description", "")
    if ev and ev != "nan":
        return _EVENT_LABELS.get(ev, ev.replace("_", " ").title())
    if desc and desc != "nan":
        return _DESCRIPTION_LABELS.get(desc, desc.replace("_", " ").title())
    return "Unknown"
```

- [ ] **Step 2: Post-process `top_pitches` in the `predict_demo` response**

In `predict_demo()`, find the `"historical"` section of the response dict (around line 354):

```python
            # Path B: Historical similarity
            "historical": {
                "swing_rate": round(mr.historical_swing_rate, 4),
                "contact_rate": round(mr.historical_contact_rate, 4),
                "hard_hit_rate": round(mr.historical_hard_hit_rate, 4),
                "p_hard": round(mr.historical_p_hard, 4),
                "n_similar": mr.n_similar_pitches,
                "confidence": round(mr.confidence, 4),
                "top_pitches": mr.top_similar_pitches,
                "outcome_distribution": mr.outcome_distribution,
            },
```

Replace `"top_pitches": mr.top_similar_pitches,` with:

```python
                "top_pitches": [
                    {**p, "outcome": _resolve_pitch_outcome(p)}
                    for p in mr.top_similar_pitches
                ],
```

The full updated block becomes:
```python
            # Path B: Historical similarity
            "historical": {
                "swing_rate": round(mr.historical_swing_rate, 4),
                "contact_rate": round(mr.historical_contact_rate, 4),
                "hard_hit_rate": round(mr.historical_hard_hit_rate, 4),
                "p_hard": round(mr.historical_p_hard, 4),
                "n_similar": mr.n_similar_pitches,
                "confidence": round(mr.confidence, 4),
                "top_pitches": [
                    {**p, "outcome": _resolve_pitch_outcome(p)}
                    for p in mr.top_similar_pitches
                ],
                "outcome_distribution": mr.outcome_distribution,
            },
```

- [ ] **Step 3: Update the JS outcome resolution in `index.html`**

In `index.html`, find the Path B top similar pitches loop (around line 1101):

```js
                    for (const p of h.top_pitches) {
                        const outcome = p.events || p.description;
```

Change that one line:

```js
                    for (const p of h.top_pitches) {
                        const outcome = p.outcome || p.events || p.description || 'Unknown';
```

- [ ] **Step 4: Smoke-test the Path B fix**

Start the server in a background terminal:
```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -m src.demo.server &
sleep 2
curl -s -X POST http://localhost:5050/api/predict_demo \
  -H "Content-Type: application/json" \
  -d '{"hitter_name":"Aaron Judge","pitch":{"pitch_type":"FF","release_speed":94,"plate_x":0,"plate_z":2.5,"balls":0,"strikes":0}}' \
  | python -m json.tool | grep -A5 '"top_pitches"' | head -20
```

Expected: top_pitches entries have `"outcome": "Ball"` or `"outcome": "Called Strike"` etc. — no `"nan"` values.

Kill the test server:
```bash
kill %1 2>/dev/null; true
```

- [ ] **Step 5: Commit**

```bash
git add src/demo/server.py src/demo/templates/index.html
git commit -m "fix: resolve Path B 'nan' outcomes — server post-processes events/description into clean outcome string"
```

---

## Task 4: `server.py` ABS zone integration

**Files:**
- Modify: `src/demo/server.py`

- [ ] **Step 1: Add the import at the top of `server.py`**

In `src/demo/server.py`, find the existing imports block. After `from src.model.predict_combined import predict_matchup`, add:

```python
from src.hitters.abs_zone import get_abs_zone, get_height_inches
```

- [ ] **Step 2: Augment `_load_hitter_profile_for_display()`**

The function currently returns dicts without ABS zone info. The updated version tracks `pid` across both lookup branches and appends `height_inches`, `height_display`, and `abs_zone`.

Replace the entire `_load_hitter_profile_for_display` function (lines 400–447) with:

```python
def _load_hitter_profile_for_display(hitter_name: str) -> dict:
    """Load hitter profile summary for frontend display."""
    from src.hitters.profiles import get_player_id, load_profile

    profile = None
    pid = None

    # Try parquet-based lookup first
    try:
        df = pd.read_parquet(
            ROOT / "data" / "processed" / "statcast_processed.parquet",
            columns=["batter", "player_name"],
        )
        pid = get_player_id(hitter_name, df)
        profile = load_profile(pid, PROFILE_DIR)
    except (FileNotFoundError, ValueError):
        pass

    # Fallback: scan profile JSONs directly (works without parquet)
    if profile is None:
        for path in PROFILE_DIR.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("player_name", "").lower() == hitter_name.lower():
                    pid = data["player_id"]
                    profile = load_profile(pid, PROFILE_DIR)
                    break
            except Exception:
                continue

    # Look up ABS zone (reads from local cache — no network call for demo hitters)
    h_in = get_height_inches(pid) if pid is not None else None
    abs_zone = get_abs_zone(pid) if pid is not None else None
    height_display = f"{h_in // 12}'{h_in % 12}\"" if h_in is not None else None

    if profile:
        return {
            "player_name": profile.player_name,
            "swing_rate": round(profile.swing_rate, 3),
            "chase_rate": round(profile.chase_rate, 3),
            "contact_rate": round(profile.contact_rate, 3),
            "hard_hit_rate": round(profile.hard_hit_rate, 3),
            "whiff_rate": round(profile.whiff_rate, 3),
            "sample_size": profile.sample_size,
            "is_thin_sample": profile.is_thin_sample,
            "height_inches": h_in,
            "height_display": height_display,
            "abs_zone": abs_zone,
        }

    return {
        "player_name": hitter_name,
        "swing_rate": 0.47, "chase_rate": 0.29,
        "contact_rate": 0.72, "hard_hit_rate": 0.35,
        "whiff_rate": 0.25, "sample_size": 0,
        "is_thin_sample": True,
        "height_inches": None,
        "height_display": None,
        "abs_zone": None,
    }
```

- [ ] **Step 3: Add `abs_zone` and `hitter_height_inches` to the `/api/predict_demo` response**

In `predict_demo()`, find the `response = { ... }` dict (around line 336). After `"hitter_profile": hitter_profile,`, add two top-level fields:

```python
        response = {
            "input_features": input_features,
            "model_version": model_version,
            "model_label": MODEL_VERSIONS.get(model_version, {}).get("label", "unknown"),
            "hitter": hitter_name,
            "hitter_profile": hitter_profile,
            "abs_zone": hitter_profile.get("abs_zone"),
            "hitter_height_inches": hitter_profile.get("height_inches"),
            ...
        }
```

The full updated `response` dict (replace lines 336–390 in the current file):

```python
        response = {
            "input_features": input_features,
            "model_version": model_version,
            "model_label": MODEL_VERSIONS.get(model_version, {}).get("label", "unknown"),
            "hitter": hitter_name,
            "hitter_profile": hitter_profile,
            "abs_zone": hitter_profile.get("abs_zone"),
            "hitter_height_inches": hitter_profile.get("height_inches"),

            # Path A: Model predictions
            "model": {
                "p_swing": round(mr.model_p_swing, 4),
                "p_contact": round(mr.model_p_contact, 4),
                "p_hard_contact": round(mr.model_p_hard_contact, 4),
                "p_hard": round(mr.model_p_hard, 4),
                "stage_drivers": stage_drivers,
            },

            # Path B: Historical similarity
            "historical": {
                "swing_rate": round(mr.historical_swing_rate, 4),
                "contact_rate": round(mr.historical_contact_rate, 4),
                "hard_hit_rate": round(mr.historical_hard_hit_rate, 4),
                "p_hard": round(mr.historical_p_hard, 4),
                "n_similar": mr.n_similar_pitches,
                "confidence": round(mr.confidence, 4),
                "top_pitches": [
                    {**p, "outcome": _resolve_pitch_outcome(p)}
                    for p in mr.top_similar_pitches
                ],
                "outcome_distribution": mr.outcome_distribution,
            },

            # Blend
            "blend": {
                "alpha": round(mr.alpha, 4),
                "p_swing": round(mr.p_swing, 4),
                "p_contact": round(mr.p_contact, 4),
                "p_hard_contact": round(mr.p_hard_contact, 4),
                "p_hard": round(mr.p_hard, 4),
            },

            # Verdict
            "verdict": {
                "danger_level": danger,
                "zone": zone,
                "quality_color": color,
                "pitch_quality": mr.pitch_type,
                "summary": _verdict_text(mr, danger),
            },

            # xwOBA regressor (parallel output — None if model not yet trained)
            "xwoba": {
                "predicted_xwoba_per_pitch": (
                    round(mr.predicted_xwoba_per_pitch, 3)
                    if mr.predicted_xwoba_per_pitch is not None else None
                ),
                "xwoba_context": mr.xwoba_context,
            },
        }
```

- [ ] **Step 4: Smoke-test the ABS zone in `/api/hitter_profile`**

```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -m src.demo.server &
sleep 2
curl -s "http://localhost:5050/api/hitter_profile?name=Aaron+Judge" | python -m json.tool | grep -A6 '"abs_zone"'
```

Expected:
```json
"abs_zone": {
    "bottom_ft": 1.71,
    "left_ft": -0.708,
    "right_ft": 0.708,
    "top_ft": 3.389
},
```

Also test a hitter NOT in the cache (any fallback profile):
```bash
curl -s "http://localhost:5050/api/hitter_profile?name=Shohei+Ohtani" | python -m json.tool | grep '"abs_zone"'
```

Expected: `"abs_zone": null` OR a valid zone dict (Ohtani is in the cache as 660271, but verify the player_id matches).

Kill the test server:
```bash
kill %1 2>/dev/null; true
```

- [ ] **Step 5: Commit**

```bash
git add src/demo/server.py
git commit -m "feat: wire ABS zone into server.py — hitter profile and predict endpoints include abs_zone + height"
```

---

## Task 5: `index.html` — xwOBA relabeling

**Files:**
- Modify: `src/demo/templates/index.html`

Four text changes and two About panel prose updates. No JS logic changes.

- [ ] **Step 1: Relabel xwOBA in Section A (Path A model output)**

In `index.html`, find the Section A xwOBA label block (around line 1023):
```html
                        <span style="font-size:0.82em;color:#6e7681;">xwOBA per pitch</span>
                        <span title="Per-pitch expected xwOBA. Values 0.04–0.15 are typical. NOT comparable to season xwOBA (~0.310)." style="...">?</span>
```

Change label and tooltip:
```html
                        <span style="font-size:0.82em;color:#6e7681;">Expected damage (per-pitch)</span>
                        <span title="Expected damage from this pitch alone, calibrated against actual MLB outcomes. Values 0.04–0.15 are typical per-pitch — much lower than season xwOBA (~0.310) because most individual pitches don't generate damage. Higher = more dangerous." style="cursor:help;font-size:0.72em;color:#484f58;border:1px solid #484f58;border-radius:50%;width:13px;height:13px;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;">?</span>
```

- [ ] **Step 2: Relabel xwOBA in Verdict section (Section V)**

Find the Verdict section xwOBA label (around line 1171):
```html
                        <span style="font-size:0.8em;color:#6e7681;">xwOBA per pitch</span>
                        <span title="Per-pitch expected xwOBA. Values 0.04–0.15 are typical. NOT comparable to season xwOBA (~0.310)." style="...">?</span>
```

Change label and tooltip:
```html
                        <span style="font-size:0.8em;color:#6e7681;">Expected damage (per-pitch)</span>
                        <span title="Expected damage from this pitch alone, calibrated against actual MLB outcomes. Values 0.04–0.15 are typical per-pitch — much lower than season xwOBA (~0.310) because most individual pitches don't generate damage. Higher = more dangerous." style="cursor:help;font-size:0.72em;color:#484f58;border:1px solid #484f58;border-radius:50%;width:13px;height:13px;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;">?</span>
```

- [ ] **Step 3: Update About panel prose**

Line 226 — change `xwOBA estimate` to `expected damage estimate`:
```html
                            xwOBA estimate for the quality of contact if it's made.
```
→
```html
                            expected damage estimate for the quality of contact if it's made.
```

Line 255 — change `xwOBA regressor` to `expected-damage regressor`:
```html
                        plus a parallel xwOBA regressor for contact quality.
```
→
```html
                        plus a parallel expected-damage regressor for contact quality.
```

- [ ] **Step 4: Verify no remaining "xwOBA per pitch" occurrences**

```bash
grep -n "xwOBA per pitch" /Users/g/Documents/pitch_duel/src/demo/templates/index.html
```

Expected: no output (zero matches).

- [ ] **Step 5: Commit**

```bash
git add src/demo/templates/index.html
git commit -m "feat: relabel 'xwOBA per pitch' to 'Expected damage (per-pitch)' in UI + About panel"
```

---

## Task 6: `index.html` — ABS zone frontend

**Files:**
- Modify: `src/demo/templates/index.html`

This is the main frontend task. Four sub-changes:
(a) Add state variables `_absZone_demo` and `_absZone_atbat`
(b) Update `loadHitterCard()` to store zone and redraw pickers
(c) Update `buildPickerZoneSvg()` to accept 7th `absZone` parameter
(d) Update `buildZoneSvg()` to accept `opts.absZone`
(e) Update all 4 affected call sites

### 6a. Add state variables

- [ ] **Step 1: Add module-level zone state variables**

Find the Tab 2 state variables block (around line 1191):
```js
        let ab_pitches = [];      // raw pitch inputs
        let ab_results = [];      // prediction results per pitch
        let ab_balls = 0;
```

Insert two new lines just before:
```js
        // Per-tab ABS strike zone (null = use rulebook fallback)
        let _absZone_demo  = null;
        let _absZone_atbat = null;
```

### 6b. Update `loadHitterCard()`

- [ ] **Step 2: Update `loadHitterCard()` to store zone and trigger redraws**

Replace the entire `loadHitterCard` function (lines 552–580) with:

```js
        async function loadHitterCard(mode) {
            const sel = document.getElementById(`hitter_${mode}`);
            if (!sel) return;
            const name = sel.value;
            if (!name || name === 'Loading...') return;
            const card = document.getElementById(`hitter_card_${mode}`);
            const nameEl = document.getElementById(`hitter_card_${mode}_name`);
            const thinEl = document.getElementById(`hitter_card_${mode}_thin`);
            const statsEl = document.getElementById(`hitter_card_${mode}_stats`);
            try {
                const res = await fetch(`/api/hitter_profile?name=${encodeURIComponent(name)}`);
                if (!res.ok) return;
                const p = await res.json();
                nameEl.textContent = p.player_name;
                thinEl.style.display = p.is_thin_sample ? 'inline-block' : 'none';
                const stats = [
                    { v: (p.swing_rate*100).toFixed(0)+'%',    l: 'Swing' },
                    { v: (p.chase_rate*100).toFixed(0)+'%',    l: 'Chase' },
                    { v: (p.contact_rate*100).toFixed(0)+'%',  l: 'Contact' },
                    { v: (p.hard_hit_rate*100).toFixed(0)+'%', l: 'Hard Hit' },
                    { v: (p.whiff_rate*100).toFixed(0)+'%',    l: 'Whiff' },
                    { v: p.sample_size > 0 ? p.sample_size.toLocaleString() : '—', l: 'Pitches' },
                ];
                statsEl.innerHTML = stats.map(s =>
                    `<div class="hitter-card-stat"><div class="v">${s.v}</div><div class="l">${s.l}</div></div>`
                ).join('');
                card.classList.add('loaded');

                // Build ABS zone object with label for SVG display
                const az = p.abs_zone ? { ...p.abs_zone } : null;
                if (az) az.label = p.height_display ? `ABS zone — ${p.height_display}` : 'ABS zone';

                if (mode === 'demo') {
                    _absZone_demo = az;
                    syncDemoZone();  // Redraw Tab 1 picker with new zone
                }
                if (mode === 'atbat') {
                    _absZone_atbat = az;
                    // Redraw Tab 2 picker with new zone
                    const hitter = document.getElementById('hitter_atbat').value;
                    const pThrows = document.getElementById('p_throws_atbat').value;
                    const side = getEffectiveStand(hitter, pThrows);
                    const sw = (HITTER_STAND[hitter] === 'S');
                    document.getElementById('ab_picker_svg').innerHTML =
                        buildPickerZoneSvg(400, 200, null, null, side, sw, _absZone_atbat);
                    attachAbPickerClick();
                }
            } catch (e) { console.error('hitter card error:', e); }
        }
```

### 6c. Update `buildPickerZoneSvg()`

- [ ] **Step 3: Update `buildPickerZoneSvg()` to accept `absZone` and use dynamic bounds**

Replace the entire `buildPickerZoneSvg` function (lines 755–802) with:

```js
        function buildPickerZoneSvg(w, h, markerX, markerZ, batterSide, isSwitch, absZone) {
            // Zone is drawn based on h (height). Extra width provides space for batter silhouettes.
            const pad = (w - h) / 2;  // horizontal padding on each side
            // Offset pitch coords into the centered zone area
            function pToSvg(px, pz) {
                const p = pitchToSvg(px, pz, h);
                return { x: p.x + pad, y: p.y };
            }
            function fToPx(ft) { return ftToPx(ft, h); }

            // Zone bounds: use ABS zone if provided, otherwise rulebook
            const zXMin = absZone ? absZone.left_ft   : SZ_X_MIN;
            const zXMax = absZone ? absZone.right_ft  : SZ_X_MAX;
            const zZMin = absZone ? absZone.bottom_ft : SZ_Z_MIN;
            const zZMax = absZone ? absZone.top_ft    : SZ_Z_MAX;

            const zTL = pToSvg(zXMin, zZMax), zBR = pToSvg(zXMax, zZMin);
            const cTL = pToSvg(zXMin - CHASE_BORDER, zZMax + CHASE_BORDER);
            const cBR = pToSvg(zXMax + CHASE_BORDER, zZMin - CHASE_BORDER);

            const zoneLabel = absZone
                ? (absZone.label || 'ABS zone')
                : 'Rulebook zone (height unknown)';

            let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="background:#0d1117;border:1px solid #30363d;border-radius:4px;">`;
            svg += `<text x="4" y="10" font-size="8" font-family="Courier New" fill="#6e7681">${zoneLabel}</text>`;
            svg += `<rect x="${cTL.x}" y="${cTL.y}" width="${cBR.x-cTL.x}" height="${cBR.y-cTL.y}" fill="none" stroke="#30363d" stroke-width="1" stroke-dasharray="4,4"/>`;
            svg += `<rect x="${zTL.x}" y="${zTL.y}" width="${zBR.x-zTL.x}" height="${zBR.y-zTL.y}" fill="none" stroke="#8b949e" stroke-width="2"/>`;
            const tw = (zBR.x-zTL.x)/3, th = (zBR.y-zTL.y)/3;
            for (let i = 1; i < 3; i++) {
                svg += `<line x1="${zTL.x+tw*i}" y1="${zTL.y}" x2="${zTL.x+tw*i}" y2="${zTL.y+(zBR.y-zTL.y)}" stroke="#21262d" stroke-width="1"/>`;
                svg += `<line x1="${zTL.x}" y1="${zTL.y+th*i}" x2="${zTL.x+(zBR.x-zTL.x)}" y2="${zTL.y+th*i}" stroke="#21262d" stroke-width="1"/>`;
            }
            const hp = pToSvg(0, 0.9), hpW = fToPx(17/12);
            svg += `<polygon points="${hp.x},${hp.y+6} ${hp.x-hpW/2},${hp.y} ${hp.x-hpW/2},${hp.y-6} ${hp.x+hpW/2},${hp.y-6} ${hp.x+hpW/2},${hp.y}" fill="none" stroke="#484f58" stroke-width="1"/>`;

            // Batter silhouette
            if (batterSide) {
                const scale = h / 220;  // normalize to base size
                // RHH on right side, LHH on left side — use full width for positioning
                const bx = batterSide === 'R' ? w - 18 * scale : 18 * scale;
                const by = hp.y - 2;
                svg += batterSilhouette(bx, by, batterSide, scale);
                // Batting side label
                const labelX = batterSide === 'R' ? w - 8 : 8;
                const anchor = batterSide === 'R' ? 'end' : 'start';
                const label = isSwitch ? `SW(${batterSide})` : `${batterSide}HH`;
                svg += `<text x="${labelX}" y="14" text-anchor="${anchor}" fill="${isSwitch ? '#d29922' : '#484f58'}" font-size="${10*scale}px" font-family="Courier New">${label}</text>`;
            }

            if (markerX !== null && markerZ !== null) {
                const pos = pToSvg(markerX, markerZ);
                svg += `<circle cx="${pos.x}" cy="${pos.y}" r="8" fill="#58a6ff" fill-opacity="0.9" stroke="#fff" stroke-width="2"/>`;
                svg += `<line x1="${pos.x-12}" y1="${pos.y}" x2="${pos.x+12}" y2="${pos.y}" stroke="#58a6ff" stroke-width="1" opacity="0.5"/>`;
                svg += `<line x1="${pos.x}" y1="${pos.y-12}" x2="${pos.x}" y2="${pos.y+12}" stroke="#58a6ff" stroke-width="1" opacity="0.5"/>`;
            }
            svg += '</svg>';
            return svg;
        }
```

### 6d. Update `buildZoneSvg()`

- [ ] **Step 4: Update `buildZoneSvg()` to accept `opts.absZone`**

Replace the entire `buildZoneSvg` function (lines 694–734) with:

```js
        function buildZoneSvg(pitches, sz, opts = {}) {
            const showNumbers = opts.showNumbers !== false;
            const filters = opts.filters || {};
            const absZone = opts.absZone || null;
            const r = sz >= 400 ? 14 : (sz >= 250 ? 10 : 8);
            const fs = sz >= 400 ? 11 : (sz >= 250 ? 8 : 7);

            // Zone bounds: use ABS zone if provided, otherwise rulebook
            const zXMin = absZone ? absZone.left_ft   : SZ_X_MIN;
            const zXMax = absZone ? absZone.right_ft  : SZ_X_MAX;
            const zZMin = absZone ? absZone.bottom_ft : SZ_Z_MIN;
            const zZMax = absZone ? absZone.top_ft    : SZ_Z_MAX;

            const zTL = pitchToSvg(zXMin, zZMax, sz), zBR = pitchToSvg(zXMax, zZMin, sz);
            const cTL = pitchToSvg(zXMin - CHASE_BORDER, zZMax + CHASE_BORDER, sz);
            const cBR = pitchToSvg(zXMax + CHASE_BORDER, zZMin - CHASE_BORDER, sz);

            let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${sz}" height="${sz}" viewBox="0 0 ${sz} ${sz}" style="background:#0d1117;border:1px solid #30363d;border-radius:4px;">`;
            if (absZone) {
                const zoneLabel = absZone.label || 'ABS zone';
                svg += `<text x="4" y="10" font-size="8" font-family="Courier New" fill="#6e7681">${zoneLabel}</text>`;
            }
            svg += `<rect x="${cTL.x}" y="${cTL.y}" width="${cBR.x-cTL.x}" height="${cBR.y-cTL.y}" fill="none" stroke="#30363d" stroke-width="1" stroke-dasharray="4,4"/>`;
            svg += `<rect x="${zTL.x}" y="${zTL.y}" width="${zBR.x-zTL.x}" height="${zBR.y-zTL.y}" fill="none" stroke="#8b949e" stroke-width="2"/>`;
            const tw = (zBR.x-zTL.x)/3, th = (zBR.y-zTL.y)/3;
            for (let i = 1; i < 3; i++) {
                svg += `<line x1="${zTL.x+tw*i}" y1="${zTL.y}" x2="${zTL.x+tw*i}" y2="${zTL.y+(zBR.y-zTL.y)}" stroke="#21262d" stroke-width="1"/>`;
                svg += `<line x1="${zTL.x}" y1="${zTL.y+th*i}" x2="${zTL.x+(zBR.x-zTL.x)}" y2="${zTL.y+th*i}" stroke="#21262d" stroke-width="1"/>`;
            }
            const hp = pitchToSvg(0, 0.9, sz), hpW = ftToPx(17/12, sz);
            svg += `<polygon points="${hp.x},${hp.y+6} ${hp.x-hpW/2},${hp.y} ${hp.x-hpW/2},${hp.y-6} ${hp.x+hpW/2},${hp.y-6} ${hp.x+hpW/2},${hp.y}" fill="none" stroke="#484f58" stroke-width="1"/>`;

            const fp = pitches.filter(p => {
                if (filters.pitchTypes && filters.pitchTypes.size > 0 && !filters.pitchTypes.has(p.pitch_type)) return false;
                if (filters.outcomes && filters.outcomes.size > 0) {
                    const oc = p.event || p.description || '';
                    if (!filters.outcomes.has(oc)) return false;
                }
                return true;
            });

            for (const p of fp) {
                const pos = pitchToSvg(p.plate_x, p.plate_z, sz);
                const color = p.quality_color || dangerColor(p.p_hard || p.p_hard_contact || p.xwoba_pred || 0);
                const tip = `${p.pitch_type} #${p.pitch_num} | ${p.speed||''}mph | P(hard)=${((p.p_hard || p.p_hard_contact||p.xwoba_pred||0)).toFixed(3)}${p.description?' | '+p.description:''}`;
                svg += `<circle cx="${pos.x}" cy="${pos.y}" r="${r}" fill="${color}" fill-opacity="0.85" stroke="#0d1117" stroke-width="1.5"><title>${tip}</title></circle>`;
                if (showNumbers)
                    svg += `<text x="${pos.x}" y="${pos.y+fs/3}" text-anchor="middle" fill="#fff" font-size="${fs}" font-family="Courier New" font-weight="bold" pointer-events="none">${p.pitch_num}</text>`;
            }
            svg += '</svg>';
            return svg;
        }
```

### 6e. Update all 4 affected call sites

- [ ] **Step 5: Update Tab 1 picker call site in `syncDemoZone()`**

Find (line ~890):
```js
            container.innerHTML = buildPickerZoneSvg(400, 250, px, pz, side, sw);
```

Replace with:
```js
            container.innerHTML = buildPickerZoneSvg(400, 250, px, pz, side, sw, _absZone_demo);
```

- [ ] **Step 6: Update Tab 2 initial/reset picker in `resetAtBat()`**

Find (line ~1214):
```js
            document.getElementById('ab_picker_svg').innerHTML = buildPickerZoneSvg(400, 200, null, null, side, sw);
```

Replace with:
```js
            document.getElementById('ab_picker_svg').innerHTML = buildPickerZoneSvg(400, 200, null, null, side, sw, _absZone_atbat);
```

- [ ] **Step 7: Update Tab 2 active-pitch picker in `syncAbPicker()`**

Find (line ~1328):
```js
            container.innerHTML = buildPickerZoneSvg(400, 250, pitch.plate_x, pitch.plate_z, side, sw);
```

Replace with:
```js
            container.innerHTML = buildPickerZoneSvg(400, 250, pitch.plate_x, pitch.plate_z, side, sw, _absZone_atbat);
```

- [ ] **Step 8: Update Tab 2 results zone SVG in `updateAbZone()`**

Find (line ~1226):
```js
            document.getElementById('ab_zone_svg').innerHTML = buildZoneSvg(zonePitches, 250, {showNumbers: true});
```

Replace with:
```js
            document.getElementById('ab_zone_svg').innerHTML = buildZoneSvg(zonePitches, 250, {showNumbers: true, absZone: _absZone_atbat});
```

- [ ] **Step 9: Commit**

```bash
git add src/demo/templates/index.html
git commit -m "feat: ABS strike zone in pitch picker and result SVGs — zone resizes per hitter height"
```

---

## Task 7: Verification

- [ ] **Step 1: Run all tests to check nothing is broken**

```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -m pytest tests/ -v
```

Expected: all tests pass (at minimum the 9 abs_zone tests).

- [ ] **Step 2: Start the dev server**

```bash
cd /Users/g/Documents/pitch_duel && source venv/bin/activate && python -m src.demo.server
```

Open `http://localhost:5050` in a browser.

- [ ] **Step 3: Verify checklist**

Work through each item manually in the browser:

| Check | How to verify |
|-------|---------------|
| `compute_abs_zone(72)` → top=3.21, bottom=1.62 | Run `python -c "from src.hitters.abs_zone import compute_abs_zone; print(compute_abs_zone(72))"` |
| `compute_abs_zone(79)` → top=3.52, bottom=1.78 | Run `python -c "from src.hitters.abs_zone import compute_abs_zone; print(compute_abs_zone(79))"` |
| Judge picker zone visibly taller than Altuve | Select Judge vs Altuve in Tab 1 picker |
| "Rulebook zone (height unknown)" label for unknown hitter | Select a hitter not in the JSON (or temporarily empty the JSON for a test) |
| "Expected damage (per-pitch)" in Section A | Run a prediction in Tab 1 |
| "Expected damage (per-pitch)" in Verdict | Verify the Verdict section label |
| New tooltip text in both locations | Hover over the `?` badge |
| `predicted_xwoba_per_pitch` JSON field unchanged | Check browser network tab |
| Path B: no "nan" rows in similar pitches table | Run prediction, expand Path B section |
| AB-ending outcomes (Single, Strikeout) show correctly | Check a high-confidence hitter's similar pitch table |

- [ ] **Step 4: Final commit if any minor fixes were made**

```bash
git add -p  # review any changes
git commit -m "fix: post-verification adjustments for Task 8"
```

---

## Implementation Notes

**ABS zone coordinate system:** `plate_x` and `plate_z` in Statcast are in feet, catcher's perspective. The existing `pitchToSvg()` function maps `plate_x ∈ [-2.0, 2.0]` and `plate_z ∈ [0.5, 4.5]` to the SVG canvas — ABS zone bounds (top ≈ 3.0–3.5 ft, bottom ≈ 1.5–1.7 ft, width ±0.708 ft) fit within this space. No coordinate system changes needed.

**Why `SZ_X_MIN/SZ_X_MAX` constants are preserved:** They are still used by `classify_pitch_zone()` in `server.py` for verdict zone labels ("in_zone", "chase", "waste"). Changing those would affect model logic. The spec explicitly leaves this out of scope.

**Tab 3 game log:** `buildZoneSvg()` calls at lines 1682 and 1755 are left unchanged — no per-hitter context is available in the game log flow so they continue to use the rulebook zone (no `absZone` passed = uses `SZ_*` constants).

**Height cache atomicity:** `_save_cache()` overwrites the JSON file in a single `write_text()` call. For the demo's 12 hitters already in the cache, no network calls occur at all — the 2-second timeout is only a fallback for unknown IDs.
