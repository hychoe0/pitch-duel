# Task 8: ABS Strike Zone Display + Two UI Clarity Fixes — Design Spec

**Date:** 2026-04-27  
**Scope:** Flask demo UI only. No model retraining, no pipeline changes.  
**Time box:** 1–2 days.

---

## Overview

Three independent improvements to `src/demo/`:

| Part | Change | Files |
|------|--------|-------|
| 1 | Per-hitter ABS strike zone in all zone visualizations | `src/hitters/abs_zone.py` (new), `data/processed/hitter_heights.json` (new), `server.py`, `index.html` |
| 2 | Relabel "xwOBA per pitch" → "Expected damage (per-pitch)" | `index.html` |
| 3 | Fix Path B "nan" outcomes — show Ball/Strike/Foul instead | `server.py` |

Parts 2 and 3 are self-contained. Part 1 is the main work.

---

## Part 1 — ABS Strike Zone Display

### Background

MLB's 2026 ABS Challenge System gives every hitter a personalized strike zone based on their height:
- Top: 53.5% of height
- Bottom: 27.0% of height
- Width: exactly 17 inches (±0.708 ft), centered on the plate

This differs from both the rulebook definition and the current demo's `SZ_X_MIN = -0.85, SZ_X_MAX = 0.85, SZ_Z_MIN = 1.5, SZ_Z_MAX = 3.5` constants.

The demo has two types of zone SVGs:
- **Picker** (`buildPickerZoneSvg`) — interactive input, shown before prediction. Used in Tab 1 and Tab 2.
- **Results** (`buildZoneSvg`) — read-only display of thrown pitches colored by risk. Used in Tabs 2 and 3.

Both must show the ABS zone. When a pitcher selects Aaron Judge, the picker zone should be visibly taller than when they select José Ramírez.

### 1.1 Hitter Height Cache

**New file:** `data/processed/hitter_heights.json`

Format: `{ "player_id_str": height_inches_int, ... }`

Keys are player IDs as strings (matching the `batter` column in Statcast / profile file names).

**Seeded with 12 demo hitters:**

```json
{
  "660271": 76,
  "592450": 79,
  "596019": 70,
  "608070": 68,
  "682998": 75,
  "119534": 74,
  "605196": 71,
  "691176": 69,
  "686948": 72,
  "693307": 73,
  "665966": 71,
  "701762": 77
}
```

Heights sourced:
- 9 via `statsapi.mlb.com/api/v1/people/{id}` (API reachable, confirmed working)
- 3 hardcoded (IDs 682998/Merrill, 119534/Murakami, 605196/McGonigle map to wrong players in API — looked up from team rosters and NPB records)

Height strings from API parsed: `"6' 4\""` → `feet*12 + inches` = 76.

### 1.2 Python Helper — `src/hitters/abs_zone.py`

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
    Caches the result if the API name matches what we expect.
    Returns None if not in cache and API fails/is unreachable.
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

**Sanity checks (to run in the plan's verification step):**
- 6'0" (72 in): top = 3.21 ft, bottom = 1.62 ft ✓
- 6'4" (76 in): top = 3.39 ft, bottom = 1.71 ft ✓  
- 5'9" (69 in): top = 3.07 ft, bottom = 1.55 ft ✓

### 1.3 Server Changes — `server.py`

**`_load_hitter_profile_for_display()` → augment return dict:**

The function already has the profile (which has `player_id`). Add:

```python
from src.hitters.abs_zone import get_abs_zone, get_height_inches

# ... existing profile loading ...
pid = profile.player_id  # already resolved
h_in = get_height_inches(pid)
abs_zone = get_abs_zone(pid)

return {
    ...existing fields...,
    "height_inches": h_in,
    "height_display": f"{h_in//12}'{h_in%12}\"" if h_in else None,
    "abs_zone": abs_zone,  # None if height unknown
}
```

The fallback return (no profile found) sets both to `None`.

**`/api/predict_demo` response:** Add `abs_zone` and `hitter_height_inches` at the top level (alongside the existing `hitter_profile`). This provides the zone for the result SVG.

**`/api/hitter_profile` endpoint:** Already calls `_load_hitter_profile_for_display()`, so `abs_zone` will be in the response automatically once the helper is updated. This drives the picker update when a hitter is selected.

### 1.4 Frontend Changes — `index.html`

**Module-level zone state (one per tab):**

```js
let _absZone_demo  = null;  // Tab 1 current hitter's ABS zone
let _absZone_atbat = null;  // Tab 2 current hitter's ABS zone
```

**`loadHitterCard(mode)` — store ABS zone after fetch:**

```js
const p = await res.json();
// existing display code ...
if (mode === 'demo')  _absZone_demo  = p.abs_zone || null;
if (mode === 'atbat') _absZone_atbat = p.abs_zone || null;
// redraw picker with new zone
updatePitchMarker();  // Tab 1 already calls this on its own
// Tab 2: explicitly redraw picker
if (mode === 'atbat') {
    const side = ...;
    document.getElementById('ab_picker_svg').innerHTML =
        buildPickerZoneSvg(400, 200, null, null, side, sw, _absZone_atbat);
}
```

**`buildPickerZoneSvg(w, h, markerX, markerZ, batterSide, isSwitch, absZone)`:**

Add `absZone` as optional 7th parameter. When provided:
- Replace `SZ_X_MIN/SZ_X_MAX/SZ_Z_MIN/SZ_Z_MAX` with `absZone.left_ft/right_ft/bottom_ft/top_ft` for the zone rectangle
- Chase zone border extends ±0.5 ft from the ABS zone bounds
- Add a text label at the top of the SVG: `"ABS zone — {height}"` (e.g., `"ABS zone — 6'4\""`) in small gray text
- When `absZone` is null: draw rulebook zone with a label `"Rulebook zone (height unknown)"`

**`buildZoneSvg(pitches, sz, opts)`:**

Add `opts.absZone` support. Same coordinate replacement as above. Label rendered as SVG text at the top edge.

**All call sites:** Updated to pass the relevant zone variable:
- Line 890 (Tab 1 picker): `buildPickerZoneSvg(400, 250, px, pz, side, sw, _absZone_demo)`
- Line 1214 (Tab 2 initial): `buildPickerZoneSvg(400, 200, null, null, side, sw, _absZone_atbat)`
- Line 1328 (Tab 2 pitch editor): `buildPickerZoneSvg(400, 250, px, pz, side, sw, _absZone_atbat)`
- Line 1226 (Tab 2 results): `buildZoneSvg(zonePitches, 250, {showNumbers: true, absZone: _absZone_atbat})`
- Lines 1682, 1776 (Tab 3 game log): Game log doesn't have a per-hitter context, so no ABS zone — leave as-is with rulebook zone.
- Line 1755 (Tab 3 per-PA): Same — no hitter context, leave as-is.

**Zone label position:** Small `<text>` element at the top-left of the SVG, 8px font, `#6e7681` color, inside the SVG element. Not a separate div (avoids layout shifts).

### 1.5 Coordinate system note

The SVG canvas maps plate_x ∈ [-2.0, 2.0] and plate_z ∈ [0.5, 4.5] into a square. The existing `pitchToSvg()` function handles this. Both rulebook and ABS zone coordinates are valid in this coordinate space — just different rectangle corners. The global `SZ_X_MIN/SZ_X_MAX` constants are left unchanged (they're also used by `classify_pitch_zone()` in server.py for the verdict label, which is not being changed in this task).

---

## Part 2 — Relabel "xwOBA per pitch"

**Two occurrences in `index.html`, both changed identically:**

| Location | Old label | New label |
|----------|-----------|-----------|
| Section A (Path A model, ~line 1023) | `xwOBA per pitch` | `Expected damage (per-pitch)` |
| Section V (Verdict, ~line 1171) | `xwOBA per pitch` | `Expected damage (per-pitch)` |

**Tooltip replacement (both occurrences):**

Old: `"Per-pitch expected xwOBA. Values 0.04–0.15 are typical. NOT comparable to season xwOBA (~0.310)."`

New: `"Expected damage from this pitch alone, calibrated against actual MLB outcomes. Values 0.04–0.15 are typical per-pitch — much lower than season xwOBA (~0.310) because most individual pitches don't generate damage. Higher = more dangerous."`

**About panel text (~lines 226, 255):** Update the two references to "xwOBA estimate" / "xwOBA regressor" to use "expected damage" in the user-facing prose. Technical references in code comments are untouched.

**No JSON field name changes.** `predicted_xwoba_per_pitch` stays as-is in API responses.

---

## Part 3 — Path B Outcome Display Fix

### Root cause

In `predict_combined.py` line 156:
```python
ev = str(row.get("events", "") or "").strip()
```
`float('nan') or ""` evaluates to `float('nan')` in Python (NaN is truthy), so `str(float('nan'))` = `"nan"`. This string is truthy in JavaScript, so `p.events || p.description` always returns `"nan"` instead of falling through to `description`.

`events` is NaN on most pitches by design — it only populates on the final pitch of a plate appearance. NaN is not an error.

### Fix location

`server.py` — post-process `mr.top_similar_pitches` before serializing. No changes to `predict_combined.py`.

### Implementation

Add module-level label dicts and a helper to `server.py`:

```python
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

In the `/api/predict_demo` response, replace:
```python
"top_pitches": mr.top_similar_pitches,
```
with:
```python
"top_pitches": [
    {**p, "outcome": _resolve_pitch_outcome(p)}
    for p in mr.top_similar_pitches
],
```

In `index.html`, replace the outcome resolution in the Path B table:
```js
// OLD:
const outcome = p.events || p.description;
// NEW:
const outcome = p.outcome || p.events || p.description || 'Unknown';
```
The `p.outcome` field from the server is always a clean string. The fallback chain is a belt-and-suspenders safety net.

---

## File Inventory

| File | Action | What changes |
|------|--------|--------------|
| `src/hitters/abs_zone.py` | **Create** | `compute_abs_zone()`, `get_height_inches()`, `get_abs_zone()` |
| `data/processed/hitter_heights.json` | **Create** | 12 demo hitters: player_id → height_inches |
| `src/demo/server.py` | **Modify** | (a) Import + use `abs_zone` in `_load_hitter_profile_for_display()`; (b) add `abs_zone`/`hitter_height_inches` to prediction response; (c) add `_EVENT_LABELS`, `_DESCRIPTION_LABELS`, `_resolve_pitch_outcome()`; (d) post-process `top_pitches` in prediction response |
| `src/demo/templates/index.html` | **Modify** | (a) Add `_absZone_demo`/`_absZone_atbat` state vars; (b) store ABS zone in `loadHitterCard()`; (c) update `buildPickerZoneSvg()` and `buildZoneSvg()` signatures; (d) update all 6 call sites; (e) relabel xwOBA → "Expected damage" in 2 places + 2 tooltips + About panel |

No changes to `predict.py`, `predict_combined.py`, `profiles.py`, or any model files.

---

## Verification Checklist

- [ ] `compute_abs_zone(72)` → top=3.21, bottom=1.62
- [ ] `compute_abs_zone(79)` → top=3.52, bottom=1.78  
- [ ] Selecting Judge (6'7"): picker zone noticeably taller than Ramírez (5'8")
- [ ] Selecting a hitter without a cached height: fallback "Rulebook zone (height unknown)" label shows
- [ ] "Expected damage (per-pitch)" appears in both Section A and Verdict; "xwOBA per pitch" gone
- [ ] Tooltip shows new explanation text
- [ ] JSON response `predicted_xwoba_per_pitch` field name unchanged
- [ ] Path B similar pitches show "Ball", "Called Strike", "Foul", etc. — zero "nan" rows
- [ ] AB-ending outcomes (Single, Strikeout) still show correctly

---

## Out of Scope (deferred)

- ABS zone as model feature (changes `predict.py` + retraining)
- Updating `classify_pitch_zone()` in `server.py` to use ABS bounds (affects verdict zone label)
- Game log (Tab 3) per-PA zone — no per-hitter context is available in that flow
- Heights for hitters added to the demo after this task
