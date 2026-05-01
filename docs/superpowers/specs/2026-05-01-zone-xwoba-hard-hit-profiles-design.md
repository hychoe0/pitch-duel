# Design: Zone-Specific xwOBA and Hard-Hit Rates in HitterProfile

**Date:** 2026-05-01  
**Scope:** Demo hitters only (15 profiles) — validate before full rebuild  
**Approach:** Option A — minimal, spec-compliant, no changes to existing zone logic

---

## Problem

`HitterProfile` stores zone-level swing and whiff rates but has no zone-level damage rates. This means the model knows where a hitter chases or makes contact, but not how hard they hit the ball (or how dangerous the contact is) by location. This is the "Location vs. X" gap.

---

## Goal

Add `zone_xwoba_rates` and `zone_hard_hit_rates` to `HitterProfile`, wire them into `build_profile()`, expose them via a new resolver function, and validate the implementation on the 15 demo hitters before any full rebuild.

---

## Known Issue (not fixed in this prompt)

Existing `zone_swing_rates` and `zone_whiff_rates` use **integer** dict keys in memory but become **string** keys after JSON serialization/deserialization. This means cached profiles silently fall back to `LEAGUE_AVG` values for every zone lookup in `profile_to_feature_dict()`. The new zone damage fields use **string keys throughout** to avoid this problem. The existing fields should be fixed in a dedicated follow-up.

---

## Data Model Changes (`src/hitters/profiles.py`)

### New constant

```python
MIN_ZONE_BATTED_BALLS = 10  # tunable threshold — adjust as model accuracy improves
```

Placed alongside `MIN_ZONE_PITCHES` and `MIN_FAMILY_PITCHES`.

### Updated `LEAGUE_AVG`

```python
LEAGUE_AVG = {
    "swing_rate":    0.47,
    "chase_rate":    0.30,
    "contact_rate":  0.78,
    "hard_hit_rate": 0.38,
    "whiff_rate":    0.25,
    "zone_xwoba":    0.380,  # MLB avg xwOBA on batted balls (pitch-clock era)
}
```

### New `HitterProfile` fields

```python
zone_xwoba_rates:    dict = field(default_factory=dict)  # str(zone_id) -> float
zone_hard_hit_rates: dict = field(default_factory=dict)  # str(zone_id) -> float
```

**Key format:** string zone IDs (`"1"` through `"9"`, `"11"` through `"14"`), consistent with JSON round-trip. Existing `zone_swing_rates` / `zone_whiff_rates` (integer keys) are not touched.

**Backward compatibility:** `field(default_factory=dict)` means `HitterProfile(**legacy_json_dict)` succeeds when the JSON lacks these keys — they default to `{}`, which the resolver treats as thin-sample and falls back gracefully.

---

## Compute Helpers

Both follow the same pattern as `compute_zone_swing_rates` / `compute_zone_whiff_rates`.

### `compute_zone_xwoba_rates(df, w) -> dict`

- Filter rows: `description` ∈ `{hit_into_play, hit_into_play_no_out, hit_into_play_score}` **and** `estimated_woba_using_speedangle` is not null
- Group by zone: `str(int(zone))` for dict keys
- Compute weighted mean of `estimated_woba_using_speedangle` per zone
- **Omit** zone if weighted batted-ball count < `MIN_ZONE_BATTED_BALLS` (do not fill with a fallback value — the resolver handles fallback at lookup time)

### `compute_zone_hard_hit_rates(df, w) -> dict`

- Same batted-ball filter, same zone grouping
- Weighted rate of `launch_speed >= 95` per zone
- Same omit-on-thin behavior

Both wired into `build_profile()`. When `n == 0` (no data for player), both return `{}`.

---

## Resolver Function

```python
def resolve_hitter_features_for_pitch(
    profile: HitterProfile,
    zone: int,
) -> dict:
```

Returns 7 features — the 5 existing global rates plus two new zone-damage rates:

| Key | Value | Fallback when zone missing |
|-----|-------|---------------------------|
| `hitter_swing_rate` | `profile.swing_rate` | — |
| `hitter_chase_rate` | `profile.chase_rate` | — |
| `hitter_contact_rate` | `profile.contact_rate` | — |
| `hitter_hard_hit_rate` | `profile.hard_hit_rate` | — |
| `hitter_whiff_rate` | `profile.whiff_rate` | — |
| `hitter_zone_xwoba` | `profile.zone_xwoba_rates[str(zone)]` | `LEAGUE_AVG["zone_xwoba"]` (0.380) |
| `hitter_zone_hard_hit_rate` | `profile.zone_hard_hit_rates[str(zone)]` | `profile.hard_hit_rate` |

**Fallback rationale:**
- `hitter_zone_xwoba` falls back to league average because no hitter-global xwOBA field is stored in the profile (adding one is out of scope here). Range: 0.0–2.0+.
- `hitter_zone_hard_hit_rate` falls back to the hitter's own `hard_hit_rate` because the player's global rate is a better prior than league average for thin zones (Judge's 0.61 is a much better guess than the league's 0.38). Range: 0.0–1.0.

This function is added **alongside** `profile_to_feature_dict()` — it does not replace it. Wiring into `predict.py` and `merge_profiles_into_df()` is deferred to Prompt 2.

---

## CLI: `--rebuild-demo`

Module-level constant:

```python
DEMO_HITTERS: dict[str, int] = {
    "Judge, Aaron":           592450,
    "Betts, Mookie":          605141,
    "Kiner-Falefa, Isiah":    643396,
    "Ohtani, Shohei":         660271,
    "Raleigh, Cal":           663728,
    "Soto, Juan":             665742,
    "Witt, Bobby":            677951,
    "Duran, Jarren":          680776,
    "Carroll, Corbin":        682998,
    "Henderson, Gunnar":      683002,
    "Volpe, Anthony":         683011,
    "Caissie, Owen":          683357,
    "Winn, Masyn":            691026,
    "Langford, Wyatt":        694671,
    "Lee, Jung Hoo":          808982,
}
```

When `--rebuild-demo` is passed:
1. Load `data/processed/statcast_processed.parquet`
2. Populate `_NAME_CACHE` for just these 15 IDs
3. For each: force-build (skip any cached JSON), save to `data/processed/profiles/{player_id}.json`
4. Print per-hitter: name, sample_size, `len(zone_xwoba_rates)`, xwOBA for zone 5, xwOBA for zone 14

Existing `--name` / `--names` / default behavior unchanged.

---

## JSON Output Shape

```json
{
  "player_id": 592450,
  "player_name": "Judge, Aaron",
  "zone_swing_rates": {"1": 0.47, ...},
  "zone_whiff_rates":  {"1": 0.18, ...},
  "zone_xwoba_rates":  {"1": 0.42, "2": 0.51, "5": 0.68, ...},
  "zone_hard_hit_rates": {"1": 0.62, "5": 0.71, ...},
  "family_swing_rates": {...},
  "family_whiff_rates": {...}
}
```

Zones with < 10 weighted batted balls are absent from the new dicts (not null, just absent).

---

## Unit Tests (`tests/test_profiles.py`)

| Test | Description |
|------|-------------|
| `test_judge_zone_xwoba_heart_vs_chase` | Load rebuilt 592450.json, assert `zone_xwoba_rates["5"] > 0.40` and zone 14 < zone 5. Skipped if JSON not present. |
| `test_thin_zone_omitted_from_xwoba` | Synthetic df with <10 weighted batted balls in zone 1, ≥10 in zone 5. Assert zone 1 absent, zone 5 present in result. |
| `test_resolver_fallbacks` | Profile with empty zone dicts. Assert `hitter_zone_xwoba ≈ 0.380` and `hitter_zone_hard_hit_rate == profile.hard_hit_rate`. |
| `test_legacy_profile_loads` | Manually build dict without new keys, call `HitterProfile(**data)`, assert success and both fields default to `{}`. |

---

## What Does NOT Change

- `profile_to_feature_dict()` — signature and behavior unchanged
- `merge_profiles_into_df()` — unchanged (Prompt 2)
- `predict.py` — unchanged (Prompt 2)
- `preprocess.py` — unchanged
- `zone_swing_rates` / `zone_whiff_rates` logic — unchanged (int key bug deferred)
- `_blend_profile()` — does not blend new zone damage fields (thin zones are omitted, not blended)
- All existing profile JSONs except the 15 demo hitters
- Any model training code

---

## Verification Checklist

1. `python -m src.hitters.profiles --rebuild-demo` completes for all 15 hitters with non-empty `zone_xwoba_rates`
2. `data/processed/profiles/592450.json` has both new fields; zones 4/5/6 show xwOBA > 0.40
3. `data/processed/profiles/808982.json` (Lee Jung Hoo, thin sample) builds successfully with fewer zone keys
4. `pytest tests/test_profiles.py` — all new and existing tests pass
