# Zone xwOBA and Hard-Hit Rate Profiles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `zone_xwoba_rates` and `zone_hard_hit_rates` to `HitterProfile`, rebuild the 15 demo hitters, and add a `resolve_hitter_features_for_pitch` lookup function — validating the new fields before any full rebuild.

**Architecture:** String-keyed zone dicts (omit-on-thin, not fill-on-thin) stored alongside existing zone_swing/whiff fields. A new resolver function returns 7 features (5 global + 2 zone-damage) for use in Prompt 2. Existing `profile_to_feature_dict`, `merge_profiles_into_df`, and `predict.py` are unchanged.

**Tech Stack:** Python 3.11+, pandas, numpy, XGBoost (model unchanged), pytest, pybaseball (name cache only). Always run commands with `./venv/bin/python` or `./venv/bin/pytest`.

---

## File Map

| File | Change |
|------|--------|
| `src/hitters/profiles.py` | Add constant, LEAGUE_AVG entry, 2 dataclass fields, 2 compute helpers, resolver function, wire into `build_profile()`, add `--rebuild-demo` CLI |
| `tests/test_profiles.py` | Create: backward-compat, resolver fallbacks, thin-zone omission, Judge zone validation |

---

### Task 1: Add constant, LEAGUE_AVG entry, and HitterProfile fields

**Files:**
- Modify: `src/hitters/profiles.py:21-147`

- [ ] **Step 1: Add `MIN_ZONE_BATTED_BALLS` constant**

In `src/hitters/profiles.py`, after line 74 (`MIN_FAMILY_PITCHES = 30`), add:

```python
MIN_ZONE_BATTED_BALLS = 10   # tunable — adjust as model accuracy improves
```

- [ ] **Step 2: Add `zone_xwoba` to `LEAGUE_AVG`**

The `LEAGUE_AVG` dict currently ends at `"whiff_rate": 0.25`. Add one entry:

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

- [ ] **Step 3: Add two new fields to `HitterProfile`**

In `src/hitters/profiles.py`, inside the `HitterProfile` dataclass, add the two new fields immediately after `zone_whiff_rates` (which is currently around line 142):

```python
    zone_swing_rates: dict = field(default_factory=dict)           # zone_int -> float
    zone_whiff_rates: dict = field(default_factory=dict)           # zone_int -> float
    zone_xwoba_rates: dict = field(default_factory=dict)           # str(zone_id) -> float
    zone_hard_hit_rates: dict = field(default_factory=dict)        # str(zone_id) -> float
    family_swing_rates: dict = field(default_factory=dict)         # family_str -> float
```

Note: string keys (`"1"` through `"14"`) — these new fields use str keys throughout to be safe across JSON round-trips. The existing `zone_swing_rates` / `zone_whiff_rates` (int keys) are not touched.

- [ ] **Step 4: Commit**

```bash
git add src/hitters/profiles.py
git commit -m "feat: add zone_xwoba_rates and zone_hard_hit_rates fields to HitterProfile"
```

---

### Task 2: Create test file + backward compatibility test

**Files:**
- Create: `tests/test_profiles.py`

- [ ] **Step 1: Create `tests/test_profiles.py` with backward-compat test**

```python
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
```

- [ ] **Step 2: Run test — expect PASS**

```bash
./venv/bin/pytest tests/test_profiles.py::test_legacy_profile_loads_without_new_fields -v
```

Expected: `PASSED` (the `field(default_factory=dict)` defaults fill in the missing keys).

- [ ] **Step 3: Commit**

```bash
git add tests/test_profiles.py
git commit -m "test: backward-compat — legacy profiles load without zone damage fields"
```

---

### Task 3: Implement `resolve_hitter_features_for_pitch` (test-first)

**Files:**
- Modify: `tests/test_profiles.py` (add 2 tests)
- Modify: `src/hitters/profiles.py` (add resolver function after `profile_to_feature_dict`)

- [ ] **Step 1: Add resolver tests to `tests/test_profiles.py`**

Append after the backward-compat section:

```python
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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
./venv/bin/pytest tests/test_profiles.py -k "resolver" -v
```

Expected: `ImportError` or `AttributeError` — `resolve_hitter_features_for_pitch` does not exist yet.

- [ ] **Step 3: Implement `resolve_hitter_features_for_pitch` in `profiles.py`**

Add this function immediately after `profile_to_feature_dict` (after the closing of that function, around line 538):

```python
def resolve_hitter_features_for_pitch(
    profile: "HitterProfile",
    zone: int,
) -> dict:
    """
    Returns the 5 global rate features plus zone-specific damage rates.
    Falls back to LEAGUE_AVG["zone_xwoba"] for xwOBA (no hitter-global stored),
    and to profile.hard_hit_rate for zone hard-hit rate (better prior than league avg).
    """
    return {
        "hitter_swing_rate":          profile.swing_rate,
        "hitter_chase_rate":          profile.chase_rate,
        "hitter_contact_rate":        profile.contact_rate,
        "hitter_hard_hit_rate":       profile.hard_hit_rate,
        "hitter_whiff_rate":          profile.whiff_rate,
        "hitter_zone_xwoba":          profile.zone_xwoba_rates.get(
                                          str(zone), LEAGUE_AVG["zone_xwoba"]
                                      ),
        "hitter_zone_hard_hit_rate":  profile.zone_hard_hit_rates.get(
                                          str(zone), profile.hard_hit_rate
                                      ),
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
./venv/bin/pytest tests/test_profiles.py -k "resolver" -v
```

Expected: all 4 resolver tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/hitters/profiles.py tests/test_profiles.py
git commit -m "feat: add resolve_hitter_features_for_pitch with zone-damage fallbacks"
```

---

### Task 4: Implement `compute_zone_xwoba_rates` (test-first)

**Files:**
- Modify: `tests/test_profiles.py` (add 2 tests)
- Modify: `src/hitters/profiles.py` (add compute function after `compute_family_whiff_rates`)

- [ ] **Step 1: Add xwOBA compute tests to `tests/test_profiles.py`**

Append after the resolver tests:

```python
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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
./venv/bin/pytest tests/test_profiles.py -k "zone_xwoba" -v
```

Expected: `ImportError` — `compute_zone_xwoba_rates` does not exist yet.

- [ ] **Step 3: Implement `compute_zone_xwoba_rates` in `profiles.py`**

Add this function after `compute_family_whiff_rates` (around line 325):

```python
def compute_zone_xwoba_rates(df: pd.DataFrame, w: np.ndarray) -> dict:
    """
    Weighted mean estimated_woba_using_speedangle per Statcast zone, batted balls only.
    Zones with weighted batted-ball count < MIN_ZONE_BATTED_BALLS are omitted.
    Returns str(zone_id) -> float.
    """
    batted_mask = (
        df["description"].isin({"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"})
        & df["estimated_woba_using_speedangle"].notna()
        & df["zone"].notna()
    )
    valid = df[batted_mask].copy()
    valid_w = w[df.index.get_indexer(valid.index)]
    result = {}
    for z in ALL_ZONES:
        mask = valid["zone"] == z
        grp = valid[mask]
        grp_w = valid_w[valid.index.get_indexer(grp.index)]
        if grp_w.sum() >= MIN_ZONE_BATTED_BALLS:
            result[str(z)] = float(np.average(
                grp["estimated_woba_using_speedangle"].values, weights=grp_w
            ))
    return result
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
./venv/bin/pytest tests/test_profiles.py -k "zone_xwoba" -v
```

Expected: all 3 `zone_xwoba` tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/hitters/profiles.py tests/test_profiles.py
git commit -m "feat: add compute_zone_xwoba_rates — omit thin zones, str keys"
```

---

### Task 5: Implement `compute_zone_hard_hit_rates` (test-first)

**Files:**
- Modify: `tests/test_profiles.py`
- Modify: `src/hitters/profiles.py` (add after `compute_zone_xwoba_rates`)

- [ ] **Step 1: Add hard-hit compute tests to `tests/test_profiles.py`**

Append after the xwOBA compute tests:

```python
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
    assert result["5"] == pytest.approx(1.0)  # _make_batted_ball_df uses launch_speed=98


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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
./venv/bin/pytest tests/test_profiles.py -k "zone_hard_hit" -v
```

Expected: `ImportError` — `compute_zone_hard_hit_rates` does not exist yet.

- [ ] **Step 3: Implement `compute_zone_hard_hit_rates` in `profiles.py`**

Add immediately after `compute_zone_xwoba_rates`:

```python
def compute_zone_hard_hit_rates(df: pd.DataFrame, w: np.ndarray) -> dict:
    """
    Weighted rate of launch_speed >= 95 per Statcast zone, batted balls only.
    Zones with weighted batted-ball count < MIN_ZONE_BATTED_BALLS are omitted.
    Returns str(zone_id) -> float.
    """
    batted_mask = (
        df["description"].isin({"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"})
        & df["zone"].notna()
    )
    valid = df[batted_mask].copy()
    valid_w = w[df.index.get_indexer(valid.index)]
    result = {}
    for z in ALL_ZONES:
        mask = valid["zone"] == z
        grp = valid[mask]
        grp_w = valid_w[valid.index.get_indexer(grp.index)]
        if grp_w.sum() >= MIN_ZONE_BATTED_BALLS:
            hard = (grp["launch_speed"].fillna(0) >= 95).values
            result[str(z)] = float((grp_w * hard).sum() / grp_w.sum())
    return result
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
./venv/bin/pytest tests/test_profiles.py -k "zone_hard_hit" -v
```

Expected: all 3 `zone_hard_hit` tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/hitters/profiles.py tests/test_profiles.py
git commit -m "feat: add compute_zone_hard_hit_rates — omit thin zones, str keys"
```

---

### Task 6: Wire both compute functions into `build_profile()` + regression check

**Files:**
- Modify: `src/hitters/profiles.py:472-489` (the `HitterProfile(...)` constructor call in `build_profile`)

- [ ] **Step 1: Add the two new fields to the `HitterProfile(...)` call in `build_profile()`**

Locate the constructor call in `build_profile()`. It currently has `zone_whiff_rates=...` and then `family_swing_rates=...`. Insert the two new lines between them:

```python
    profile = HitterProfile(
        player_id=player_id,
        player_name=name,
        stand=stand,
        swing_rate=swing_rate,
        chase_rate=chase_rate,
        contact_rate=contact_rate,
        hard_hit_rate=hard_hit_rate,
        whiff_rate=whiff_rate,
        swing_rate_by_count=compute_swing_rate_by_count(sub, w) if n > 0 else {},
        contact_rate_by_pitch_type=compute_contact_rate_by_pitch_type(sub, w) if n > 0 else {},
        zone_swing_rates=compute_zone_swing_rates(sub, w, swing_rate) if n > 0 else {z: swing_rate for z in ALL_ZONES},
        zone_whiff_rates=compute_zone_whiff_rates(sub, w, whiff_rate) if n > 0 else {z: whiff_rate for z in ALL_ZONES},
        zone_xwoba_rates=compute_zone_xwoba_rates(sub, w) if n > 0 else {},
        zone_hard_hit_rates=compute_zone_hard_hit_rates(sub, w) if n > 0 else {},
        family_swing_rates=compute_family_swing_rates(sub, w, swing_rate) if n > 0 else {f: swing_rate for f in PITCH_FAMILY_LIST},
        family_whiff_rates=compute_family_whiff_rates(sub, w, whiff_rate) if n > 0 else {f: whiff_rate for f in PITCH_FAMILY_LIST},
        sample_size=n,
        is_thin_sample=is_thin,
    )
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

```bash
./venv/bin/pytest tests/ -v
```

Expected: all tests from `test_abs_zone.py` and `test_profiles.py` `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add src/hitters/profiles.py
git commit -m "feat: wire zone_xwoba_rates and zone_hard_hit_rates into build_profile"
```

---

### Task 7: Add `DEMO_HITTERS` constant and `--rebuild-demo` CLI flag

**Files:**
- Modify: `src/hitters/profiles.py` (top-level constant + `__main__` block)

- [ ] **Step 1: Add `DEMO_HITTERS` constant**

In `src/hitters/profiles.py`, add this module-level constant just before the `if __name__ == "__main__":` block (or after the `list_available_hitters` function):

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

- [ ] **Step 2: Update the `__main__` argparse block**

Replace the existing `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build hitter profiles")
    parser.add_argument("--name",         type=str,            help="Build profile for a single player by name")
    parser.add_argument("--names",        type=str, nargs="+", help="Build profiles for multiple players by name")
    parser.add_argument("--rebuild-demo", action="store_true", help="Force-rebuild the 15 demo hitter profiles")
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/statcast_processed.parquet")

    if args.rebuild_demo:
        demo_ids = list(DEMO_HITTERS.values())
        _build_name_cache(demo_ids)
        # Seed name cache with known names as fallback if lookup missed a player
        for name, pid in DEMO_HITTERS.items():
            _NAME_CACHE.setdefault(pid, name)
        print(f"\nRebuilding {len(DEMO_HITTERS)} demo hitter profiles...\n")
        for name, pid in DEMO_HITTERS.items():
            try:
                profile = build_profile(pid, df)
                save_profile(profile)
                z5  = profile.zone_xwoba_rates.get("5")
                z14 = profile.zone_xwoba_rates.get("14")
                z5_str  = f"{z5:.3f}"  if z5  is not None else "—"
                z14_str = f"{z14:.3f}" if z14 is not None else "—"
                print(
                    f"  {profile.player_name:<25} "
                    f"n={profile.sample_size:>6,}  "
                    f"zones={len(profile.zone_xwoba_rates):>2}  "
                    f"z5={z5_str}  z14={z14_str}"
                )
            except Exception as e:
                print(f"  ERROR — {name}: {e}")

    elif args.names:
        _build_name_cache([])
        targets = args.names
        print(f"\nBuilding profiles for {len(targets)} player(s)...\n")
        for name in targets:
            try:
                pid = get_player_id(name, df)
                _NAME_CACHE[pid] = _NAME_CACHE.get(pid) or name
                profile = build_profile(pid, df)
                save_profile(profile)
                _print_profile(profile)
            except Exception as e:
                print(f"  ERROR — {name}: {e}")

    elif args.name:
        _build_name_cache([])
        pid = get_player_id(args.name, df)
        profile = build_profile(pid, df)
        save_profile(profile)
        _print_profile(profile)

    else:
        merged = merge_profiles_into_df(df)
        merged.to_parquet("data/processed/statcast_processed.parquet", index=False)
        print("Saved updated processed file with hitter profile features.")
```

- [ ] **Step 3: Commit**

```bash
git add src/hitters/profiles.py
git commit -m "feat: add DEMO_HITTERS constant and --rebuild-demo CLI flag"
```

---

### Task 8: Run `--rebuild-demo` and verify output + JSON structure

**Files:** None modified — this is a verification-only task.

- [ ] **Step 1: Run the rebuild**

```bash
./venv/bin/python -m src.hitters.profiles --rebuild-demo
```

Expected console output (approximate values — exact xwOBA numbers will vary):

```
Resolved names for 15 players.

Rebuilding 15 demo hitter profiles...

  Judge, Aaron              n=  4,XXX  zones=XX  z5=0.XXX  z14=0.XXX
  Betts, Mookie             n=  3,XXX  zones=XX  z5=0.XXX  z14=0.XXX
  ...
  Lee, Jung Hoo             n=    XXX  zones=XX  z5=0.XXX  z14=—
```

Check: no `ERROR —` lines. All 15 hitters printed. `zones=` values > 0 for most. Lee Jung Hoo may have `z14=—` (thin sample, fewer batted balls on chase pitches) — that is expected.

- [ ] **Step 2: Spot-check Aaron Judge's JSON**

```bash
cat data/processed/profiles/592450.json | python -c "
import json, sys
d = json.load(sys.stdin)
print('zone_xwoba_rates keys:', sorted(d['zone_xwoba_rates'].keys()))
print('zone_hard_hit_rates keys:', sorted(d['zone_hard_hit_rates'].keys()))
print('zone 5 xwoba:', d['zone_xwoba_rates'].get('5'))
print('zone 5 hard_hit:', d['zone_hard_hit_rates'].get('5'))
print('zone 14 xwoba:', d['zone_xwoba_rates'].get('14', 'ABSENT (thin)'))
"
```

Expected:
- `zone_xwoba_rates` has keys for most zones 1–9 (Judge has thousands of batted balls)
- Zone 5 xwOBA should be > 0.40 (heart of plate for an elite slugger)
- Zone 5 hard-hit rate should be > 0.50
- Zone 14 may be present or absent depending on batted-ball count on that location

- [ ] **Step 3: Spot-check Lee Jung Hoo's JSON**

```bash
cat data/processed/profiles/808982.json | python -c "
import json, sys
d = json.load(sys.stdin)
print('sample_size:', d['sample_size'])
print('is_thin_sample:', d['is_thin_sample'])
print('zone_xwoba_rates key count:', len(d['zone_xwoba_rates']))
print('zone_xwoba_rates keys:', sorted(d['zone_xwoba_rates'].keys()))
"
```

Expected: fewer zone keys than Judge (thin-sample player with limited career MLB PAs). Profile should still have both new fields present (possibly empty dicts or only a few zones). No error.

---

### Task 9: Write and run Judge zone xwOBA validation test + final test run

**Files:**
- Modify: `tests/test_profiles.py` (append the Judge test)

- [ ] **Step 1: Append Judge zone test to `tests/test_profiles.py`**

```python
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
```

- [ ] **Step 2: Run the Judge tests**

```bash
./venv/bin/pytest tests/test_profiles.py -k "judge" -v
```

Expected: both Judge tests `PASSED` (profile JSON was written in Task 8).

- [ ] **Step 3: Run the full test suite**

```bash
./venv/bin/pytest tests/ -v
```

Expected: all tests across `test_abs_zone.py` and `test_profiles.py` `PASSED`. Zero failures.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_profiles.py
git commit -m "test: add Judge zone xwOBA integration tests + full test suite passes"
```

---

## Self-Review Checklist

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| `zone_xwoba_rates` + `zone_hard_hit_rates` dataclass fields with `default_factory` | Task 1 |
| `MIN_ZONE_BATTED_CALLS = 10` constant | Task 1 |
| `LEAGUE_AVG["zone_xwoba"] = 0.380` | Task 1 |
| `compute_zone_xwoba_rates` — batted balls, weighted mean, omit thin | Task 4 |
| `compute_zone_hard_hit_rates` — batted balls, launch_speed >= 95, omit thin | Task 5 |
| Wire both into `build_profile()` | Task 6 |
| `resolve_hitter_features_for_pitch` with asymmetric fallbacks | Task 3 |
| `DEMO_HITTERS` constant + `--rebuild-demo` CLI | Task 7 |
| Console output: name, sample_size, zone count, z5, z14 | Task 7 |
| Run rebuild, verify all 15 hitters built | Task 8 |
| JSON spot-check: Judge (rich) + Lee Jung Hoo (thin) | Task 8 |
| Unit test: backward compatibility (legacy JSON loads) | Task 2 |
| Unit test: thin-zone omission for xwOBA | Task 4 |
| Unit test: thin-zone omission for hard-hit | Task 5 |
| Unit test: resolver fallbacks (both directions) | Task 3 |
| Unit test: Judge zone 5 xwOBA > 0.40 | Task 9 |
| `profile_to_feature_dict` unchanged | Not touched in any task ✓ |
| `merge_profiles_into_df` unchanged | Not touched in any task ✓ |
| `predict.py` unchanged | Not touched in any task ✓ |
| `_blend_profile` not updated for new zone damage fields | Not touched in any task ✓ |

**Placeholder scan:** No TBDs. All code steps contain complete, runnable code. ✓

**Type consistency:**
- `zone_xwoba_rates` → `dict` (str key), used as `str(z)` in compute functions, `str(zone)` in resolver ✓
- `zone_hard_hit_rates` → same pattern ✓
- `resolve_hitter_features_for_pitch(profile: HitterProfile, zone: int)` — same signature in Task 3 definition and referenced nowhere else (Prompt 2 wires it in) ✓
- `compute_zone_xwoba_rates(df, w)` and `compute_zone_hard_hit_rates(df, w)` — called in Task 6 with `(sub, w)`, matching Task 4/5 definitions ✓
