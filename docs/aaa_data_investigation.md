# AAA Statcast Data Investigation

**Date:** 2026-04-23  
**Scope:** Phase 1 of Task 2 — Triple-A Hitter Profile Integration  
**Conclusion: GO — proceed to Phase 2**

---

## 1. pybaseball Capabilities for Minor League Data

**No native AAA/MiLB functions exist in pybaseball 2.2.7.**

All `statcast*` functions in pybaseball (`statcast()`, `statcast_batter()`, `statcast_pitcher()`, etc.) hit `https://baseballsavant.mlb.com/statcast_search/csv` with hardcoded URL parameters that do not include a sport/level selector. The `_SC_SMALL_REQUEST` template:

```
/statcast_search/csv?all=true&...&hfGT=R%7CPO%7CS%7C=&...
```

filters by game type (R=Regular, PO=Postseason, S=Spring) but has no `sport_id` or `hfLevel` parameter. There are no `statcast_minor*`, `milb_*`, or `triple_a_*` functions anywhere in the package.

**Workaround:** Call the Baseball Savant CSV endpoint directly with `sport_id=11` (see §2). This is the same endpoint pybaseball uses for MLB; AAA is just a different scope on the same API.

---

## 2. Baseball Savant AAA Endpoint

**Endpoint confirmed working (no authentication required):**

```
https://baseballsavant.mlb.com/statcast_search/csv
  ?all=true
  &hfPT=&hfAB=&hfGT=R%7C
  &player_type=pitcher
  &game_date_gt={YYYY-MM-DD}
  &game_date_lt={YYYY-MM-DD}
  &type=details
  &sport_id=11
```

`sport_id=11` = Triple-A (all affiliated AAA leagues: International League + Pacific Coast League).  
`sport_id` is absent in the MLB query, which defaults to MLB (sport_id=1).

---

## 3. Column Comparison: AAA vs. MLB Statcast

**Zero column differences.** Both return exactly 118 columns.

```
Columns in AAA but not MLB: []
Columns in MLB but not AAA: []
```

Full schema is identical including all fields needed for profile building and model inference.

### Critical field fill rates (3-day AAA sample, April 2023, n=12,217)

| Field | Fill Rate | Notes |
|---|---|---|
| `description` | 100.0% | All pitch outcomes present |
| `plate_x` | 99.6% | < 0.5% null (tracking failure) |
| `plate_z` | 99.6% | Same |
| `zone` | 99.6% | Values 1–9 (strike) + 11–14 (ball) — identical to MLB |
| `pitch_type` | 99.6% | Same codes as MLB (FF, SI, SL, CH, FC, ST, CU, FS…) |
| `batter` | 100.0% | MLBAM IDs |
| `player_name` | 100.0% | "Last, First" format (same as MLB Savant exports) |
| `stand` | 100.0% | L/R values |
| `balls` / `strikes` | 100.0% | Count before pitch |
| `release_speed` | 99.6% | Velocity range: 64.8–102.1 mph (mean 88.7) |
| `release_spin_rate` | 98.6% | |
| `pfx_x` / `pfx_z` | 99.6% | |
| `p_throws` | 100.0% | |
| `release_extension` | 99.4% | |
| `launch_speed` | 32.3% | Expected — only populated on balls in play (~25% of pitches) |
| `launch_angle` | 32.4% | Same |
| `events` | 25.4% | Same — terminal pitch of PA only |
| `estimated_woba_using_speedangle` | 25.2% | Same — in-play only |

All "WARN" fields are expected sparse. They are NOT missing — they simply don't apply to non-contact pitches, exactly as in MLB data.

---

## 4. Data Volume

### Row cap warning
A single query covering a full month returns exactly 25,000 rows (hard cap). **Do not query more than 4–5 days at a time.** Short-window chunking (≤5 days per request) bypasses the cap and returns all data.

### Actual volume (5-day chunks, confirmed)

| Period | Rows |
|---|---|
| April 2024, 5-day chunks (6 chunks) | **116,768** rows |

| Estimate | Rows |
|---|---|
| One AAA month (~116k) | ~116,000 |
| One AAA season (5.5 months) | ~640,000 |
| 2023 + 2024 + 2025 + 2026 YTD | **~2.0–2.5M rows total** |

This is roughly 1/3 the size of the MLB dataset (6.4M rows per 10 years). Very manageable.

### Year availability confirmed
- 2023: ✓ available (first year of AAA Statcast partnership)
- 2024: ✓ available
- 2025: ✓ available
- 2026 YTD: ✓ available

### Request timing
- 3-day chunk: ~8 seconds
- 5-day chunk: ~20–25 seconds
- 7-day chunk: times out at 30s; need 60s timeout or smaller chunks
- **Recommended chunk size: 3–5 days**, 60-second timeout per request

---

## 5. Authentication & Rate Limiting

- **Authentication:** None required. All requests are anonymous HTTP GET to the public Savant CSV endpoint.
- **Rate limiting:** No explicit rate limit observed. Requests take 20–25s for 5-day chunks due to server-side query cost, not throttling. Practical throughput: ~2–3 chunks/minute.
- **Robots.txt / ToS:** Baseball Savant is MLB's official public data service; fetching structured CSV data is the documented use. No scraping of HTML is involved.

---

## 6. Notable Data Characteristics

- **team codes** use MLB affiliate abbreviations (WSH, CHC, MIL, NYY, etc.) not independent AAA team names.
- **batter IDs** are MLBAM IDs — the same namespace as MLB profiles. Players on rehab assignments from MLB will appear with their existing MLBAM IDs. This is actually useful: if a player has both MLB and AAA rows, they share a `batter` ID.
- **game_type = 'R' only** — regular season games. No spring training, no playoffs.
- **inning range:** 1–11 (extra innings in AAA).
- **description values** match MLB exactly: `ball`, `foul`, `hit_into_play`, `called_strike`, `swinging_strike`, `blocked_ball`, `foul_tip`, `swinging_strike_blocked`, `hit_by_pitch`, `automatic_ball` (automated balls/strikes pilot in some AAA games).

One **noteworthy difference**: AAA includes `automatic_ball` / `automatic_strike` from the ABS (Automated Ball-Strike) system pilot used in Triple-A since 2023. These are valid pitch outcomes but do not exist in MLB data. The preprocess pipeline's `PREV_RESULT_MAP` does not include them — they should be mapped to `"ball"` and `"called_strike"` respectively.

---

## 7. Go / No-Go Recommendation

**GO.**

| Criterion | Status |
|---|---|
| Public API access, no auth | ✓ |
| Identical column schema to MLB | ✓ |
| All hitter profile fields present and filled | ✓ |
| Sufficient data volume for profile building | ✓ ~2M rows, 2023–2026 |
| No legal / ToS concerns | ✓ |
| Clear fetch strategy (5-day chunks, 60s timeout) | ✓ |

The only engineering considerations for Phase 2:
1. Use 5-day chunks to avoid the 25k row cap.
2. Add `"automatic_ball"` → `"ball"` and `"automatic_strike"` → `"called_strike"` mappings in the AAA preprocessor.
3. Handle rehab players: if a batter has rows in both MLB and AAA data, the MLB profile takes precedence (they already have a full MLB profile from the existing pipeline).
