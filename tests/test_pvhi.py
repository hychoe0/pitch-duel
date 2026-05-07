"""Tests for pvhi.py — unified kNN architecture.

Tests use synthetic in-memory indexes written to tmp_path so the full
Statcast parquet is never required. Integration tests that use real
indexes from data/processed/pvhi_indexes/ are skipped when not present.
"""
import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PVHI_DIMS = [
    "release_speed", "spin_to_velo_ratio",
    "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension",
    "plate_x", "plate_z", "balls", "strikes",
]

_STD = {f: {"mean": 0.0, "std": 1.0} for f in PVHI_DIMS}  # identity standardization


@pytest.fixture(autouse=True)
def clear_pvhi_cache():
    """Prevent module-level index/std caches from leaking across tests."""
    from src.model import pvhi as pvhi_mod
    pvhi_mod._pvhi_index_cache.clear()
    pvhi_mod._pvhi_std_cache.clear()
    yield
    pvhi_mod._pvhi_index_cache.clear()
    pvhi_mod._pvhi_std_cache.clear()


def _make_profile(overall_xwoba=0.085, **kw):
    from src.hitters.profiles import HitterProfile
    defaults = dict(
        player_id=99999, player_name="Synthetic", stand="R",
        swing_rate=0.47, chase_rate=0.30, contact_rate=0.78,
        hard_hit_rate=0.38, whiff_rate=0.25,
        overall_xwoba_per_pitch=overall_xwoba,
    )
    defaults.update(kw)
    return HitterProfile(**defaults)


def _make_pitch(**overrides):
    p = dict(
        pitch_type="FF", release_speed=94.0, release_spin_rate=2400.0,
        pfx_x=-0.5, pfx_z=1.2, release_pos_x=-1.5, release_pos_z=6.0,
        release_extension=6.3, plate_x=0.0, plate_z=2.5,
        balls=1, strikes=2,
    )
    p.update(overrides)
    return p


def _write_synthetic_index(tmp_path, player_id, n_pitches=50,
                            xwoba_mean=0.125, pitch_type="FF"):
    """Write a .npz PVHI index with identical pitch vectors and uniform xwOBA."""
    from src.model.pvhi import PVHI_DIMENSIONS
    rng = np.random.default_rng(42)
    pvhi_matrix = rng.standard_normal((n_pitches, len(PVHI_DIMENSIONS))).astype(np.float32)
    xwoba_vals  = np.full(n_pitches, xwoba_mean, dtype=np.float32)
    pitch_types    = np.array([pitch_type] * n_pitches, dtype="U4")
    pitch_families = np.array(["fastball"] * n_pitches, dtype="U10")

    idx_dir = tmp_path / "pvhi_indexes"
    idx_dir.mkdir()
    std_path = tmp_path / "pvhi_std.json"
    np.savez_compressed(
        idx_dir / f"{player_id}.npz",
        pvhi_matrix=pvhi_matrix,
        xwoba_vals=xwoba_vals,
        pitch_types=pitch_types,
        pitch_families=pitch_families,
    )
    std_path.write_text(json.dumps(_STD))
    return idx_dir, std_path


# ---------------------------------------------------------------------------
# Test 1 — compute_pvhi returns all required keys
# ---------------------------------------------------------------------------

def test_compute_pvhi_returns_all_keys(tmp_path):
    from src.model.pvhi import compute_pvhi
    pid = 99999
    idx_dir, std_path = _write_synthetic_index(tmp_path, pid, n_pitches=50)
    r = compute_pvhi(_make_pitch(), _make_profile(), player_id=pid,
                     index_dir=idx_dir, standardization_path=std_path)
    expected = {"pvhi", "pvhi_interpretation", "pvhi_n_neighbors",
                "pvhi_relaxation_level", "pvhi_similarity_quality", "debug"}
    assert set(r) == expected


# ---------------------------------------------------------------------------
# Test 2 — fallback when no index → pvhi = 100
# ---------------------------------------------------------------------------

def test_fallback_when_no_index(tmp_path):
    from src.model.pvhi import compute_pvhi
    idx_dir = tmp_path / "empty_indexes"
    idx_dir.mkdir()
    std_path = tmp_path / "pvhi_std.json"
    std_path.write_text(json.dumps(_STD))
    r = compute_pvhi(_make_pitch(), _make_profile(), player_id=99999,
                     index_dir=idx_dir, standardization_path=std_path)
    assert r["pvhi"] == pytest.approx(100.0)
    assert r["pvhi_relaxation_level"] == 5
    assert r["pvhi_n_neighbors"] == 0
    assert r["debug"].get("fallback") is True


# ---------------------------------------------------------------------------
# Test 3 — relaxation cascade: exact type preferred over family fallback
# ---------------------------------------------------------------------------

def test_relaxation_exact_type_before_family(tmp_path):
    from src.model.pvhi import compute_pvhi, PVHI_DIMENSIONS
    pid = 99999
    n = 30
    # Build index: 30 SL pitches (no FF) — should fall to Level 2 (family) or 3
    rng = np.random.default_rng(0)
    pvhi_matrix    = rng.standard_normal((n, len(PVHI_DIMENSIONS))).astype(np.float32)
    xwoba_vals     = np.full(n, 0.100, dtype=np.float32)
    pitch_types    = np.array(["SL"] * n, dtype="U4")
    pitch_families = np.array(["breaking"] * n, dtype="U10")

    idx_dir = tmp_path / "pvhi_indexes"
    idx_dir.mkdir()
    std_path = tmp_path / "pvhi_std.json"
    std_path.write_text(json.dumps(_STD))
    np.savez_compressed(
        idx_dir / f"{pid}.npz",
        pvhi_matrix=pvhi_matrix, xwoba_vals=xwoba_vals,
        pitch_types=pitch_types, pitch_families=pitch_families,
    )

    # Query with FF — Level 1 fails (no FF), no family match either → Level 3
    r = compute_pvhi(_make_pitch(pitch_type="FF"), _make_profile(), player_id=pid,
                     index_dir=idx_dir, standardization_path=std_path)
    assert r["pvhi_relaxation_level"] >= 2


# ---------------------------------------------------------------------------
# Test 4 — blending: Level 3 blends toward 100 (confidence=0.70)
# ---------------------------------------------------------------------------

def test_level3_blends_toward_100(tmp_path):
    from src.model.pvhi import compute_pvhi, PVHI_DIMENSIONS, _RELAXATION_CONFIDENCE
    pid = 99999
    n = 30
    # Build index: all one pitch type that is NOT FF and NOT fastball family
    # so Level 1 and Level 2 both fail, forcing Level 3
    rng = np.random.default_rng(1)
    pvhi_matrix    = rng.standard_normal((n, len(PVHI_DIMENSIONS))).astype(np.float32)
    # Give very high xwOBA so pvhi_raw >> 100
    xwoba_vals     = np.full(n, 0.500, dtype=np.float32)
    pitch_types    = np.array(["KC"] * n, dtype="U4")
    pitch_families = np.array(["breaking"] * n, dtype="U10")

    idx_dir = tmp_path / "pvhi_indexes"
    idx_dir.mkdir()
    std_path = tmp_path / "pvhi_std.json"
    std_path.write_text(json.dumps(_STD))
    np.savez_compressed(
        idx_dir / f"{pid}.npz",
        pvhi_matrix=pvhi_matrix, xwoba_vals=xwoba_vals,
        pitch_types=pitch_types, pitch_families=pitch_families,
    )

    # Query FF against a "breaking only" index → forces Level 3
    profile = _make_profile(overall_xwoba=0.085)
    r = compute_pvhi(_make_pitch(pitch_type="FF"), profile, player_id=pid,
                     index_dir=idx_dir, standardization_path=std_path)

    if r["pvhi_relaxation_level"] == 3:
        conf = _RELAXATION_CONFIDENCE[3]  # 0.70
        d = r["debug"]
        pvhi_raw  = d["pvhi_raw"]
        expected  = conf * pvhi_raw + (1.0 - conf) * 100.0
        assert r["pvhi"] == pytest.approx(min(200.0, max(0.0, expected)), abs=0.1)


# ---------------------------------------------------------------------------
# Test 5 — scale bounds: pvhi always in [0, 200]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xwoba_mean,overall_xwoba", [
    (2.0,  0.001),  # extreme ratio → capped at 200
    (0.0,  0.100),  # zero numerator → pvhi = 0
    (0.10, 0.085),  # typical case
])
def test_pvhi_always_in_valid_range(tmp_path, xwoba_mean, overall_xwoba):
    from src.model.pvhi import compute_pvhi
    pid = 99999
    idx_dir, std_path = _write_synthetic_index(tmp_path, pid, n_pitches=50,
                                               xwoba_mean=xwoba_mean)
    r = compute_pvhi(_make_pitch(), _make_profile(overall_xwoba=overall_xwoba),
                     player_id=pid, index_dir=idx_dir, standardization_path=std_path)
    assert 0.0 <= r["pvhi"] <= 200.0


# ---------------------------------------------------------------------------
# Test 6 — denominator guard: 0.0 overall_xwoba falls back to league avg
# ---------------------------------------------------------------------------

def test_denominator_falls_back_when_zero(tmp_path):
    from src.model.pvhi import compute_pvhi
    from src.hitters.profiles import LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH
    pid = 99999
    idx_dir, std_path = _write_synthetic_index(tmp_path, pid, n_pitches=50, xwoba_mean=0.10)
    profile = _make_profile(overall_xwoba=0.0)
    r = compute_pvhi(_make_pitch(), profile, player_id=pid,
                     index_dir=idx_dir, standardization_path=std_path)
    assert r["pvhi"] > 0.0  # no divide-by-zero
    # denominator used must be the league avg, not 0.0
    assert r["debug"]["denom_overall_xwoba_per_pitch"] == pytest.approx(
        LEAGUE_AVG_OVERALL_XWOBA_PER_PITCH, rel=0.01
    )


# ---------------------------------------------------------------------------
# Test 7 — interpretation thresholds (unchanged from new arch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pvhi,expected", [
    (130.0, "high_danger"),
    (129.9, "above_average_danger"),
    (110.0, "above_average_danger"),
    (109.9, "neutral"),
    (90.0,  "neutral"),
    (89.9,  "safer_than_average"),
    (70.0,  "safer_than_average"),
    (69.9,  "very_safe"),
    (0.0,   "very_safe"),
])
def test_interpretation_thresholds(pvhi, expected):
    from src.model.pvhi import interpret_pvhi
    assert interpret_pvhi(pvhi) == expected


# ---------------------------------------------------------------------------
# Test 8 — integration: Judge PVHI > KIF PVHI for same center-cut FF
#           (requires real indexes built by --build-indexes --demo-only)
# ---------------------------------------------------------------------------

_JUDGE_IDX = Path("data/processed/pvhi_indexes/592450.npz")
_KIF_IDX   = Path("data/processed/pvhi_indexes/664285.npz")
_STD_FILE  = Path("models/pvhi_standardization.json")
_JUDGE_PROFILE = Path("data/processed/profiles_demo/592450.json")
_KIF_PROFILE   = Path("data/processed/profiles_demo/664285.json")

_INTEGRATION_SKIP = pytest.mark.skipif(
    not (_JUDGE_IDX.exists() and _STD_FILE.exists() and _JUDGE_PROFILE.exists()),
    reason="Real PVHI indexes not built — run: python -m src.model.pvhi --build-indexes --demo-only",
)


@_INTEGRATION_SKIP
def test_judge_pvhi_higher_than_kif_same_pitch():
    from src.model.pvhi import compute_pvhi
    from src.hitters.profiles import HitterProfile

    pitch = _make_pitch(pitch_type="FF", release_speed=95.0,
                        plate_x=0.0, plate_z=2.8, balls=1, strikes=1)

    judge_profile = HitterProfile(**json.loads(_JUDGE_PROFILE.read_text()))
    r_judge = compute_pvhi(pitch, judge_profile, player_id=592450)

    if not _KIF_IDX.exists() or not _KIF_PROFILE.exists():
        pytest.skip("KIF index or profile not built")

    kif_profile = HitterProfile(**json.loads(_KIF_PROFILE.read_text()))
    r_kif = compute_pvhi(pitch, kif_profile, player_id=664285)

    assert r_judge["pvhi"] >= r_kif["pvhi"], (
        f"Judge PVHI {r_judge['pvhi']:.1f} should be >= KIF {r_kif['pvhi']:.1f} "
        f"for a center-cut fastball (Judge is the superior hitter)"
    )
