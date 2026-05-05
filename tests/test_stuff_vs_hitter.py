"""Tests for stuff_vs_hitter.py — hitter-calibrated physics similarity score."""
import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers — build minimal in-memory indexes without touching disk
# ---------------------------------------------------------------------------

def _make_index(
    n: int,
    pitch_types: list[str] | None = None,
    xwoba: float = 0.35,
    whiff_rate: float = 0.30,
) -> dict:
    """Return a dict mimicking what _load_index returns."""
    rng = np.random.default_rng(42)
    pm = rng.standard_normal((n, 7)).astype(np.float32)
    xv = np.full(n, xwoba, dtype=np.float32)
    wf = (rng.random(n) < whiff_rate).astype(np.uint8)
    if pitch_types is None:
        pt = np.array(["FF"] * n, dtype="U4")
    else:
        pt = np.array(pitch_types, dtype="U4")
    return {"physics_matrix": pm, "xwoba_vals": xv, "whiff_flags": wf, "pitch_types": pt}


# ---------------------------------------------------------------------------
# Thin-sample fallback
# ---------------------------------------------------------------------------

def test_thin_sample_result_values():
    from src.model.stuff_vs_hitter import (
        _thin_sample_result, LEAGUE_AVG_STUFF_XWOBA, LEAGUE_AVG_STUFF_WHIFF
    )
    r = _thin_sample_result()
    assert r["stuff_vs_hitter_xwoba"] == pytest.approx(LEAGUE_AVG_STUFF_XWOBA)
    assert r["stuff_vs_hitter_whiff"]  == pytest.approx(LEAGUE_AVG_STUFF_WHIFF)
    assert r["n_neighbors_found"] == 0
    assert r["similarity_quality"] == pytest.approx(0.0)


def test_thin_sample_result_keys():
    from src.model.stuff_vs_hitter import _thin_sample_result
    r = _thin_sample_result()
    assert set(r) == {
        "stuff_vs_hitter_xwoba", "stuff_vs_hitter_whiff",
        "n_neighbors_found", "similarity_quality",
    }


# ---------------------------------------------------------------------------
# _filter_index — exact → family → all pitch-type cascade
# ---------------------------------------------------------------------------

def test_filter_exact_match():
    """Enough exact-match rows → returns only those."""
    from src.model.stuff_vs_hitter import _filter_index, K_MIN_NEIGHBORS
    types = ["FF"] * 20 + ["SL"] * 50
    idx = _make_index(70, pitch_types=types)
    pm, xv, wf = _filter_index(idx, "FF")
    assert len(pm) == 20


def test_filter_relaxes_to_family_when_exact_thin():
    """< K_MIN_NEIGHBORS exact matches → relaxes to pitch family."""
    from src.model.stuff_vs_hitter import _filter_index, K_MIN_NEIGHBORS
    # 3 FF + 30 SI (both fastball family) + 50 SL
    types = ["FF"] * 3 + ["SI"] * 30 + ["SL"] * 50
    idx = _make_index(83, pitch_types=types)
    pm, xv, wf = _filter_index(idx, "FF")
    # Should include FF + SI (both fastball family)
    assert len(pm) == 33


def test_filter_uses_all_when_family_also_thin():
    """Both exact and family are thin → uses entire index."""
    from src.model.stuff_vs_hitter import _filter_index, K_MIN_NEIGHBORS
    # Only 2 FF, 2 SI — fastball family total = 4 < K_MIN_NEIGHBORS
    types = ["FF"] * 2 + ["SI"] * 2 + ["CH"] * 5 + ["SL"] * 60
    idx = _make_index(69, pitch_types=types)
    pm, xv, wf = _filter_index(idx, "FF")
    assert len(pm) == 69  # fell back to all types


def test_filter_unknown_pitch_type_uses_other_family():
    """Unknown pitch_type maps to 'other' family; falls through to all if thin."""
    from src.model.stuff_vs_hitter import _filter_index
    types = ["XX"] * 2 + ["FF"] * 50
    idx = _make_index(52, pitch_types=types)
    pm, xv, wf = _filter_index(idx, "XX")
    # 2 exact < K_MIN_NEIGHBORS; "XX" → "other" family has 2 → < K_MIN_NEIGHBORS; use all 52
    assert len(pm) == 52


# ---------------------------------------------------------------------------
# Leave-one-out: dist == 0 exclusion
# ---------------------------------------------------------------------------

def test_loo_excludes_exact_self_match(tmp_path, monkeypatch):
    """
    A query vector identical to one stored vector must not be its own neighbor.
    With _exclude_dist_zero=True the exact self-match (dist=0) is removed.
    """
    from src.model import stuff_vs_hitter as svh

    # Build a 20-row index where row 0 is all zeros
    pm = np.random.default_rng(7).standard_normal((20, 7)).astype(np.float32)
    pm[0] = 0.0  # deliberately make row 0 the "self" vector
    xv = np.ones(20, dtype=np.float32) * 0.8   # high xwOBA for the self-row
    wf = np.zeros(20, dtype=np.uint8)
    pt = np.array(["FF"] * 20, dtype="U4")
    idx = {"physics_matrix": pm, "xwoba_vals": xv, "whiff_flags": wf, "pitch_types": pt}

    # Build a minimal standardization (mean=0, std=1 → no-op transform)
    std_path = tmp_path / "std.json"
    import json
    std = {f: {"mean": 0.0, "std": 1.0} for f in svh.PHYSICS_FEATURES}
    std_path.write_text(json.dumps(std))

    # Patch _load_index and _load_standardization to use our in-memory objects
    monkeypatch.setitem(svh._index_cache, 999, idx)
    monkeypatch.setitem(svh._standardization_cache, str(std_path), std)

    # Query vector == row 0 (the self-match)
    query_physics = {f: 0.0 for f in svh.PHYSICS_FEATURES}

    # Without LOO — the self-match with dist=0 IS included
    r_with = svh.compute_stuff_vs_hitter(
        query_physics, 999, "FF",
        index_dir=tmp_path, standardization_path=std_path,
        k_neighbors=10, _exclude_dist_zero=False,
    )

    # With LOO — the self-match is excluded
    r_loo = svh.compute_stuff_vs_hitter(
        query_physics, 999, "FF",
        index_dir=tmp_path, standardization_path=std_path,
        k_neighbors=10, _exclude_dist_zero=True,
    )

    # The self-row has xwOBA=0.8; the rest are ~0.8 too (same xv array)
    # The key check: LOO must not fail even when one row is excluded
    assert r_loo["n_neighbors_found"] > 0
    # And the self-excluded result used fewer candidates in the pool
    # (we can't guarantee different values since all xv are 0.8, but LOO must not error)
    assert r_loo["stuff_vs_hitter_xwoba"] is not None


def test_loo_changes_result_when_self_row_is_outlier(tmp_path, monkeypatch):
    """
    When the self-match is an outlier (very different xwOBA), excluding it changes the result.
    """
    from src.model import stuff_vs_hitter as svh
    import json

    # 15 rows with xwOBA=0.10 (neighbors), plus 1 row with xwOBA=0.99 (self)
    pm = np.zeros((16, 7), dtype=np.float32)
    pm[1:] = np.random.default_rng(3).standard_normal((15, 7)).astype(np.float32) * 0.1
    # row 0 is the self-match at the origin (query == row 0)
    xv = np.full(16, 0.10, dtype=np.float32)
    xv[0] = 0.99   # self-row is an outlier
    wf = np.zeros(16, dtype=np.uint8)
    pt = np.array(["FF"] * 16, dtype="U4")
    idx = {"physics_matrix": pm, "xwoba_vals": xv, "whiff_flags": wf, "pitch_types": pt}

    std_path = tmp_path / "std.json"
    std = {f: {"mean": 0.0, "std": 1.0} for f in svh.PHYSICS_FEATURES}
    std_path.write_text(json.dumps(std))
    monkeypatch.setitem(svh._index_cache, 998, idx)
    monkeypatch.setitem(svh._standardization_cache, str(std_path), std)

    query_physics = {f: 0.0 for f in svh.PHYSICS_FEATURES}

    r_no_loo = svh.compute_stuff_vs_hitter(
        query_physics, 998, "FF", index_dir=tmp_path,
        standardization_path=std_path, k_neighbors=16, _exclude_dist_zero=False,
    )
    r_loo = svh.compute_stuff_vs_hitter(
        query_physics, 998, "FF", index_dir=tmp_path,
        standardization_path=std_path, k_neighbors=16, _exclude_dist_zero=True,
    )
    # LOO should give lower xwOBA (self-row with 0.99 is excluded)
    assert r_loo["stuff_vs_hitter_xwoba"] < r_no_loo["stuff_vs_hitter_xwoba"]


# ---------------------------------------------------------------------------
# Blend logic
# ---------------------------------------------------------------------------

def test_blend_result_at_k_min():
    """Exactly K_MIN_NEIGHBORS neighbors → blend weight = 1/3."""
    from src.model.stuff_vs_hitter import _blend_result, K_MIN_NEIGHBORS, LEAGUE_AVG_STUFF_XWOBA, LEAGUE_AVG_STUFF_WHIFF
    computed_x, computed_w = 0.40, 0.50
    blended_x, blended_w = _blend_result(computed_x, computed_w, K_MIN_NEIGHBORS)
    w = K_MIN_NEIGHBORS / (K_MIN_NEIGHBORS * 3)
    expected_x = w * computed_x + (1 - w) * LEAGUE_AVG_STUFF_XWOBA
    expected_w = w * computed_w + (1 - w) * LEAGUE_AVG_STUFF_WHIFF
    assert blended_x == pytest.approx(expected_x)
    assert blended_w == pytest.approx(expected_w)


def test_blend_result_above_threshold():
    """>= 3 × K_MIN_NEIGHBORS → no blend, computed values returned as-is."""
    from src.model.stuff_vs_hitter import _blend_result, K_MIN_NEIGHBORS
    x, w = 0.35, 0.22
    bx, bw = _blend_result(x, w, K_MIN_NEIGHBORS * 3)
    assert bx == pytest.approx(x)
    assert bw == pytest.approx(w)


# ---------------------------------------------------------------------------
# Integration tests (require built indexes)
# ---------------------------------------------------------------------------

_JUDGE_IDX   = Path("data/processed/physics_indexes/592450.npz")
_LEE_IDX     = Path("data/processed/physics_indexes/808982.npz")
_STD_PATH    = Path("models/physics_standardization.json")


@pytest.mark.skipif(
    not (_JUDGE_IDX.exists() and _STD_PATH.exists()),
    reason="Run 'python -m src.model.stuff_vs_hitter --build-indexes --demo-only' first",
)
def test_judge_queries_a_and_b_differ():
    """Elite FF and below-average FF produce different stuff_vs_hitter values for Judge."""
    from src.model.stuff_vs_hitter import compute_stuff_vs_hitter

    qa = dict(
        release_speed=97.0, spin_to_velo_ratio=26.5,
        pfx_x=-0.3, pfx_z=1.6,
        release_pos_x=-1.5, release_pos_z=6.2, release_extension=6.5,
    )
    qb = dict(
        release_speed=93.0, spin_to_velo_ratio=24.73,
        pfx_x=0.4, pfx_z=1.0,
        release_pos_x=-1.5, release_pos_z=6.0, release_extension=6.2,
    )

    ra = compute_stuff_vs_hitter(qa, 592450, "FF")
    rb = compute_stuff_vs_hitter(qb, 592450, "FF")

    # Both should use real data (not fallback)
    from src.model.stuff_vs_hitter import LEAGUE_AVG_STUFF_XWOBA
    assert ra["stuff_vs_hitter_xwoba"] != pytest.approx(LEAGUE_AVG_STUFF_XWOBA)
    assert rb["stuff_vs_hitter_xwoba"] != pytest.approx(LEAGUE_AVG_STUFF_XWOBA)

    # The two queries must produce different values (sensitivity check)
    assert ra["stuff_vs_hitter_xwoba"] != pytest.approx(rb["stuff_vs_hitter_xwoba"], abs=1e-4)
    assert ra["stuff_vs_hitter_whiff"]  != pytest.approx(rb["stuff_vs_hitter_whiff"],  abs=1e-4)

    # n_neighbors must be > 30 for a hitter with thousands of swings
    assert ra["n_neighbors_found"] > 30
    assert rb["n_neighbors_found"] > 30


@pytest.mark.skipif(
    not _STD_PATH.exists(),
    reason="Run 'python -m src.model.stuff_vs_hitter --build-indexes --demo-only' first",
)
def test_lee_jung_hoo_returns_fallback():
    """Lee Jung Hoo has no pre-2023 index — should return the thin-sample fallback."""
    from src.model.stuff_vs_hitter import (
        compute_stuff_vs_hitter, LEAGUE_AVG_STUFF_XWOBA, LEAGUE_AVG_STUFF_WHIFF
    )
    physics = dict(
        release_speed=92.0, spin_to_velo_ratio=24.0,
        pfx_x=-0.2, pfx_z=1.1,
        release_pos_x=-1.5, release_pos_z=6.0, release_extension=6.3,
    )
    r = compute_stuff_vs_hitter(physics, 808982, "FF")
    assert r["stuff_vs_hitter_xwoba"] == pytest.approx(LEAGUE_AVG_STUFF_XWOBA)
    assert r["stuff_vs_hitter_whiff"]  == pytest.approx(LEAGUE_AVG_STUFF_WHIFF)
    assert r["n_neighbors_found"] == 0


@pytest.mark.skipif(
    not (_JUDGE_IDX.exists() and _STD_PATH.exists()),
    reason="Run 'python -m src.model.stuff_vs_hitter --build-indexes --demo-only' first",
)
def test_judge_n_neighbors_full():
    """Judge has >5k swings — k=50 neighbors should be fully satisfied."""
    from src.model.stuff_vs_hitter import compute_stuff_vs_hitter
    physics = dict(
        release_speed=95.0, spin_to_velo_ratio=25.0,
        pfx_x=-0.3, pfx_z=1.2,
        release_pos_x=-1.5, release_pos_z=6.1, release_extension=6.4,
    )
    r = compute_stuff_vs_hitter(physics, 592450, "FF", k_neighbors=50)
    assert r["n_neighbors_found"] == 50
