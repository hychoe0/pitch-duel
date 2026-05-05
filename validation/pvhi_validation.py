"""
pvhi_validation.py — Three-phase PVHI validation.

Phase 1: Same pitch vs. all 15 demo hitters (hitter differentiation)
Phase 2: Same hitter (Judge), 5 distinct pitch specs (pitch differentiation)
Phase 3: 6 pitch archetypes × 15 hitters (full PVHI matrix)

Usage:
    python -m validation.pvhi_validation
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

OUT_P1 = Path("validation/pvhi_phase1_single_pitch.csv")
OUT_P2 = Path("validation/pvhi_phase2_judge_arsenal.csv")
OUT_P3 = Path("validation/pvhi_phase3_matrix.csv")

# ---------------------------------------------------------------------------
# Shared pitch builder
# ---------------------------------------------------------------------------

_BASE_PITCH = dict(
    p_throws="R",
    pitch_number=3,
    prev_pitch_type="FF", prev_pitch_speed=94.0, prev_pitch_result="ball",
    on_1b=None, on_2b=None, on_3b=None,
    inning=5, score_diff=0,
    release_pos_x=-1.5, release_pos_z=6.2, release_extension=6.4,
    release_spin_rate=2300,
)

from src.hitters.profiles import DEMO_HITTERS  # noqa: E402


def _pitch(**overrides) -> dict:
    p = dict(_BASE_PITCH)
    p.update(overrides)
    return p


def _zone_plate_coords(zone: int) -> tuple[float, float]:
    """Return representative (plate_x, plate_z) for a Statcast zone."""
    coords = {
        1:  (-0.45, 3.2),
        2:  (0.00,  3.2),
        3:  (0.45,  3.2),
        4:  (-0.45, 2.5),
        5:  (0.00,  2.5),
        6:  (0.45,  2.5),
        7:  (-0.45, 1.8),
        8:  (0.00,  1.8),
        9:  (0.45,  1.8),
        11: (-1.1,  3.3),
        12: (1.1,   3.3),
        13: (-1.1,  1.2),
        14: (1.1,   1.2),
    }
    return coords.get(zone, (0.0, 2.5))


def _predict_pvhi(pitch: dict, hitter_name: str) -> dict | None:
    from src.model.predict import predict_pitch
    try:
        return predict_pitch(pitch, hitter_name)
    except Exception as e:
        warnings.warn(f"predict_pitch failed for {hitter_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase 1 — Same pitch vs. all 15 demo hitters
# ---------------------------------------------------------------------------

def phase1() -> pd.DataFrame:
    """FF, ~94-95 mph, zone 1, 1-2 count — who is most threatened?"""
    px, pz = _zone_plate_coords(1)
    pitch = _pitch(
        pitch_type="FF", release_speed=94.5, release_spin_rate=2350,
        pfx_x=-0.3, pfx_z=1.3,
        plate_x=px, plate_z=pz,
        balls=1, strikes=2,
    )

    rows = []
    for name in DEMO_HITTERS:
        r = _predict_pvhi(pitch, name)
        if r is None:
            continue
        rows.append({
            "hitter":        name,
            "pvhi":          r["pvhi"],
            "pvhi_stuff":    r["pvhi_stuff"],
            "pvhi_location": r["pvhi_location"],
            "pvhi_count":    r["pvhi_count"],
            "interpretation": r["pvhi_interpretation"],
            "p_hard_contact": r["p_hard_contact"],
        })

    df = pd.DataFrame(rows).sort_values("pvhi", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    OUT_P1.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_P1, index=False)

    print("\n── Phase 1: FF zone-1, 1-2 count vs. all DEMO_HITTERS ─────────────")
    print(f"  {'Rank':<5} {'Hitter':<26} {'PVHI':>7}  {'Stuff':>7}  {'Loc':>7}  {'Count':>7}")
    print(f"  {'─'*5} {'─'*26} {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for _, row in df.iterrows():
        print(f"  #{int(row['rank']):<4} {row['hitter']:<26} {row['pvhi']:>7.1f}  "
              f"{row['pvhi_stuff']:>7.1f}  {row['pvhi_location']:>7.1f}  {row['pvhi_count']:>7.1f}")

    spread = df["pvhi"].max() - df["pvhi"].min()
    print(f"\n  Spread (max–min): {spread:.1f}")
    check = spread >= 30
    print(f"  Gate: spread ≥ 30 → {'PASS' if check else 'FAIL'}")
    return df


# ---------------------------------------------------------------------------
# Phase 2 — Judge vs. 5 distinct pitch specs
# ---------------------------------------------------------------------------

def _judge_arsenal() -> list[tuple[str, dict]]:
    px1, pz1 = _zone_plate_coords(1)
    px5, pz5 = _zone_plate_coords(5)
    px14, pz14 = _zone_plate_coords(14)
    px13, pz13 = _zone_plate_coords(13)
    return [
        ("a_FF_zone1_12",  _pitch(pitch_type="FF", release_speed=95.0, release_spin_rate=2300,
                                  pfx_x=-0.3, pfx_z=1.5, plate_x=px1,  plate_z=pz1,  balls=1, strikes=2)),
        ("b_FF_zone5_00",  _pitch(pitch_type="FF", release_speed=90.0, release_spin_rate=2100,
                                  pfx_x=-0.2, pfx_z=0.8, plate_x=px5,  plate_z=pz5,  balls=0, strikes=0)),
        ("c_SL_zone14_12", _pitch(pitch_type="SL", release_speed=87.0, release_spin_rate=2400,
                                  pfx_x=1.0,  pfx_z=-0.5, plate_x=px14, plate_z=pz14, balls=1, strikes=2)),
        ("d_CH_zone13_12", _pitch(pitch_type="CH", release_speed=84.0, release_spin_rate=1900,
                                  pfx_x=-0.8, pfx_z=0.5, plate_x=px13, plate_z=pz13, balls=1, strikes=2)),
        ("e_CU_zone5_31",  _pitch(pitch_type="CU", release_speed=78.0, release_spin_rate=2600,
                                  pfx_x=0.5,  pfx_z=-1.2, plate_x=px5,  plate_z=pz5,  balls=3, strikes=1)),
    ]


def phase2() -> pd.DataFrame:
    rows = []
    for label, pitch in _judge_arsenal():
        r = _predict_pvhi(pitch, "Judge, Aaron")
        if r is None:
            continue
        rows.append({
            "pitch_spec":    label,
            "pvhi":          r["pvhi"],
            "pvhi_stuff":    r["pvhi_stuff"],
            "pvhi_location": r["pvhi_location"],
            "pvhi_count":    r["pvhi_count"],
            "interpretation": r["pvhi_interpretation"],
        })

    df = pd.DataFrame(rows).sort_values("pvhi", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_P2, index=False)

    print("\n── Phase 2: Aaron Judge vs. 5 pitch specs ──────────────────────────")
    print(f"  {'Spec':<22} {'PVHI':>7}  {'Stuff':>7}  {'Loc':>7}  {'Count':>7}  Interp")
    print(f"  {'─'*22} {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  ─────────────")
    for _, row in df.iterrows():
        print(f"  {row['pitch_spec']:<22} {row['pvhi']:>7.1f}  {row['pvhi_stuff']:>7.1f}  "
              f"{row['pvhi_location']:>7.1f}  {row['pvhi_count']:>7.1f}  {row['interpretation']}")

    pvhi_by_spec = {row["pitch_spec"]: row["pvhi"] for _, row in df.iterrows()}
    e_highest = pvhi_by_spec.get("e_CU_zone5_31", 0)  == df["pvhi"].max()
    c_lowest  = pvhi_by_spec.get("c_SL_zone14_12", 999) == df["pvhi"].min()
    print(f"\n  Gate: spec-e (hanger CU) → highest PVHI: {'PASS' if e_highest else 'FAIL'}")
    print(f"  Gate: spec-c (chase SL)  → lowest PVHI:  {'PASS' if c_lowest  else 'FAIL'}")
    return df


# ---------------------------------------------------------------------------
# Phase 3 — 6 archetypes × 15 hitters matrix
# ---------------------------------------------------------------------------

def _archetypes() -> dict[str, dict]:
    def _zp(z): return _zone_plate_coords(z)
    return {
        "FF_z1":  _pitch(pitch_type="FF", release_speed=94.0, release_spin_rate=2300,
                         pfx_x=-0.3, pfx_z=1.4,
                         plate_x=_zp(1)[0], plate_z=_zp(1)[1], balls=1, strikes=1),
        "FF_z9":  _pitch(pitch_type="FF", release_speed=94.0, release_spin_rate=2300,
                         pfx_x=-0.3, pfx_z=0.9,
                         plate_x=_zp(9)[0], plate_z=_zp(9)[1], balls=1, strikes=1),
        "SL_z14": _pitch(pitch_type="SL", release_speed=86.0, release_spin_rate=2400,
                         pfx_x=1.2, pfx_z=-0.4,
                         plate_x=_zp(14)[0], plate_z=_zp(14)[1], balls=0, strikes=2),
        "CH_z13": _pitch(pitch_type="CH", release_speed=83.0, release_spin_rate=1850,
                         pfx_x=-0.9, pfx_z=0.4,
                         plate_x=_zp(13)[0], plate_z=_zp(13)[1], balls=0, strikes=2),
        "CU_z5":  _pitch(pitch_type="CU", release_speed=78.0, release_spin_rate=2600,
                         pfx_x=0.4, pfx_z=-1.1,
                         plate_x=_zp(5)[0], plate_z=_zp(5)[1], balls=3, strikes=1),
        "FF_z5":  _pitch(pitch_type="FF", release_speed=93.0, release_spin_rate=2250,
                         pfx_x=-0.2, pfx_z=1.1,
                         plate_x=_zp(5)[0], plate_z=_zp(5)[1], balls=0, strikes=0),
    }


def phase3() -> pd.DataFrame:
    archs = _archetypes()
    records = []
    for hitter in DEMO_HITTERS:
        row = {"hitter": hitter}
        for archetype, pitch in archs.items():
            r = _predict_pvhi(pitch, hitter)
            row[archetype] = r["pvhi"] if r is not None else None
        records.append(row)

    df = pd.DataFrame(records).set_index("hitter")
    df.to_csv(OUT_P3)

    print("\n── Phase 3: 6×15 PVHI matrix ───────────────────────────────────────")
    stds = {col: df[col].std() for col in df.columns}
    print(f"\n  Hitter differentiation (std per pitch archetype):")
    for col, std in sorted(stds.items(), key=lambda x: -x[1]):
        flag = "PASS" if std > 15 else "fail"
        print(f"  [{flag}] {col:<10}  std={std:.1f}")

    n_pass = sum(1 for v in stds.values() if v > 15)
    print(f"\n  Gate: std > 15 for ≥ 4 of 6 archetypes → {'PASS' if n_pass >= 4 else 'FAIL'}  ({n_pass}/6)")
    best  = max(stds, key=stds.get)
    worst = min(stds, key=stds.get)
    print(f"  Highest differentiation: {best} (std={stds[best]:.1f})")
    print(f"  Lowest differentiation:  {worst} (std={stds[worst]:.1f})")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("\n=== PVHI Validation ===")
    phase1()
    phase2()
    phase3()
    print(f"\nSaved:\n  {OUT_P1}\n  {OUT_P2}\n  {OUT_P3}")


if __name__ == "__main__":
    run()
