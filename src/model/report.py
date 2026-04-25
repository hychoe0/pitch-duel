"""
report.py — Per-league pitch outcome report generator.

Runs a set of named pitches against a list of hitters in one league and
returns a structured summary dict. Call generate_league_report() twice
(once for MLB, once for AAA) to produce separated, non-combined reports.

Usage:
    from src.model.report import generate_league_report, render_report

    report = generate_league_report(pitches, hitter_names, league="MLB",
                                    pitcher_description="Mid-tier D1 RHP")
    print(render_report(report))
"""

import warnings
from pathlib import Path

import pandas as pd


def generate_league_report(
    pitches: list[dict],
    hitter_names: list[str],
    league: str,
    pitcher_description: str = "",
) -> dict:
    """
    Run all pitches against a list of specific named hitters in one league.

    Args:
        pitches: list of pitch dicts (Trackman-ingested format)
        hitter_names: specific hitters in this league to predict against
        league: "MLB" or "AAA" — controls profile lookup directory
        pitcher_description: optional short string for report header

    Returns dict with keys:
        league, pitcher_description, n_pitches, n_hitters,
        hitters (sorted by avg_p_hard desc), pitch_type_breakdown,
        records (raw per-pitch per-hitter rows, for CSV export)
    """
    from src.model.predict_combined import predict_matchup

    records = []
    for pitch in pitches:
        for name in hitter_names:
            try:
                r = predict_matchup(pitch, name, show_evidence=False, league=league)
                records.append({
                    "hitter":      name,
                    "pitch_type":  pitch.get("pitch_type", "?"),
                    "velo":        round(float(pitch.get("release_speed", 0)), 1),
                    "plate_x":     round(float(pitch.get("plate_x", 0)), 3),
                    "plate_z":     round(float(pitch.get("plate_z", 0)), 3),
                    "balls":       pitch.get("balls", 0),
                    "strikes":     pitch.get("strikes", 0),
                    "p_swing":     r.p_swing,
                    "p_contact":   r.p_contact,
                    "p_hard":      r.p_hard,
                    "xwoba":       r.predicted_xwoba_per_pitch,  # None when model not trained
                    "alpha":       r.alpha,
                    "n_similar":   r.n_similar_pitches,
                })
            except Exception as e:
                warnings.warn(f"predict_matchup failed for '{name}' (league={league}): {e}")

    if not records:
        return {
            "league": league.upper(),
            "pitcher_description": pitcher_description,
            "n_pitches": len(pitches),
            "n_hitters": len(hitter_names),
            "hitters": [],
            "pitch_type_breakdown": {},
            "records": [],
        }

    df = pd.DataFrame(records)

    # Determine danger metric: xwOBA when the model is trained, p_hard otherwise.
    # xwOBA is None for every row when the xwOBA regressor hasn't been trained yet.
    xwoba_available = df["xwoba"].notna().any()
    if xwoba_available:
        danger_col = "xwoba"
        danger_90p = df["xwoba"].quantile(0.90)
    else:
        danger_col = "p_hard"
        danger_90p = df["p_hard"].quantile(0.90)

    hitters_out = []
    for name in hitter_names:
        sub = df[df["hitter"] == name]
        if len(sub) == 0:
            continue

        most_danger_idx = sub[danger_col].idxmax()
        md = sub.loc[most_danger_idx]

        xwoba_vals = sub["xwoba"].dropna()
        avg_xwoba = round(float(xwoba_vals.mean()), 4) if len(xwoba_vals) > 0 else None

        hitters_out.append({
            "name":          name,
            "avg_p_swing":   round(float(sub["p_swing"].mean()), 4),
            "avg_p_contact": round(float(sub["p_contact"].mean()), 4),
            "avg_p_hard":    round(float(sub["p_hard"].mean()), 4),
            "avg_xwoba":     avg_xwoba,
            "n_danger_pitches": int((sub[danger_col] >= danger_90p).sum()),
            "most_dangerous_pitch": {
                "pitch_type": str(md["pitch_type"]),
                "velo":       round(float(md["velo"]), 1),
                "plate_x":    round(float(md["plate_x"]), 3),
                "plate_z":    round(float(md["plate_z"]), 3),
                "xwoba":      round(float(md["xwoba"]), 4) if pd.notna(md["xwoba"]) else None,
            },
        })

    hitters_out.sort(key=lambda h: -h["avg_p_hard"])

    # Pitch type breakdown: avg P(hard) and the single most-dangerous hitter per type
    pitch_type_breakdown = {}
    for pt, grp in df.groupby("pitch_type"):
        hitter_avg = grp.groupby("hitter")["p_hard"].mean()
        most_dangerous = str(hitter_avg.idxmax()) if len(hitter_avg) > 0 else ""
        pitch_type_breakdown[str(pt)] = {
            "avg_p_hard":               round(float(grp["p_hard"].mean()), 4),
            "most_dangerous_hitter_name": most_dangerous,
        }

    return {
        "league":               league.upper(),
        "pitcher_description":  pitcher_description,
        "n_pitches":            len(pitches),
        "n_hitters":            len(hitter_names),
        "hitters":              hitters_out,
        "pitch_type_breakdown": pitch_type_breakdown,
        "records":              records,
    }


def render_report(report: dict) -> str:
    """
    Render a league report dict as a coach-readable terminal string.
    Does not print; returns the full string so the caller controls output.
    """
    W = 64
    lines = []

    league = report["league"]
    lines.append("=" * W)
    lines.append(f"  {league} HITTER PREDICTIONS")
    lines.append("=" * W)
    if report["pitcher_description"]:
        lines.append(f"  Pitcher: {report['pitcher_description']}")
    lines.append(f"  Pitches analyzed: {report['n_pitches']}")
    lines.append(f"  Hitters: {report['n_hitters']}")
    lines.append("")

    # Hitter table — xwOBA column shows N/A when regressor not yet trained
    xwoba_trained = any(h["avg_xwoba"] is not None for h in report["hitters"])
    danger_label = "Danger pitches" if xwoba_trained else "Danger (P(hard))"
    xwoba_hdr    = "avg xwOBA" if xwoba_trained else "avg xwOBA"
    lines.append(
        f"  {'Hitter':<26}  {'avg P(hard)':<12}  {xwoba_hdr:<10}  {danger_label}"
    )
    lines.append(f"  {'─'*26}  {'─'*12}  {'─'*10}  {'─'*16}")
    for h in report["hitters"]:
        xwoba_str = f"{h['avg_xwoba']:.3f}" if h["avg_xwoba"] is not None else "N/A"
        lines.append(
            f"  {h['name']:<26}  {h['avg_p_hard']:<12.3f}  "
            f"{xwoba_str:<10}  {h['n_danger_pitches']}"
        )
    if not xwoba_trained:
        lines.append("  (xwOBA regressor not yet trained — danger count uses top-decile P(hard))")
    lines.append("")

    # Pitch type breakdown
    ptb = report["pitch_type_breakdown"]
    if ptb:
        by_p_hard = sorted(ptb.items(), key=lambda kv: -kv[1]["avg_p_hard"])
        hardest_pt,  hardest_val  = by_p_hard[0]
        easiest_pt, easiest_val = by_p_hard[-1]

        lines.append(
            f"  Most hittable pitch type:  {hardest_pt} "
            f"(avg P(hard) {hardest_val['avg_p_hard']:.3f})"
        )
        lines.append(
            f"    Most dangerous hitter on {hardest_pt}: "
            f"{hardest_val['most_dangerous_hitter_name']}"
        )
        lines.append(
            f"  Most effective pitch type: {easiest_pt} "
            f"(avg P(hard) {easiest_val['avg_p_hard']:.3f})"
        )
        lines.append(
            f"    Most dangerous hitter on {easiest_pt}: "
            f"{easiest_val['most_dangerous_hitter_name']}"
        )

    lines.append("=" * W)
    return "\n".join(lines)
