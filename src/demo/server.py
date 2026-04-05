"""
server.py — Flask web interface for Pitch Duel model testing.

Provides endpoints for single-pitch prediction, at-bat simulation,
data lookups, and game-log evaluation.

Run:
    python -m src.demo.server
"""

import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

from src.model.predict import predict_hit_probability
from src.demo.at_bat import simulate_at_bat

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "data" / "processed" / "profiles"
PITCHER_FEAT = ROOT / "data" / "processed" / "pitcher_features.parquet"

app = Flask(__name__, template_folder=ROOT / "src" / "demo" / "templates")


# ──────────────────────────────────────────────────────────────────────────────
# Zone classification constants
# ──────────────────────────────────────────────────────────────────────────────

ZONE_X_MIN, ZONE_X_MAX = -0.85, 0.85
ZONE_Z_MIN, ZONE_Z_MAX = 1.5, 3.5
CHASE_BORDER = 0.5  # feet outside zone edges


def classify_pitch_zone(plate_x: float, plate_z: float) -> str:
    """Return zone location: 'in_zone', 'chase', or 'waste'."""
    in_x = ZONE_X_MIN <= plate_x <= ZONE_X_MAX
    in_z = ZONE_Z_MIN <= plate_z <= ZONE_Z_MAX
    if in_x and in_z:
        return "in_zone"
    chase_x = (ZONE_X_MIN - CHASE_BORDER) <= plate_x <= (ZONE_X_MAX + CHASE_BORDER)
    chase_z = (ZONE_Z_MIN - CHASE_BORDER) <= plate_z <= (ZONE_Z_MAX + CHASE_BORDER)
    if chase_x and chase_z:
        return "chase"
    return "waste"


def classify_pitch_quality(plate_x: float, plate_z: float, xwoba: float) -> dict:
    """
    Combine zone location with xwOBA to produce a pitch quality label and color.

    Returns: {"zone": str, "quality": str, "color": str}
    """
    zone = classify_pitch_zone(plate_x, plate_z)

    if zone == "in_zone":
        if xwoba < 0.06:
            return {"zone": zone, "quality": "Dominant", "color": "#3fb950"}
        elif xwoba < 0.15:
            return {"zone": zone, "quality": "Competitive", "color": "#d29922"}
        elif xwoba < 0.20:
            return {"zone": zone, "quality": "Danger", "color": "#db6d28"}
        else:
            return {"zone": zone, "quality": "Danger — mistake pitch", "color": "#f85149"}
    elif zone == "chase":
        if xwoba < 0.06:
            return {"zone": zone, "quality": "Chase pitch", "color": "#3fb950"}
        elif xwoba < 0.15:
            return {"zone": zone, "quality": "Nibbling", "color": "#d29922"}
        else:
            return {"zone": zone, "quality": "Caught too much plate", "color": "#db6d28"}
    else:  # waste
        if xwoba > 0.10:
            return {"zone": zone, "quality": "Ball — hitter chased", "color": "#d29922"}
        return {"zone": zone, "quality": "Ball", "color": "#8b949e"}

# ──────────────────────────────────────────────────────────────────────────────
# Data loaders (cached at startup)
# ──────────────────────────────────────────────────────────────────────────────

_HITTERS_CACHE = None
_PITCHERS_CACHE = None


def load_hitters():
    global _HITTERS_CACHE
    if _HITTERS_CACHE is not None:
        return _HITTERS_CACHE

    hitters = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            hitters.append(data.get("player_name", f"Player {data.get('player_id')}"))
        except Exception:
            pass
    _HITTERS_CACHE = sorted(set(hitters))
    return _HITTERS_CACHE


def load_pitchers():
    global _PITCHERS_CACHE
    if _PITCHERS_CACHE is not None:
        return _PITCHERS_CACHE

    try:
        df = pd.read_parquet(PITCHER_FEAT, columns=["pitcher"])
        pitchers = []
        for pid in sorted(df["pitcher"].unique()):
            pitchers.append({"id": int(pid), "name": f"Pitcher {pid}"})
        _PITCHERS_CACHE = pitchers
    except Exception:
        _PITCHERS_CACHE = []
    return _PITCHERS_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main interface."""
    return render_template("index.html")


@app.route("/hitters", methods=["GET"])
def hitters():
    """Return list of available hitter names."""
    return jsonify({"hitters": load_hitters()})


@app.route("/pitchers", methods=["GET"])
def pitchers():
    """Return list of available pitcher IDs and names."""
    return jsonify({"pitchers": load_pitchers()})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single-pitch prediction endpoint.

    Request JSON:
    {
        "hitter_name": str,
        "pitcher_id": int (optional),
        "pitch": {
            "release_speed": float,
            "release_spin_rate": float,
            "pfx_x": float,
            "pfx_z": float,
            "release_pos_x": float,
            "release_pos_z": float,
            "release_extension": float,
            "plate_x": float,
            "plate_z": float,
            "pitch_type": str,
            "balls": int,
            "strikes": int,
            "pitch_number": int,
            "prev_pitch_type": str,
            "prev_pitch_speed": float,
            "prev_pitch_result": str,
            "on_1b": bool/null,
            "on_2b": bool/null,
            "on_3b": bool/null,
            "inning": int,
            "score_diff": int,
            "p_throws": str,
        }
    }

    Response JSON:
    {
        "xwoba": float,
        "interpretation": str,
        "model_type": str,
    }
    """
    try:
        data = request.get_json()
        hitter_name = data.get("hitter_name")
        pitcher_id = data.get("pitcher_id")
        pitch_dict = data.get("pitch", {})

        # Fill in missing required fields with defaults
        defaults = {
            "release_spin_rate": 2400.0,
            "pfx_x": -0.5,
            "pfx_z": 1.2,
            "release_pos_x": -1.5,
            "release_pos_z": 6.0,
            "release_extension": 6.3,
            "pitch_number": 1,
            "prev_pitch_type": "FIRST_PITCH",
            "prev_pitch_speed": 0.0,
            "prev_pitch_result": "FIRST_PITCH",
            "on_1b": None,
            "on_2b": None,
            "on_3b": None,
            "inning": 1,
            "score_diff": 0,
            "p_throws": "R",
            "game_date": date.today().isoformat(),
        }
        for k, v in defaults.items():
            if k not in pitch_dict:
                pitch_dict[k] = v

        result = predict_hit_probability(pitch_dict, hitter_name, pitcher_id)

        xw = result["xwoba_prediction"]
        px = float(pitch_dict.get("plate_x", 0))
        pz = float(pitch_dict.get("plate_z", 2.5))
        pq = classify_pitch_quality(px, pz, xw)

        return jsonify({
            "xwoba": round(xw, 4),
            "interpretation": pq["quality"],
            "zone": pq["zone"],
            "quality_color": pq["color"],
            "model_type": result.get("model_type", "regressor"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/at_bat", methods=["POST"])
def at_bat():
    """
    At-bat simulator endpoint.

    Request JSON:
    {
        "hitter_name": str,
        "pitcher_id": int (optional),
        "p_throws": str (default "R"),
        "pitches": [
            {
                "pitch_type": str,
                "release_speed": float,
                "plate_x": float,
                "plate_z": float,
                "release_spin_rate": float (optional),
                "pfx_x": float (optional),
                "pfx_z": float (optional),
                ...other optional fields...
            },
            ...
        ]
    }

    Response JSON:
    {
        "hitter": str,
        "pitches": [
            {
                "pitch_num": int,
                "pitch_type": str,
                "speed": float,
                "xwoba": float,
                "interpretation": str,
                "location": str,
            },
            ...
        ],
        "avg_xwoba": float,
        "verdict": str,
    }
    """
    try:
        data = request.get_json()
        hitter_name = data.get("hitter_name")
        pitcher_id = data.get("pitcher_id")
        p_throws = data.get("p_throws", "R")
        pitches = data.get("pitches", [])

        if not pitches:
            return jsonify({"error": "No pitches provided"}), 400

        result = simulate_at_bat(
            pitcher_arsenal=pitches,
            hitter_name=hitter_name,
            pitcher_id=pitcher_id,
            p_throws=p_throws,
            verbose=False,
        )

        return jsonify({
            "hitter": result["hitter"],
            "pitches": [
                {
                    "pitch_num": p["pitch_num"],
                    "pitch_type": p["pitch_type"],
                    "speed": round(p["speed"], 1),
                    "location": p["location"],
                    "xwoba": round(p["xwoba"], 4),
                    "interpretation": p["interpretation"],
                }
                for p in result["pitches"]
            ],
            "avg_xwoba": result["avg_xwoba"],
            "best_pitch": {
                "num": result["best_pitch"]["pitch_num"],
                "type": result["best_pitch"]["pitch_type"],
                "xwoba": round(result["best_pitch"]["xwoba"], 4),
            },
            "worst_pitch": {
                "num": result["worst_pitch"]["pitch_num"],
                "type": result["worst_pitch"]["pitch_type"],
                "xwoba": round(result["worst_pitch"]["xwoba"], 4),
            },
            "verdict": result["verdict"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch", methods=["POST"])
def batch():
    """
    Batch prediction for CSV data.

    Request JSON:
    {
        "hitter_name": str,
        "pitcher_id": int (optional),
        "pitches": [
            {
                "pitch_type": str,
                "release_speed": float,
                "plate_x": float,
                "plate_z": float,
                "release_spin_rate": float (optional),
                ...
            },
            ...
        ]
    }

    Response JSON:
    {
        "predictions": [
            {
                "pitch_num": int,
                "pitch_type": str,
                "speed": float,
                "plate_x": float,
                "plate_z": float,
                "xwoba": float,
                "interpretation": str,
            },
            ...
        ],
        "summary": {
            "mean_xwoba": float,
            "min_xwoba": float,
            "max_xwoba": float,
        }
    }
    """
    try:
        data = request.get_json()
        hitter_name = data.get("hitter_name")
        pitcher_id = data.get("pitcher_id")
        pitches = data.get("pitches", [])

        if not pitches:
            return jsonify({"error": "No pitches provided"}), 400

        predictions = []
        defaults = {
            "release_spin_rate": 2400.0,
            "pfx_x": -0.5,
            "pfx_z": 1.2,
            "release_pos_x": -1.5,
            "release_pos_z": 6.0,
            "release_extension": 6.3,
            "pitch_number": 1,
            "prev_pitch_type": "FIRST_PITCH",
            "prev_pitch_speed": 0.0,
            "prev_pitch_result": "FIRST_PITCH",
            "on_1b": None,
            "on_2b": None,
            "on_3b": None,
            "inning": 1,
            "score_diff": 0,
            "p_throws": "R",
            "game_date": date.today().isoformat(),
        }

        xwobas = []
        for i, pitch_data in enumerate(pitches, 1):
            pitch_dict = {**defaults, **pitch_data}
            result = predict_hit_probability(pitch_dict, hitter_name, pitcher_id)
            xw = result["xwoba_prediction"]
            xwobas.append(xw)

            px = float(pitch_data.get("plate_x", 0))
            pz = float(pitch_data.get("plate_z", 0))
            pq = classify_pitch_quality(px, pz, xw)

            predictions.append({
                "pitch_num": i,
                "pitch_type": pitch_data.get("pitch_type", "?"),
                "speed": round(pitch_data.get("release_speed", 0), 1),
                "plate_x": round(px, 2),
                "plate_z": round(pz, 2),
                "xwoba": round(xw, 4),
                "interpretation": pq["quality"],
                "zone": pq["zone"],
                "quality_color": pq["color"],
            })

        return jsonify({
            "hitter": hitter_name,
            "predictions": predictions,
            "summary": {
                "n_pitches": len(xwobas),
                "mean_xwoba": round(sum(xwobas) / len(xwobas), 4) if xwobas else 0,
                "min_xwoba": round(min(xwobas), 4) if xwobas else 0,
                "max_xwoba": round(max(xwobas), 4) if xwobas else 0,
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/game_log", methods=["POST"])
def game_log():
    """
    Hitter game-log evaluation endpoint.

    Accepts a Baseball Savant CSV (text body or file upload), parses every
    pitch thrown TO the hitter, runs predict_hit_probability for each,
    and compares predicted xwOBA to actual outcomes.

    Baseball Savant column names are model-native (release_speed, pfx_x, etc.).
    Rows are sorted ascending by (game_pk, at_bat_number, pitch_number) since
    Savant exports in reverse order within at-bats.

    Returns structured JSON with per-pitch, per-PA, and summary data.
    """
    try:
        # Accept CSV as raw text in JSON body, or file upload
        if request.content_type and "multipart" in request.content_type:
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No file uploaded"}), 400
            csv_text = f.read().decode("utf-8")
        else:
            data = request.get_json()
            csv_text = data.get("csv", "")
            if not csv_text:
                return jsonify({"error": "No CSV data provided"}), 400

        df = pd.read_csv(io.StringIO(csv_text))

        # Validate required columns
        required = ["pitch_type", "release_speed", "plate_x", "plate_z",
                     "balls", "strikes", "pitch_number", "at_bat_number",
                     "game_pk", "batter", "description"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # Resolve hitter name from batter ID
        batter_id = int(df["batter"].iloc[0])
        hitter_name = _resolve_hitter_name(batter_id)

        # Sort ascending (Savant exports reverse order within at-bats)
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

        # Build pitcher lookup for each PA
        pitcher_ids = df.groupby(["game_pk", "at_bat_number"])["pitcher"].first().to_dict() if "pitcher" in df.columns else {}

        # Process each at-bat
        at_bats = []
        all_pitches = []
        outcome_buckets = {}  # description → list of xwobas

        for (gpk, abn), ab_df in df.groupby(["game_pk", "at_bat_number"], sort=False):
            ab_df = ab_df.sort_values("pitch_number").reset_index(drop=True)
            pa_pitches = []
            pitcher_id = pitcher_ids.get((gpk, abn))
            if pitcher_id is not None:
                pitcher_id = int(pitcher_id)

            for idx, row in ab_df.iterrows():
                # Build prev-pitch context
                if idx == 0:
                    prev_type = "FIRST_PITCH"
                    prev_speed = 0.0
                    prev_result = "FIRST_PITCH"
                else:
                    prev_row = ab_df.iloc[idx - 1]
                    prev_type = str(prev_row.get("pitch_type", "FIRST_PITCH"))
                    if prev_type == "nan":
                        prev_type = "FIRST_PITCH"
                    prev_speed = float(prev_row.get("release_speed", 0) or 0)
                    prev_result = str(prev_row.get("description", "FIRST_PITCH"))

                # Compute score_diff from CSV columns
                score_diff = 0
                if "bat_score_diff" in row.index and pd.notna(row.get("bat_score_diff")):
                    score_diff = int(row["bat_score_diff"])
                elif "bat_score" in row.index and "fld_score" in row.index:
                    bs = row.get("bat_score", 0)
                    fs = row.get("fld_score", 0)
                    if pd.notna(bs) and pd.notna(fs):
                        score_diff = int(bs) - int(fs)

                # Handle NaN pitch_type
                raw_pt = row.get("pitch_type", "FF")
                if pd.isna(raw_pt) or str(raw_pt) == "nan":
                    raw_pt = "FF"

                pitch_dict = {
                    "release_speed": float(row.get("release_speed", 90) or 90),
                    "release_spin_rate": float(row.get("release_spin_rate", 2400) or 2400),
                    "pfx_x": float(row.get("pfx_x", 0) or 0),
                    "pfx_z": float(row.get("pfx_z", 0) or 0),
                    "release_pos_x": float(row.get("release_pos_x", -1.5) or -1.5),
                    "release_pos_z": float(row.get("release_pos_z", 6.0) or 6.0),
                    "release_extension": float(row.get("release_extension", 6.3) or 6.3),
                    "plate_x": float(row.get("plate_x", 0) or 0),
                    "plate_z": float(row.get("plate_z", 2.5) or 2.5),
                    "pitch_type": str(raw_pt),
                    "balls": int(row.get("balls", 0) or 0),
                    "strikes": int(row.get("strikes", 0) or 0),
                    "pitch_number": int(row.get("pitch_number", 1) or 1),
                    "prev_pitch_type": prev_type,
                    "prev_pitch_speed": prev_speed,
                    "prev_pitch_result": prev_result,
                    "on_1b": row.get("on_1b") if pd.notna(row.get("on_1b")) else None,
                    "on_2b": row.get("on_2b") if pd.notna(row.get("on_2b")) else None,
                    "on_3b": row.get("on_3b") if pd.notna(row.get("on_3b")) else None,
                    "inning": int(row.get("inning", 1) or 1),
                    "score_diff": score_diff,
                    "p_throws": str(row.get("p_throws", "R") or "R"),
                    "stand": str(row.get("stand", "R") or "R"),
                    "game_date": str(row.get("game_date", date.today().isoformat())),
                }

                result = predict_hit_probability(pitch_dict, hitter_name, pitcher_id=pitcher_id)
                xw = result["xwoba_prediction"]

                px = pitch_dict["plate_x"]
                pz = pitch_dict["plate_z"]
                pq = classify_pitch_quality(px, pz, xw)

                # Actual outcome data
                desc = str(row.get("description", ""))
                event = str(row.get("events", "")) if pd.notna(row.get("events")) else ""
                actual_xwoba = float(row.get("estimated_woba_using_speedangle", 0)) if pd.notna(row.get("estimated_woba_using_speedangle")) else None
                woba_value = float(row.get("woba_value", 0)) if pd.notna(row.get("woba_value")) else None

                pitch_info = {
                    "pitch_num": int(row["pitch_number"]),
                    "pitch_type": pitch_dict["pitch_type"],
                    "speed": round(pitch_dict["release_speed"], 1),
                    "plate_x": round(px, 2),
                    "plate_z": round(pz, 2),
                    "count": f"{pitch_dict['balls']}-{pitch_dict['strikes']}",
                    "xwoba_pred": round(xw, 4),
                    "xwoba_actual": round(actual_xwoba, 4) if actual_xwoba is not None else None,
                    "description": desc,
                    "event": event,
                    "quality": pq["quality"],
                    "quality_color": pq["color"],
                    "zone": pq["zone"],
                }

                pa_pitches.append(pitch_info)
                all_pitches.append(pitch_info)

                # Bucket by pitch description for outcome breakdown
                bucket = desc
                if bucket not in outcome_buckets:
                    outcome_buckets[bucket] = []
                outcome_buckets[bucket].append(xw)

            # PA-level summary
            pa_event = pa_pitches[-1]["event"] if pa_pitches else ""
            pa_avg_xwoba = sum(p["xwoba_pred"] for p in pa_pitches) / len(pa_pitches) if pa_pitches else 0
            pa_woba = None
            if "woba_value" in ab_df.columns:
                last_woba = ab_df.iloc[-1].get("woba_value")
                if pd.notna(last_woba):
                    pa_woba = round(float(last_woba), 3)

            at_bats.append({
                "game_pk": int(gpk),
                "at_bat_number": int(abn),
                "pitcher_name": str(ab_df.iloc[0].get("player_name", "")) if "player_name" in ab_df.columns else "",
                "pitcher_id": pitcher_id,
                "n_pitches": len(pa_pitches),
                "event": pa_event,
                "avg_xwoba_pred": round(pa_avg_xwoba, 4),
                "woba_value": pa_woba,
                "pitches": pa_pitches,
            })

        # Outcome breakdown table
        outcome_breakdown = []
        for desc, xwobas in sorted(outcome_buckets.items(), key=lambda x: np.mean(x[1])):
            outcome_breakdown.append({
                "description": desc,
                "count": len(xwobas),
                "mean_xwoba": round(float(np.mean(xwobas)), 4),
                "min_xwoba": round(float(np.min(xwobas)), 4),
                "max_xwoba": round(float(np.max(xwobas)), 4),
            })

        # Pitch type breakdown
        pitch_type_buckets = {}
        for p in all_pitches:
            pt = p["pitch_type"]
            if pt not in pitch_type_buckets:
                pitch_type_buckets[pt] = []
            pitch_type_buckets[pt].append(p["xwoba_pred"])

        pitch_type_breakdown = []
        for pt, xwobas in sorted(pitch_type_buckets.items(), key=lambda x: np.mean(x[1])):
            pitch_type_breakdown.append({
                "pitch_type": pt,
                "count": len(xwobas),
                "mean_xwoba": round(float(np.mean(xwobas)), 4),
            })

        # Summary
        all_xwobas = [p["xwoba_pred"] for p in all_pitches]
        actual_xwobas = [p["xwoba_actual"] for p in all_pitches if p["xwoba_actual"] is not None]

        # Consistency check: does the model rank outcomes directionally correct?
        # swinging_strike should have lower mean xwoba than hit_into_play
        ss_mean = np.mean(outcome_buckets.get("swinging_strike", [0]))
        hip_mean = np.mean(outcome_buckets.get("hit_into_play", [1]))
        consistent = bool(ss_mean < hip_mean)

        summary = {
            "hitter": hitter_name,
            "batter_id": batter_id,
            "n_pitches": len(all_pitches),
            "n_pas": len(at_bats),
            "mean_xwoba_pred": round(float(np.mean(all_xwobas)), 4) if all_xwobas else 0,
            "mean_xwoba_actual": round(float(np.mean(actual_xwobas)), 4) if actual_xwobas else None,
            "consistent": consistent,
        }

        return jsonify({
            "summary": summary,
            "outcome_breakdown": outcome_breakdown,
            "pitch_type_breakdown": pitch_type_breakdown,
            "at_bats": at_bats,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


def _resolve_hitter_name(batter_id: int) -> str:
    """Look up hitter name from saved profile by MLBAM ID."""
    profile_path = PROFILE_DIR / f"{batter_id}.json"
    if profile_path.exists():
        with open(profile_path) as f:
            data = json.load(f)
        return data.get("player_name", f"Player {batter_id}")
    return f"Player {batter_id}"


if __name__ == "__main__":
    print("Starting Pitch Duel web interface...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
