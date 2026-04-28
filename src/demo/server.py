"""
server.py — Flask web interface for Pitch Duel model testing.

Provides endpoints for single-pitch prediction, at-bat simulation,
data lookups, and game-log evaluation.

Run:
    python -m src.demo.server
"""

import io
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

from src.model.predict import predict_pitch, load_models, encode_pitch_dict, build_feature_row
from src.model.predict_combined import predict_matchup
from src.hitters.abs_zone import compute_abs_zone, get_height_inches
from src.demo.at_bat import simulate_at_bat

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / os.environ.get("PITCH_DUEL_PROFILES", "data/processed/profiles")
MODEL_DIR = ROOT / "models"
MODEL_V2_DIR = ROOT / "models_v2"
DEMO_MODE = os.environ.get("PITCH_DUEL_DEMO", "").lower() in ("1", "true", "yes")

# Model version registry
MODEL_VERSIONS = {
    "v1": {
        "model_dir": MODEL_DIR,
        "label": "v1 (train 2015-2024, test 2025+)",
    },
    "v2": {
        "model_dir": MODEL_V2_DIR,
        "label": "v2 (train 2015-2025, predict 2026)",
    },
}

app = Flask(__name__, template_folder=ROOT / "src" / "demo" / "templates")


# ──────────────────────────────────────────────────────────────────────────────
# Model directory resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_model_dir(version: str = None) -> Path:
    """Resolve model directory from version string. Falls back to v1."""
    if version and version in MODEL_VERSIONS:
        d = MODEL_VERSIONS[version]["model_dir"]
        if d.exists():
            return d
    return MODEL_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Feature importances (cached per model directory)
# ──────────────────────────────────────────────────────────────────────────────

_IMPORTANCES_CACHE = {}


def _load_importances(model_dir: Path = None):
    if model_dir is None:
        model_dir = MODEL_DIR
    cache_key = str(model_dir.resolve())
    if cache_key in _IMPORTANCES_CACHE:
        return _IMPORTANCES_CACHE[cache_key]

    swing_pair, contact_pair, hc_pair, feature_cols, encodings, _xwoba = load_models(model_dir)
    result = {}
    for stage_name, (model, _cal) in [("swing", swing_pair), ("contact", contact_pair), ("hard_contact", hc_pair)]:
        imp = model.feature_importances_
        result[stage_name] = {col: round(float(v), 5) for col, v in zip(feature_cols, imp)}
    _IMPORTANCES_CACHE[cache_key] = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Zone classification constants
# ──────────────────────────────────────────────────────────────────────────────

ZONE_X_MIN, ZONE_X_MAX = -0.85, 0.85
ZONE_Z_MIN, ZONE_Z_MAX = 1.5, 3.5
CHASE_BORDER = 0.5


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
    ev   = str(p.get("events",      "") or "").strip()
    desc = str(p.get("description", "") or "").strip()
    if ev and ev != "nan":
        return _EVENT_LABELS.get(ev, ev.replace("_", " ").title())
    if desc and desc != "nan":
        return _DESCRIPTION_LABELS.get(desc, desc.replace("_", " ").title())
    return "Unknown"


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


def _stage_quality_color(p_hard_contact: float, zone: str) -> str:
    """Color encoding based on hard-contact probability and zone."""
    if zone == "waste":
        return "#8b949e" if p_hard_contact < 0.05 else "#d29922"
    if p_hard_contact < 0.05:
        return "#3fb950"
    if p_hard_contact < 0.12:
        return "#79c0ff"
    if p_hard_contact < 0.22:
        return "#d29922"
    if p_hard_contact < 0.32:
        return "#db6d28"
    return "#f85149"


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders (cached at startup)
# ──────────────────────────────────────────────────────────────────────────────

_HITTERS_CACHE = None


def load_hitters():
    global _HITTERS_CACHE
    if _HITTERS_CACHE is not None:
        return _HITTERS_CACHE

    hitters = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            hitters.append({
                "name": data.get("player_name", f"Player {data.get('player_id')}"),
                "stand": data.get("stand", "R"),
            })
        except Exception:
            pass
    # Deduplicate by name, sort alphabetically
    seen = {}
    for h in hitters:
        seen[h["name"]] = h["stand"]
    _HITTERS_CACHE = [{"name": n, "stand": s} for n, s in sorted(seen.items())]
    return _HITTERS_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.route("/tablet")
def tablet():
    return render_template("tablet.html")


@app.route("/hitters", methods=["GET"])
def hitters():
    data = load_hitters()
    return jsonify({"hitters": data})


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return available model versions and which ones are ready (trained)."""
    versions = []
    for ver, info in MODEL_VERSIONS.items():
        ready = (info["model_dir"] / "swing_model.json").exists()
        versions.append({
            "version": ver,
            "label": info["label"],
            "ready": ready,
        })
    return jsonify({"models": versions})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single-pitch three-stage prediction.

    Request JSON:
    {
        "hitter_name": str,
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
            ...
        }
    }

    Response JSON:
    {
        "p_swing": float,
        "p_contact_given_swing": float,
        "p_hard_given_contact": float,
        "p_contact": float,
        "p_hard_contact": float,
        "pitch_quality": str,
        "zone": str,
        "quality_color": str,
    }
    """
    try:
        data = request.get_json()
        hitter_name = data.get("hitter_name")
        pitch_dict  = data.get("pitch", {})

        defaults = {
            "release_spin_rate": 2400.0,
            "pfx_x": -0.5, "pfx_z": 1.2,
            "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3,
            "pitch_number": 1,
            "prev_pitch_type": "FIRST_PITCH", "prev_pitch_speed": 0.0,
            "prev_pitch_result": "FIRST_PITCH",
            "on_1b": None, "on_2b": None, "on_3b": None,
            "inning": 1, "score_diff": 0, "p_throws": "R",
            "game_date": date.today().isoformat(),
        }
        for k, v in defaults.items():
            if k not in pitch_dict:
                pitch_dict[k] = v

        result = predict_pitch(pitch_dict, hitter_name)

        zone = result["zone"]
        color = _stage_quality_color(result["p_hard_contact"], zone)

        return jsonify({
            "p_swing":               result["p_swing"],
            "p_contact_given_swing": result["p_contact_given_swing"],
            "p_hard_given_contact":  result["p_hard_given_contact"],
            "p_contact":             result["p_contact"],
            "p_hard_contact":        result["p_hard_contact"],
            "pitch_quality":         result["pitch_quality"],
            "zone":                  zone,
            "quality_color":         color,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict_demo", methods=["POST"])
def predict_demo():
    """
    Full pipeline prediction with transparent breakdown.

    Returns per-stage model drivers, historical evidence, blend computation,
    hitter profile highlights, and final verdict — everything needed for
    the Prediction Demo tab to show every step of the pipeline.
    """
    try:
        data = request.get_json()
        hitter_name    = data.get("hitter_name")
        pitch_dict     = data.get("pitch", {})
        model_version  = data.get("model_version", "v1")

        model_dir = _resolve_model_dir(model_version)

        defaults = {
            "release_spin_rate": 2400.0,
            "pfx_x": -0.5, "pfx_z": 1.2,
            "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3,
            "pitch_number": 1,
            "prev_pitch_type": "FIRST_PITCH", "prev_pitch_speed": 0.0,
            "prev_pitch_result": "FIRST_PITCH",
            "on_1b": None, "on_2b": None, "on_3b": None,
            "inning": 1, "score_diff": 0, "p_throws": "R",
            "game_date": date.today().isoformat(),
        }
        for k, v in defaults.items():
            if k not in pitch_dict:
                pitch_dict[k] = v

        # Run blended prediction (model + historical similarity)
        mr = predict_matchup(pitch_dict, hitter_name, model_dir=model_dir, show_evidence=True)

        # Load hitter profile for display
        hitter_profile = _load_hitter_profile_for_display(hitter_name)

        # Get feature importances and top drivers per stage
        importances = _load_importances(model_dir)
        stage_drivers = {}
        for stage_name in ["swing", "contact", "hard_contact"]:
            stage_imp = importances[stage_name]
            # Sort by importance, take top 8
            sorted_feats = sorted(stage_imp.items(), key=lambda x: -x[1])[:8]
            stage_drivers[stage_name] = [
                {"feature": feat, "importance": imp}
                for feat, imp in sorted_feats
            ]

        # Build input features summary
        input_features = {
            "pitch_type": pitch_dict.get("pitch_type", "FF"),
            "release_speed": pitch_dict.get("release_speed", 0),
            "plate_x": round(float(pitch_dict.get("plate_x", 0)), 2),
            "plate_z": round(float(pitch_dict.get("plate_z", 2.5)), 2),
            "release_spin_rate": pitch_dict.get("release_spin_rate", 0),
            "pfx_x": pitch_dict.get("pfx_x", 0),
            "pfx_z": pitch_dict.get("pfx_z", 0),
            "release_extension": pitch_dict.get("release_extension", 0),
            "balls": pitch_dict.get("balls", 0),
            "strikes": pitch_dict.get("strikes", 0),
            "count": f"{pitch_dict.get('balls', 0)}-{pitch_dict.get('strikes', 0)}",
        }

        # Danger level
        p_hard = mr.p_hard
        if p_hard >= 0.35:
            danger = "HIGH DANGER"
        elif p_hard >= 0.15:
            danger = "MODERATE"
        else:
            danger = "LOW DANGER"

        zone = classify_pitch_zone(
            float(pitch_dict.get("plate_x", 0)),
            float(pitch_dict.get("plate_z", 2.5)),
        )
        color = _stage_quality_color(p_hard, zone)

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

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


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
    abs_zone = compute_abs_zone(h_in) if h_in is not None else None
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


def _verdict_text(mr, danger: str) -> str:
    """Generate a coach-readable verdict string."""
    if danger == "HIGH DANGER":
        return (
            f"This {mr.pitch_type} is dangerous vs {mr.hitter}. "
            f"P(hard contact) = {mr.p_hard:.1%} — expect damage."
        )
    elif danger == "MODERATE":
        return (
            f"Competitive pitch vs {mr.hitter}. "
            f"P(hard contact) = {mr.p_hard:.1%} — manageable but not safe."
        )
    else:
        return (
            f"Strong pitch vs {mr.hitter}. "
            f"P(hard contact) = {mr.p_hard:.1%} — low damage expected."
        )


@app.route("/api/hitter_profile")
def api_hitter_profile():
    """Return aggregate hitter profile stats for display."""
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    profile = _load_hitter_profile_for_display(name)
    return jsonify(profile)


@app.route("/at_bat", methods=["POST"])
def at_bat():
    """
    At-bat simulator endpoint.

    Request JSON:
    {
        "hitter_name": str,
        "p_throws": str (default "R"),
        "pitches": [
            {
                "pitch_type": str,
                "release_speed": float,
                "plate_x": float,
                "plate_z": float,
                ...optional fields...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        hitter_name = data.get("hitter_name")
        p_throws    = data.get("p_throws", "R")
        pitches     = data.get("pitches", [])

        if not pitches:
            return jsonify({"error": "No pitches provided"}), 400

        result = simulate_at_bat(
            pitcher_arsenal=pitches,
            hitter_name=hitter_name,
            p_throws=p_throws,
            verbose=False,
        )

        return jsonify({
            "hitter": result["hitter"],
            "pitches": [
                {
                    "pitch_num":             p["pitch_num"],
                    "pitch_type":            p["pitch_type"],
                    "speed":                 round(p["speed"], 1),
                    "location":              p["location"],
                    "p_swing":               round(p["p_swing"], 4),
                    "p_contact_given_swing": round(p["p_contact_given_swing"], 4),
                    "p_hard_given_contact":  round(p["p_hard_given_contact"], 4),
                    "p_contact":             round(p["p_contact"], 4),
                    "p_hard_contact":        round(p["p_hard_contact"], 4),
                    "pitch_quality":         p["pitch_quality"],
                }
                for p in result["pitches"]
            ],
            "avg_p_hard_contact": result["avg_p_hard_contact"],
            "best_pitch": {
                "num":          result["best_pitch"]["pitch_num"],
                "type":         result["best_pitch"]["pitch_type"],
                "p_hard_contact": round(result["best_pitch"]["p_hard_contact"], 4),
            },
            "worst_pitch": {
                "num":          result["worst_pitch"]["pitch_num"],
                "type":         result["worst_pitch"]["pitch_type"],
                "p_hard_contact": round(result["worst_pitch"]["p_hard_contact"], 4),
            },
            "verdict": result["verdict"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch", methods=["POST"])
def batch():
    """
    Batch prediction endpoint.

    Request JSON:
    {
        "hitter_name": str,
        "pitches": [{"pitch_type": str, "release_speed": float, ...}, ...]
    }
    """
    try:
        data = request.get_json()
        hitter_name = data.get("hitter_name")
        pitches     = data.get("pitches", [])

        if not pitches:
            return jsonify({"error": "No pitches provided"}), 400

        defaults = {
            "release_spin_rate": 2400.0,
            "pfx_x": -0.5, "pfx_z": 1.2,
            "release_pos_x": -1.5, "release_pos_z": 6.0, "release_extension": 6.3,
            "pitch_number": 1,
            "prev_pitch_type": "FIRST_PITCH", "prev_pitch_speed": 0.0,
            "prev_pitch_result": "FIRST_PITCH",
            "on_1b": None, "on_2b": None, "on_3b": None,
            "inning": 1, "score_diff": 0, "p_throws": "R",
            "game_date": date.today().isoformat(),
        }

        predictions = []
        for i, pitch_data in enumerate(pitches, 1):
            pitch_dict = {**defaults, **pitch_data}
            result = predict_pitch(pitch_dict, hitter_name)

            px = float(pitch_data.get("plate_x", 0))
            pz = float(pitch_data.get("plate_z", 0))
            zone  = result["zone"]
            color = _stage_quality_color(result["p_hard_contact"], zone)

            predictions.append({
                "pitch_num":             i,
                "pitch_type":            pitch_data.get("pitch_type", "?"),
                "speed":                 round(pitch_data.get("release_speed", 0), 1),
                "plate_x":               round(px, 2),
                "plate_z":               round(pz, 2),
                "p_swing":               result["p_swing"],
                "p_contact_given_swing": result["p_contact_given_swing"],
                "p_hard_given_contact":  result["p_hard_given_contact"],
                "p_contact":             result["p_contact"],
                "p_hard_contact":        result["p_hard_contact"],
                "pitch_quality":         result["pitch_quality"],
                "zone":                  zone,
                "quality_color":         color,
                # legacy key for SVG rendering
                "xwoba_pred":            result["p_hard_contact"],
            })

        hard_vals = [p["p_hard_contact"] for p in predictions]
        return jsonify({
            "hitter": hitter_name,
            "predictions": predictions,
            "summary": {
                "n_pitches":           len(predictions),
                "mean_p_hard_contact": round(float(np.mean(hard_vals)), 4) if hard_vals else 0,
                "min_p_hard_contact":  round(float(np.min(hard_vals)), 4) if hard_vals else 0,
                "max_p_hard_contact":  round(float(np.max(hard_vals)), 4) if hard_vals else 0,
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/game_log", methods=["POST"])
def game_log():
    """
    Hitter game-log evaluation endpoint.

    Accepts a Baseball Savant CSV (text body or file upload), parses every
    pitch thrown TO the hitter, runs predict_pitch for each, and returns
    three-stage predictions per pitch alongside actual outcomes.
    """
    try:
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

        required = ["pitch_type", "release_speed", "plate_x", "plate_z",
                    "balls", "strikes", "pitch_number", "at_bat_number",
                    "game_pk", "batter", "description"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        batter_id   = int(df["batter"].iloc[0])
        hitter_name = _resolve_hitter_name(batter_id)

        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

        at_bats = []
        all_pitches = []
        # description → lists of three-stage probabilities
        outcome_buckets: dict = {}

        for (gpk, abn), ab_df in df.groupby(["game_pk", "at_bat_number"], sort=False):
            ab_df = ab_df.sort_values("pitch_number").reset_index(drop=True)
            pa_pitches = []

            for idx, row in ab_df.iterrows():
                prev_type   = "FIRST_PITCH"
                prev_speed  = 0.0
                prev_result = "FIRST_PITCH"
                if idx > 0:
                    prev_row    = ab_df.iloc[idx - 1]
                    prev_type   = str(prev_row.get("pitch_type", "FIRST_PITCH"))
                    if prev_type == "nan":
                        prev_type = "FIRST_PITCH"
                    prev_speed  = float(prev_row.get("release_speed", 0) or 0)
                    prev_result = str(prev_row.get("description", "FIRST_PITCH"))

                score_diff = 0
                if "bat_score_diff" in row.index and pd.notna(row.get("bat_score_diff")):
                    score_diff = int(row["bat_score_diff"])
                elif "bat_score" in row.index and "fld_score" in row.index:
                    bs = row.get("bat_score", 0)
                    fs = row.get("fld_score", 0)
                    if pd.notna(bs) and pd.notna(fs):
                        score_diff = int(bs) - int(fs)

                raw_pt = row.get("pitch_type", "FF")
                if pd.isna(raw_pt) or str(raw_pt) == "nan":
                    raw_pt = "FF"

                pitch_dict = {
                    "release_speed":     float(row.get("release_speed", 90) or 90),
                    "release_spin_rate": float(row.get("release_spin_rate", 2400) or 2400),
                    "pfx_x":             float(row.get("pfx_x", 0) or 0),
                    "pfx_z":             float(row.get("pfx_z", 0) or 0),
                    "release_pos_x":     float(row.get("release_pos_x", -1.5) or -1.5),
                    "release_pos_z":     float(row.get("release_pos_z", 6.0) or 6.0),
                    "release_extension": float(row.get("release_extension", 6.3) or 6.3),
                    "plate_x":           float(row.get("plate_x", 0) or 0),
                    "plate_z":           float(row.get("plate_z", 2.5) or 2.5),
                    "pitch_type":        str(raw_pt),
                    "balls":             int(row.get("balls", 0) or 0),
                    "strikes":           int(row.get("strikes", 0) or 0),
                    "pitch_number":      int(row.get("pitch_number", 1) or 1),
                    "prev_pitch_type":   prev_type,
                    "prev_pitch_speed":  prev_speed,
                    "prev_pitch_result": prev_result,
                    "on_1b":  row.get("on_1b") if pd.notna(row.get("on_1b")) else None,
                    "on_2b":  row.get("on_2b") if pd.notna(row.get("on_2b")) else None,
                    "on_3b":  row.get("on_3b") if pd.notna(row.get("on_3b")) else None,
                    "inning":      int(row.get("inning", 1) or 1),
                    "score_diff":  score_diff,
                    "p_throws":    str(row.get("p_throws", "R") or "R"),
                    "stand":       str(row.get("stand", "R") or "R"),
                    "game_date":   str(row.get("game_date", date.today().isoformat())),
                }

                result = predict_pitch(pitch_dict, hitter_name)
                p_sw   = result["p_swing"]
                p_con  = result["p_contact_given_swing"]
                p_hard = result["p_hard_given_contact"]
                p_hc   = result["p_hard_contact"]

                px = pitch_dict["plate_x"]
                pz = pitch_dict["plate_z"]
                zone  = result["zone"]
                color = _stage_quality_color(p_hc, zone)

                desc  = str(row.get("description", ""))
                event = str(row.get("events", "")) if pd.notna(row.get("events")) else ""
                actual_xwoba = float(row.get("estimated_woba_using_speedangle", 0)) if pd.notna(row.get("estimated_woba_using_speedangle")) else None
                woba_value   = float(row.get("woba_value", 0)) if pd.notna(row.get("woba_value")) else None

                pitch_info = {
                    "pitch_num":             int(row["pitch_number"]),
                    "pitch_type":            pitch_dict["pitch_type"],
                    "speed":                 round(pitch_dict["release_speed"], 1),
                    "plate_x":               round(px, 2),
                    "plate_z":               round(pz, 2),
                    "count":                 f"{pitch_dict['balls']}-{pitch_dict['strikes']}",
                    "p_swing":               round(p_sw, 4),
                    "p_contact_given_swing": round(p_con, 4),
                    "p_hard_given_contact":  round(p_hard, 4),
                    "p_contact":             round(result["p_contact"], 4),
                    "p_hard_contact":        round(p_hc, 4),
                    "pitch_quality":         result["pitch_quality"],
                    "zone":                  zone,
                    "quality_color":         color,
                    "description":           desc,
                    "event":                 event,
                    "xwoba_actual":          round(actual_xwoba, 4) if actual_xwoba is not None else None,
                    # legacy key used by SVG builder
                    "xwoba_pred":            round(p_hc, 4),
                }

                pa_pitches.append(pitch_info)
                all_pitches.append(pitch_info)

                if desc not in outcome_buckets:
                    outcome_buckets[desc] = {
                        "p_swing": [], "p_contact": [], "p_hard": []
                    }
                outcome_buckets[desc]["p_swing"].append(p_sw)
                outcome_buckets[desc]["p_contact"].append(p_con)
                outcome_buckets[desc]["p_hard"].append(p_hard)

            pa_event    = pa_pitches[-1]["event"] if pa_pitches else ""
            pa_avg_hard = sum(p["p_hard_contact"] for p in pa_pitches) / len(pa_pitches) if pa_pitches else 0
            pa_woba = None
            if "woba_value" in ab_df.columns:
                last_woba = ab_df.iloc[-1].get("woba_value")
                if pd.notna(last_woba):
                    pa_woba = round(float(last_woba), 3)

            at_bats.append({
                "game_pk":        int(gpk),
                "at_bat_number":  int(abn),
                "pitcher_name":   str(ab_df.iloc[0].get("player_name", "")) if "player_name" in ab_df.columns else "",
                "n_pitches":      len(pa_pitches),
                "event":          pa_event,
                "avg_p_hard":     round(pa_avg_hard, 4),
                # legacy key
                "avg_xwoba_pred": round(pa_avg_hard, 4),
                "woba_value":     pa_woba,
                "pitches":        pa_pitches,
            })

        # ── Outcome breakdown table ──────────────────────────────────────────
        IN_PLAY = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}

        outcome_breakdown = []
        for desc, bucket in sorted(outcome_buckets.items()):
            n = len(bucket["p_swing"])
            entry = {
                "description":   desc,
                "count":         n,
                "mean_p_swing":  round(float(np.mean(bucket["p_swing"])), 4),
                "mean_p_contact": round(float(np.mean(bucket["p_contact"])), 4),
                "mean_p_hard":   round(float(np.mean(bucket["p_hard"])), 4)
                                  if desc in IN_PLAY or desc == "foul" else None,
            }
            outcome_breakdown.append(entry)
        # Sort: swing descending (called_strike → ball → foul → swinging_strike → hit)
        outcome_breakdown.sort(key=lambda x: -x["mean_p_swing"])

        # ── Pitch type breakdown ─────────────────────────────────────────────
        pt_buckets: dict = {}
        for p in all_pitches:
            pt = p["pitch_type"]
            if pt not in pt_buckets:
                pt_buckets[pt] = []
            pt_buckets[pt].append(p["p_hard_contact"])

        pitch_type_breakdown = [
            {
                "pitch_type":    pt,
                "count":         len(vals),
                "mean_p_hard":   round(float(np.mean(vals)), 4),
            }
            for pt, vals in sorted(pt_buckets.items(), key=lambda x: np.mean(x[1]))
        ]

        # ── Summary ──────────────────────────────────────────────────────────
        all_hard = [p["p_hard_contact"] for p in all_pitches]

        sw_bucket  = outcome_buckets.get("swinging_strike", {"p_swing": [0]})
        hip_bucket = outcome_buckets.get("hit_into_play",   {"p_swing": [0]})
        cs_bucket  = outcome_buckets.get("called_strike",   {"p_swing": [1]})
        ball_bucket = outcome_buckets.get("ball",           {"p_swing": [1]})

        sw_sw  = np.mean(sw_bucket["p_swing"])
        hip_sw = np.mean(hip_bucket["p_swing"])
        cs_sw  = np.mean(cs_bucket["p_swing"])
        ball_sw = np.mean(ball_bucket["p_swing"])

        # Sanity: swinging_strike P(swing) > called_strike P(swing) > ball P(swing)
        consistent = bool(sw_sw > cs_sw and cs_sw > ball_sw)

        summary = {
            "hitter":         hitter_name,
            "batter_id":      batter_id,
            "n_pitches":      len(all_pitches),
            "n_pas":          len(at_bats),
            "mean_p_hard":    round(float(np.mean(all_hard)), 4) if all_hard else 0,
            "consistent":     consistent,
            "sanity_note":    (
                "Consistent: swinging_strike P(swing) > called_strike > ball — model "
                "correctly predicts when the hitter swings."
                if consistent else
                "Check: P(swing) ordering may not match behavioral expectations."
            ),
            # legacy key used by frontend
            "mean_pred_all_pitches": round(float(np.mean(all_hard)), 4) if all_hard else 0,
        }

        return jsonify({
            "summary":             summary,
            "outcome_breakdown":   outcome_breakdown,
            "pitch_type_breakdown": pitch_type_breakdown,
            "at_bats":             at_bats,
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
    print("Open http://localhost:5050 in your browser")
    app.run(debug=True, port=5050)
