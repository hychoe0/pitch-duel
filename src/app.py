"""
app.py — Flask web server for Pitch Duel simulator.

Usage:
    pip install flask
    python -m src.app

Then open http://localhost:5000 in your browser.
"""

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates")

# Warm model cache at startup so first prediction isn't slow
try:
    from src.model.predict import load_model
    load_model()
    print("Model loaded.")
except FileNotFoundError:
    print("No trained model found — run train.py first.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    from src.model.predict import predict_hit_probability
    data = request.get_json()
    try:
        result = predict_hit_probability(data["pitch"], data["hitter"])
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/hitters", methods=["GET"])
def hitters():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    processed = Path("data/processed/statcast_processed.parquet")
    if not processed.exists():
        return jsonify([])
    try:
        df = pd.read_parquet(processed, columns=["player_name"])
        names = df["player_name"].dropna().unique()
        matches = sorted(n for n in names if q in str(n).lower())[:20]
        return jsonify(list(matches))
    except Exception:
        return jsonify([])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
