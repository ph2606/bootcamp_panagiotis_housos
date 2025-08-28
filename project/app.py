# project/app.py
from __future__ import annotations
from flask import Flask, request, jsonify, send_file
import io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))
from productize import load_artifacts, vector_from_payload, run_full_training_and_save

app = Flask(__name__)

# Load model on startup; if missing, attempt training
try:
    pipe, defaults, feature_order = load_artifacts(project_root)
    MODEL_READY = True
except Exception:
    try:
        run_full_training_and_save(project_root)
        pipe, defaults, feature_order = load_artifacts(project_root)
        MODEL_READY = True
    except Exception as e:
        MODEL_READY = False
        pipe = defaults = feature_order = None
        STARTUP_ERROR = str(e)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_READY})

@app.post("/predict")
def predict_post():
    if not MODEL_READY:
        return jsonify({"error": "Model not available", "details": STARTUP_ERROR}), 503
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Payload must be a JSON object mapping feature->value"}), 400
        vec = vector_from_payload(payload, feature_order, defaults)
        pred = float(pipe.predict(vec)[0])
        return jsonify({"prediction": pred, "features_used": feature_order})
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 400

# Demonstration endpoints required by homework (single or two path params)
@app.get("/predict/<float:ret>")
def predict_one(ret: float):
    if not MODEL_READY:
        return jsonify({"error": "Model not available", "details": STARTUP_ERROR}), 503
    payload = {"ret": ret}
    vec = vector_from_payload(payload, feature_order, defaults)
    pred = float(pipe.predict(vec)[0])
    return jsonify({"prediction": pred, "filled_defaults": True})

@app.get("/predict/<float:ret>/<float:ret_lag1>")
def predict_two(ret: float, ret_lag1: float):
    if not MODEL_READY:
        return jsonify({"error": "Model not available", "details": STARTUP_ERROR}), 503
    payload = {"ret": ret, "ret_lag1": ret_lag1}
    vec = vector_from_payload(payload, feature_order, defaults)
    pred = float(pipe.predict(vec)[0])
    return jsonify({"prediction": pred, "filled_defaults": True})

# Simple plot endpoint (returns PNG): top coefficients by magnitude (if accessible)
@app.get("/plot")
def plot_coeffs():
    if not MODEL_READY:
        return jsonify({"error": "Model not available", "details": STARTUP_ERROR}), 503
    try:
        # Try to access linear model coefficients (after scaler)
        model = pipe.named_steps.get("model", None)
        if model is None or not hasattr(model, "coef_"):
            raise RuntimeError("Model coefficients not available.")
        coef = model.coef_
        names = feature_order
        idx = np.argsort(np.abs(coef))[::-1][:10]
        plt.figure()
        plt.bar(range(len(idx)), np.array(coef)[idx])
        plt.xticks(range(len(idx)), [names[i] for i in idx], rotation=30, ha="right")
        plt.title("Top 10 Coefficients (magnitude)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": "Plot failed", "details": str(e)}), 500

# Optional: run full analysis & retrain endpoint
@app.post("/run_full_analysis")
def run_full():
    try:
        metrics = run_full_training_and_save(project_root)
        # reload artifacts after training
        global pipe, defaults, feature_order, MODEL_READY
        pipe, defaults, feature_order = load_artifacts(project_root)
        MODEL_READY = True
        return jsonify({"status": "ok", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
