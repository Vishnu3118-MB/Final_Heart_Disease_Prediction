import os
import json
import io
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, flash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# --------------------------
# PATH CONFIGURATION
# --------------------------
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
DATA_PATH = BASE / "data" / "heart_clean.csv"
REPORTS_DIR = BASE / "reports"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# --------------------------
# FLASK APP
# --------------------------
app = Flask(__name__)
app.secret_key = "replace-this-with-env-secret"

# --------------------------
# LOAD METRICS
# --------------------------
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        METRICS = json.load(f)
else:
    METRICS = {}

# --------------------------
# AVAILABLE MODELS
# --------------------------
MODEL_FILES = {
    "Logistic Regression": MODELS_DIR / "Logistic_Regression.joblib",
    "Random Forest": MODELS_DIR / "Random_Forest.joblib",
    "SVM (RBF)": MODELS_DIR / "SVM_RBF.joblib",
    "KNN (k=7)": MODELS_DIR / "KNN_k=7.joblib",
    "Gradient Boosting": MODELS_DIR / "Gradient_Boosting.joblib",
    "Gaussian NB": MODELS_DIR / "Gaussian_NB.joblib",
}

# --------------------------
# LOAD FEATURE COLUMNS
# --------------------------
try:
    df = pd.read_csv(DATA_PATH)
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]
except FileNotFoundError:
    feature_cols = []
    print("⚠️ Dataset not found — please ensure UCI_CLEANED_DATASET.csv exists in /data folder.")

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def load_model(name):
    """Load a trained model by name."""
    path = MODEL_FILES.get(name)
    if not path or not path.exists():
        raise FileNotFoundError(f"Model not found: {name}")
    return joblib.load(path)

# --------------------------
# ROUTES
# --------------------------
@app.route("/", methods=["GET"])
def index():
    """Main page showing metrics, models, and charts."""
    chart_files = []
    img_dir = Path(__file__).resolve().parent / "static" / "img"

    # Add charts if they exist
    for p in sorted(img_dir.glob("bar_*.png")):
        chart_files.append(("Metric Variation", f"img/{p.name}"))
    for p in sorted(img_dir.glob("roc_*.png")):
        chart_files.append(("ROC Curve", f"img/{p.name}"))
    for p in sorted(img_dir.glob("cm_*.png")):
        chart_files.append(("Confusion Matrix", f"img/{p.name}"))

    return render_template(
        "index.html",
        metrics=METRICS,
        models=list(MODEL_FILES.keys()),
        features=feature_cols,
        chart_files=chart_files
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the selected model."""
    patient_id = request.form.get("patient_id", "").strip() or "N/A"
    patient_name = request.form.get("patient_name", "").strip() or "Anonymous"
    model_name = request.form.get("model_name", "Logistic Regression")

    # Build feature vector
    x = []
    for col in feature_cols:
        try:
            x.append(float(request.form.get(col, 0)))
        except Exception:
            x.append(0.0)
    X = np.array(x, dtype=float).reshape(1, -1)

    # Load and predict
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        flash("Selected model not found. Please train models again.", "error")
        return render_template("index.html")

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        prob = 1.0 / (1.0 + np.exp(-score))
    else:
        prob = 0.0

    pred = int(model.predict(X)[0])
    result_text = "Heart Disease: POSITIVE" if pred == 1 else "Heart Disease: NEGATIVE"
    confidence = round(prob * 100.0, 2)

    return render_template(
        "result.html",
        patient_id=patient_id,
        patient_name=patient_name,
        model_name=model_name,
        features=feature_cols,
        inputs=[request.form.get(c, "") for c in feature_cols],
        result=pred,
        result_text=result_text,
        confidence=confidence,
        metrics=METRICS.get(model_name, {}),
        zip=zip
    )


@app.route("/download_report", methods=["POST"])
def download_report():
    """Generate and download a PDF report for a patient."""
    patient_id = request.form.get("patient_id", "").strip() or "N/A"
    patient_name = request.form.get("patient_name", "").strip() or "Anonymous"
    model_name = request.form.get("model_name", "Logistic Regression")
    result_text = request.form.get("result_text", "")
    confidence = request.form.get("confidence", "")

    items = [(col, request.form.get(col, "")) for col in feature_cols]

    # Build PDF in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 20 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Patient Heart Disease Report")
    y -= 10 * mm

    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Patient ID: {patient_id}")
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Patient Name: {patient_name}")
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Model Used: {model_name}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Result: {result_text}  (Confidence: {confidence}%)")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Entered Features:")
    y -= 7 * mm
    c.setFont("Helvetica", 10)

    for key, val in items:
        line = f"{key}: {val}"
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
            c.setFont("Helvetica", 10)
        c.drawString(25 * mm, y, line)
        y -= 6 * mm

    m = METRICS.get(model_name, {})
    if m:
        if y < 40 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20 * mm, y, "Model Performance:")
        y -= 8 * mm
        c.setFont("Helvetica", 10)
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if k in m and m[k] is not None:
                c.drawString(25 * mm, y, f"{k.capitalize()}: {m[k]}")
                y -= 6 * mm

    c.showPage()
    c.save()
    buffer.seek(0)

    filename = f"Patient_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")


# --------------------------
# MAIN ENTRY POINT
# --------------------------
if __name__ == "__main__":
    missing = [k for k, v in MODEL_FILES.items() if not v.exists()]
    if missing:
        print("⚠️ Missing models:", missing)
        print("Run `python train.py` before starting the app.")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
