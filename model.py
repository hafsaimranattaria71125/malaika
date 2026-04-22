"""
predictor.py — Diabetes model predictor (FastAPI-compatible version in structured loader style)
"""

import os
import numpy as np
import joblib
from fastapi import HTTPException

from schemas import PredictResponse, RiskFactor

# ── Features & Limits (UNCHANGED) ─────────────────────────────────────────────
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

LIMITS = {
    "Pregnancies":              (0,    20),
    "Glucose":                  (40,   300),
    "BloodPressure":            (20,   140),
    "SkinThickness":            (0,    110),
    "Insulin":                  (0,    1000),
    "BMI":                      (10.0, 80.0),
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age":                      (1,    120),
}

# ── Artifact Loader (same style as your second file) ─────────────────────────
_model = None
_scaler = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

def load_artifacts():
    global _model, _scaler

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)

    return _model, _scaler


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predictAns(body):
    """
    Predict diabetes risk (FastAPI endpoint compatible).
    """

    model, scaler = load_artifacts()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Extract features (UNCHANGED logic)
    feats = [body[f] for f in FEATURES]

    # ── Range validation (UNCHANGED) ─────────────────────────────────────────
    for i, col in enumerate(FEATURES):
        lo, hi = LIMITS[col]
        if not (lo <= feats[i] <= hi):
            raise HTTPException(
                status_code=422,
                detail=f"{col} = {feats[i]} is out of valid range [{lo}, {hi}].",
            )

    # ── Scaling & Prediction ────────────────────────────────────────────────
    X = np.array(feats).reshape(1, -1)
    Xs = scaler.transform(X)

    pred = int(model.predict(Xs)[0])
    prob = float(model.predict_proba(Xs)[0][1])

    # ── Risk Factor Analysis (UNCHANGED LOGIC) ─────────────────────────────
    risks: list[RiskFactor] = []

    if feats[1] > 125:
        risks.append(RiskFactor(label="High Glucose", value=f"{int(feats[1])} mg/dL", level="high"))
    elif feats[1] > 100:
        risks.append(RiskFactor(label="Borderline Glucose", value=f"{int(feats[1])} mg/dL", level="medium"))

    if feats[5] > 30:
        risks.append(RiskFactor(label="High BMI", value=str(feats[5]), level="high"))
    elif feats[5] > 25:
        risks.append(RiskFactor(label="Overweight BMI", value=str(feats[5]), level="medium"))

    if feats[7] > 45:
        risks.append(RiskFactor(label="Advanced Age", value=f"{int(feats[7])} yrs", level="high"))
    elif feats[7] > 35:
        risks.append(RiskFactor(label="Age Factor", value=f"{int(feats[7])} yrs", level="medium"))

    if feats[6] > 0.8:
        risks.append(RiskFactor(label="High Pedigree Score", value=str(feats[6]), level="high"))

    if feats[0] > 5:
        risks.append(RiskFactor(label="Multiple Pregnancies", value=str(int(feats[0])), level="medium"))

    # ── Response (UNCHANGED) ────────────────────────────────────────────────
    return {
    "prediction": pred,
    "probability": round(prob * 100, 1),
    "label": "Diabetic" if pred == 1 else "Non-Diabetic",
    "risk_factors": risks,}
