"""
streamlit_app.py — Frontend for DiabetaCheck API
Connects to FastAPI backend (/predict, /stats)
"""

import streamlit as st
import requests

# FastAPI URL
API_URL = "https://hafsaimranattaria7115-malaikawork.hf.space"
st.set_page_config(page_title="DiabetaCheck", page_icon="🩺", layout="centered")

st.title("🩺 DiabetaCheck - Diabetes Risk Predictor")
st.write("Enter patient details to predict diabetes risk using ML model.")

# ── INPUT FORM ────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 40, 300, 100)
        blood_pressure = st.number_input("Blood Pressure", 20, 140, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 110, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 1000, 80)
        bmi = st.number_input("BMI", 10.0, 80.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.05, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    submitted = st.form_submit_button("Predict")

# ── PREDICTION CALL ──────────────────────────────────────────────────────────
if submitted:
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        data = response.json()

        if response.status_code != 200:
            st.error(data.get("detail", "Error occurred"))
        else:
            st.subheader("Prediction Result")
            st.write(f"**Label:** {data['label']}")
            st.write(f"**Probability:** {data['probability']}%")

            st.subheader("Risk Factors")
            if data["risk_factors"]:
                for r in data["risk_factors"]:
                    st.warning(f"{r['label']} - {r['value']} ({r['level']})")
            else:
                st.success("No major risk factors detected.")

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
