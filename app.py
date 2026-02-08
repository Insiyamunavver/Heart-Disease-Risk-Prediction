from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# -------------------------------------------------
# Load model & scaler
# -------------------------------------------------
def load_artifacts():
    with open("best_heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# -------------------------------------------------
# Load heart image
# -------------------------------------------------
import os

image_path = os.path.join("assets", "heart_anatomy.png")

heart_image = Image.open(image_path)

# -------------------------------------------------
# Feature names
# -------------------------------------------------
FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain Type", "Blood Pressure", "Cholesterol",
    "Fasting Blood Sugar", "EKG Results", "Max Heart Rate",
    "Exercise Induced Angina", "ST Depression",
    "ST Slope", "Major Vessels", "Thallium Test"
]

# =================================================
# SIDEBAR (IMAGE + RISK SCALE)
# =================================================
with st.sidebar:
    st.markdown("## **Human Heart**")
    st.image(heart_image, use_container_width=True)

    st.markdown("## **Risk Scale**")
    st.info(
        """
        **Low Risk:** 0 – 30%  
        **Moderate Risk:** 31 – 60%  
        **High Risk:** 61 – 100%
        """
    )

    st.markdown("## **Key Risk Factors**")
    st.write(
        """
        • Increasing age  
        • High resting blood pressure  
        • Elevated cholesterol levels  
        • Exercise-induced chest pain  
        • Multiple affected coronary vessels  
        """
    )

# =================================================
# MAIN PAGE
# =================================================
st.markdown(
    """
    <h1 style="text-align:center;"> Heart Disease Risk Prediction</h1>
    <p style="text-align:center; color:gray;">
    Clinical decision-support dashboard for heart disease risk estimation
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
st.subheader("🧍 Patient Information")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
bp = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol (mg/dL)", 100, 600)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
ekg = st.selectbox("EKG Results (0–2)", [0, 1, 2])
max_hr = st.number_input("Max Heart Rate", 60, 220)
ex_ang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression", 0.0, 6.0)

st.subheader("🧪 Advanced Clinical Details")

slope = st.selectbox("Slope of ST Segment", [1, 2, 3])
vessels = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thallium Test Result", [3, 6, 7])

# -------------------------------------------------
# Encode categorical inputs
# -------------------------------------------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
ex_ang = 1 if ex_ang == "Yes" else 0

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.divider()

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
    input_data = np.array([[
        age, sex, cp, bp, chol,
        fbs, ekg, max_hr, ex_ang,
        oldpeak, slope, vessels, thal
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ---------------- RESULT ----------------
    st.subheader("📊 Prediction Result")
    st.progress(int(probability * 100))

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**  \nEstimated Risk: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ **Low Risk of Heart Disease**  \nEstimated Risk: **{probability*100:.2f}%**")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📈 Key Risk-Contributing Factors")

    coef = model.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Impact": coef
    })

    importance_df["Absolute Impact"] = importance_df["Impact"].abs()
    importance_df = importance_df.sort_values(
        "Absolute Impact", ascending=False
    ).head(5)

    st.bar_chart(
        importance_df.set_index("Feature")["Absolute Impact"]
    )

    # ---------------- CLINICAL INTERPRETATION ----------------
    st.subheader("🩺 Clinical Interpretation")

    explanations = []

    if age > 55:
        explanations.append("Advanced age increases cardiovascular risk.")
    if bp > 140:
        explanations.append("Elevated blood pressure suggests hypertension.")
    if chol > 240:
        explanations.append("High cholesterol is a major risk factor.")
    if ex_ang == 1:
        explanations.append("Exercise-induced angina indicates reduced blood flow.")
    if vessels >= 2:
        explanations.append("Multiple affected vessels increase disease severity.")
    if thal == 7:
        explanations.append("Abnormal thallium stress test suggests ischemia.")

    if explanations:
        for e in explanations:
            st.write("•", e)
    else:
        st.write(
            "Clinical parameters are largely within acceptable ranges, "
            "suggesting lower cardiovascular risk."
        )

# ---------


