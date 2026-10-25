"""
app.py
------
Streamlit Web Application for Heart Disease Risk Estimation.

Provides an interactive clinical parameter input form, loads the trained ML
pipeline, predicts risk probability, and displays a categorized risk assessment.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Estimator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "heart_disease_model.joblib"

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #d90429;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_estimator():
    """Loads the pre-trained model pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        loaded = joblib.load(MODEL_PATH)
        # Check if saved as dictionary or bare pipeline
        if isinstance(loaded, dict) and "pipeline" in loaded:
            return loaded["pipeline"]
        return loaded
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ---------------------------------------------------------
# Sidebar: Presets and Information
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls & Presets")
    st.write("Populate test values quickly to test the model:")

    if "preset" not in st.session_state:
        st.session_state.preset = "default"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 Low Risk Case"):
            st.session_state.preset = "low_risk"
            st.rerun()
    with col_btn2:
        if st.button("🔴 High Risk Case"):
            st.session_state.preset = "high_risk"
            st.rerun()

    st.markdown("---")
    st.subheader("ℹ️ About the Model")
    st.markdown("""
    - **Algorithm:** Logistic Regression with Standard Scaler
    - **Dataset:** UCI Cleveland Heart Disease benchmark
    - **Target:** Probability of significant heart disease (narrowing > 50%)
    """)

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is for educational & screening demonstration purposes only. It is not a substitute for professional clinical medical advice.")


# Default input values based on preset selection
preset_values = {
    "default": {
        "age": 52, "sex": "Male", "cp": "Typical Angina (0)", "trestbps": 125,
        "chol": 212, "fbs": "No (<= 120 mg/dl)", "restecg": "Normal (0)",
        "thalach": 155, "exang": "No", "oldpeak": 0.8, "slope": "Upsloping (0)",
        "ca": 0, "thal": "Normal (1)"
    },
    "low_risk": {
        "age": 36, "sex": "Female", "cp": "Non-anginal Pain (2)", "trestbps": 110,
        "chol": 175, "fbs": "No (<= 120 mg/dl)", "restecg": "Normal (0)",
        "thalach": 178, "exang": "No", "oldpeak": 0.0, "slope": "Upsloping (0)",
        "ca": 0, "thal": "Normal (1)"
    },
    "high_risk": {
        "age": 64, "sex": "Male", "cp": "Asymptomatic (3)", "trestbps": 160,
        "chol": 290, "fbs": "Yes (> 120 mg/dl)", "restecg": "ST-T Wave Abnormality (1)",
        "thalach": 115, "exang": "Yes", "oldpeak": 2.8, "slope": "Downsloping (2)",
        "ca": 2, "thal": "Reversible Defect (3)"
    }
}
current_vals = preset_values.get(st.session_state.preset, preset_values["default"])


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">❤️ Heart Disease Risk Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient clinical metrics below to estimate the statistical probability of heart disease.</div>', unsafe_allow_html=True)

model = load_estimator()

if model is None:
    st.warning("⚠️ **Trained model not found (`heart_disease_model.joblib`).**")
    st.info("Please train the model first by running `python train.py` in your terminal.")
    if st.button("🚀 Train Model Now via App"):
        with st.spinner("Training model with train.py..."):
            import train
            train.main()
            st.cache_resource.clear()
            st.success("Model successfully trained and saved!")
            st.rerun()
    st.stop()


# ---------------------------------------------------------
# Interactive Patient Input Form
# ---------------------------------------------------------
with st.form("patient_data_form"):
    st.subheader("📋 Patient Clinical Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Demographics & Vitals")
        age = st.slider("Age (years)", min_value=20, max_value=90, value=current_vals["age"])
        sex_str = st.selectbox("Sex", ["Male", "Female"], index=0 if current_vals["sex"] == "Male" else 1)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=current_vals["trestbps"])
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=current_vals["chol"])
        fbs_str = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"], 
                           index=0 if "No" in current_vals["fbs"] else 1)

    with col2:
        st.markdown("##### 🫀 Symptoms & Stress Test")
        cp_options = [
            "Typical Angina (0)", 
            "Atypical Angina (1)", 
            "Non-anginal Pain (2)", 
            "Asymptomatic (3)"
        ]
        cp_index = cp_options.index(current_vals["cp"]) if current_vals["cp"] in cp_options else 0
        cp_str = st.selectbox("Chest Pain Type (cp)", cp_options, index=cp_index)

        thalach = st.slider("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=current_vals["thalach"])
        exang_str = st.radio("Exercise-Induced Angina", ["No", "Yes"], index=0 if current_vals["exang"] == "No" else 1)
        oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, step=0.1, value=float(current_vals["oldpeak"]))

    with col3:
        st.markdown("##### 🔬 Diagnostic Findings")
        restecg_options = [
            "Normal (0)", 
            "ST-T Wave Abnormality (1)", 
            "Left Ventricular Hypertrophy (2)"
        ]
        restecg_index = restecg_options.index(current_vals["restecg"]) if current_vals["restecg"] in restecg_options else 0
        restecg_str = st.selectbox("Resting ECG Results", restecg_options, index=restecg_index)

        slope_options = [
            "Upsloping (0)", 
            "Flat (1)", 
            "Downsloping (2)"
        ]
        slope_index = slope_options.index(current_vals["slope"]) if current_vals["slope"] in slope_options else 0
        slope_str = st.selectbox("Slope of Peak Exercise ST", slope_options, index=slope_index)

        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3], index=current_vals["ca"])

        thal_options = [
            "Normal (1)", 
            "Fixed Defect (2)", 
            "Reversible Defect (3)"
        ]
        thal_index = thal_options.index(current_vals["thal"]) if current_vals["thal"] in thal_options else 0
        thal_str = st.selectbox("Thalassemia (thal)", thal_options, index=thal_index)

    submit_button = st.form_submit_button("🔍 Calculate Risk Assessment", type="primary", use_container_width=True)


# ---------------------------------------------------------
# Feature Encoding & Inference
# ---------------------------------------------------------
if submit_button:
    # Map UI selections to exact numerical model features
    sex = 1 if sex_str == "Male" else 0
    cp = int(cp_str.split("(")[-1].replace(")", ""))
    fbs = 1 if "Yes" in fbs_str else 0
    restecg = int(restecg_str.split("(")[-1].replace(")", ""))
    exang = 1 if exang_str == "Yes" else 0
    slope = int(slope_str.split("(")[-1].replace(")", ""))
    thal = int(thal_str.split("(")[-1].replace(")", ""))

    feature_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([feature_dict])

    # Run inference
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        risk_percentage = probabilities[1] * 100

        st.markdown("---")
        st.subheader("📊 Assessment Result")

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            st.metric(
                label="Estimated Risk Probability",
                value=f"{risk_percentage:.1f}%",
                delta=f"{'High' if risk_percentage >= 50 else 'Low'} Probability"
            )
            st.progress(risk_percentage / 100.0)

            if risk_percentage < 35:
                st.success("### 🟢 Low Risk of Heart Disease")
                st.write("Clinical indicators suggest a low likelihood of coronary disease under current baseline metrics.")
            elif 35 <= risk_percentage < 65:
                st.warning("### 🟡 Moderate Risk of Heart Disease")
                st.write("Borderline clinical findings present. Recommend closer observation and standard preventive lifestyle management.")
            else:
                st.error("### 🔴 High Risk of Heart Disease")
                st.write("Significant coronary indicators detected. Recommend prompt diagnostic follow-up by a cardiologist.")

        with res_col2:
            st.markdown("#### Key Factor Highlights")
            flags = []
            if trestbps >= 140:
                flags.append(f"• **Elevated Blood Pressure**: {trestbps} mm Hg (Stage 2 Hypertension range)")
            if chol >= 240:
                flags.append(f"• **High Cholesterol**: {chol} mg/dl (> 240 mg/dl guideline threshold)")
            if exang == 1:
                flags.append("• **Exercise-Induced Angina**: Present")
            if oldpeak >= 2.0:
                flags.append(f"• **Significant ST Depression**: {oldpeak} mm during exercise test")
            if ca > 0:
                flags.append(f"• **Fluoroscopy Findings**: {ca} major vessel(s) flagged")
            if thal == 3:
                flags.append("• **Thalassemia**: Reversible defect indicated")

            if flags:
                for f in flags:
                    st.write(f)
            else:
                st.write("• No major red flags detected across key cardiovascular baseline metrics.")

            with st.expander("🔎 View Transformed Feature Vector"):
                st.dataframe(input_df)

    except Exception as e:
        st.error(f"Inference error: {e}")
