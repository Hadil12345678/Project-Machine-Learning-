import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Cancer Predictor", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "final_pipeline.pkl")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.title("🧬 Patient Data")

# Default values
age = 25
partners = 2
first_sex = 18
pregnancies = 0
smokes = 0
smk_years = 0
smk_packs = 0
hormonal = 0
horm_years = 0
iud = 0
iud_years = 0
stds = 0
stds_num = 0
dx_cancer = 0
dx_cin = 0
dx_hpv = 0
dx = 0
hinselmann = 0
schiller = 0
citology = 0

# Manual controls
age = st.sidebar.slider("Age", 13, 80, age)
partners = st.sidebar.slider("Partners", 0, 20, partners)
first_sex = st.sidebar.slider("First intercourse", 10, 30, first_sex)
pregnancies = st.sidebar.slider("Pregnancies", 0, 10, pregnancies)

smokes = st.sidebar.radio(
    "Smokes",
    [0, 1],
    index=int(smokes),
    format_func=lambda x: "Yes" if x else "No"
)
smk_years = st.sidebar.slider("Smoking years", 0, 30, int(smk_years))
smk_packs = st.sidebar.slider("Packs/year", 0, 20, int(smk_packs))

hormonal = st.sidebar.radio("Hormonal contraceptives", [0, 1], index=int(hormonal))
horm_years = st.sidebar.slider("Years hormonal", 0, 20, int(horm_years))

iud = st.sidebar.radio("IUD", [0, 1], index=int(iud))
iud_years = st.sidebar.slider("Years IUD", 0, 20, int(iud_years))

stds = st.sidebar.radio("STDs", [0, 1], index=int(stds))
stds_num = st.sidebar.slider("Number STDs", 0, 5, int(stds_num))

st.sidebar.markdown("### Clinical tests")
hinselmann = st.sidebar.radio("Hinselmann", [0, 1], index=int(hinselmann))
schiller = st.sidebar.radio("Schiller", [0, 1], index=int(schiller))
citology = st.sidebar.radio("Citology", [0, 1], index=int(citology))

st.sidebar.markdown("### Previous diagnosis")
dx_cancer = st.sidebar.radio("Dx:Cancer", [0, 1], index=int(dx_cancer))
dx_cin = st.sidebar.radio("Dx:CIN", [0, 1], index=int(dx_cin))
dx_hpv = st.sidebar.radio("Dx:HPV", [0, 1], index=int(dx_hpv))
dx = st.sidebar.radio("Dx (any)", [0, 1], index=int(dx))

predict = st.sidebar.button("🔍 Predict")

# Risk label thresholds (can be tuned based on validation behavior)
st.sidebar.markdown("### Risk thresholds")
low_threshold = st.sidebar.slider("Low max threshold", 0.0, 1.0, 0.20, 0.01)
high_threshold = st.sidebar.slider("High min threshold", 0.0, 1.0, 0.90, 0.01)
if low_threshold >= high_threshold:
    st.sidebar.error("Low threshold must be lower than high threshold.")
    st.stop()

# ==============================
# PREDICTION
# ==============================
st.title("🔬 Cervical Cancer Risk Predictor")
st.caption("Educational prototype - not for clinical diagnosis.")

if predict:

    # Create input dataframe with ALL required features
    input_dict = {col: 0 for col in pipeline.feature_names_in_}

    # Explicit mapping for core fields expected by the model
    direct_mapping = {
        "Age": age,
        "Number of sexual partners": partners,
        "First sexual intercourse": first_sex,
        "Num of pregnancies": pregnancies,
        "Smokes": smokes,
        "Smokes (years)": smk_years,
        "Smokes (packs/year)": smk_packs,
        "Hormonal Contraceptives": hormonal,
        "Hormonal Contraceptives (years)": horm_years,
        "IUD": iud,
        "IUD (years)": iud_years,
        "STDs": stds,
        "STDs (number)": stds_num,
        "STDs: Number of diagnosis": stds_num,
        "Dx:Cancer": dx_cancer,
        "Dx:CIN": dx_cin,
        "Dx:HPV": dx_hpv,
        "Dx": dx,
        "Hinselmann": hinselmann,
        "Schiller": schiller,
        "Citology": citology,
    }
    for col, val in direct_mapping.items():
        if col in input_dict:
            input_dict[col] = val

    # Keep unknown STD subtype columns at 0.
    # Setting all STDs:* subtype flags to 1 when STDs=1 overestimates risk.

    # Feature engineering
    if "age_first_sex_gap" in input_dict:
        input_dict["age_first_sex_gap"] = age - first_sex

    if "smoke_exposure" in input_dict:
        input_dict["smoke_exposure"] = smokes * smk_years * max(smk_packs,1)

    if "stds_score" in input_dict:
        input_dict["stds_score"] = stds_num

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Predict
    proba = pipeline.predict_proba(df)[0][1]

    # ==============================
    # RESULT
    # ==============================
    st.subheader("📊 Result")

    risk_label = ""
    if proba < low_threshold:
        risk_label = "Low"
        st.success(f"🟢 Low Risk ({proba:.2%})")
    elif proba < high_threshold:
        risk_label = "Medium"
        st.warning(f"🟡 Medium Risk ({proba:.2%})")
    else:
        risk_label = "High"
        st.error(f"🔴 High Risk ({proba:.2%})")

    st.progress(float(proba))
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Probability", f"{proba:.2%}")
    c2.metric("Risk Level", risk_label)
    c3.metric("Thresholds", f"{low_threshold:.2f} / {high_threshold:.2f}")

    with st.expander("Debug input vector"):
        st.dataframe(df, use_container_width=True)