"""
╔══════════════════════════════════════════════════════════════════╗
║  Cervical Cancer Risk Predictor — Dashboard Streamlit v3         ║
║  Hadil Dhaya · 4th Year Data Science · Group 5 · 2026           ║
╚══════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import shap 

st.title("🔬 Cervical Cancer Prediction System")

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cervical Cancer · Risk Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ═══════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }
.stApp { background: #080c12; }
.main  { background: #080c12; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2530; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.kpi-card {
    background: #0d1117; border: 1px solid #1e2530;
    border-radius: 12px; padding: 18px 20px 14px;
    text-align: center; position: relative; overflow: hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0;
    height:2px; background:linear-gradient(90deg,#58a6ff,#3fb950);
}
.kpi-val { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:700; color:#e6edf3; line-height:1; }
.kpi-lbl { font-size:0.72rem; color:#8b949e; letter-spacing:0.1em; text-transform:uppercase; margin-top:6px; }
.kpi-sub { font-size:0.7rem; color:#3fb950; margin-top:4px; font-family:'IBM Plex Mono',monospace; }
.section-header {
    font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#8b949e;
    letter-spacing:0.15em; text-transform:uppercase;
    border-bottom:1px solid #1e2530; padding-bottom:8px; margin:24px 0 16px;
}
.risk-critical {
    background:linear-gradient(135deg,#1a0a0a,#2d1515);
    border:1px solid #da3633; border-left:4px solid #da3633;
    border-radius:12px; padding:24px 28px;
}
.risk-safe {
    background:linear-gradient(135deg,#0a1a0e,#0f2918);
    border:1px solid #238636; border-left:4px solid #3fb950;
    border-radius:12px; padding:24px 28px;
}
.risk-title { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; }
.risk-prob  { font-family:'IBM Plex Mono',monospace; font-size:2.8rem; font-weight:600; line-height:1; margin:8px 0; }
.risk-note  { font-size:0.82rem; color:#8b949e; margin-top:10px; line-height:1.5; }
.box-info   { background:#0c1b2e; border-left:3px solid #58a6ff; border-radius:6px;
               padding:12px 16px; font-size:0.84rem; color:#79c0ff; margin:10px 0; line-height:1.6; }
.box-warn   { background:#1c1600; border-left:3px solid #d29922; border-radius:6px;
               padding:12px 16px; font-size:0.84rem; color:#e3b341; margin:10px 0; line-height:1.6; }
.box-danger { background:#1c0a0a; border-left:3px solid #da3633; border-radius:6px;
               padding:12px 16px; font-size:0.84rem; color:#ff7b72; margin:10px 0; line-height:1.6; }
.box-success{ background:#0f2918; border-left:3px solid #3fb950; border-radius:6px;
               padding:12px 16px; font-size:0.84rem; color:#56d364; margin:10px 0; line-height:1.6; }
.factor-pill { display:inline-block; padding:4px 10px; border-radius:20px;
                font-size:0.78rem; font-weight:500; margin:3px 3px 3px 0; }
.pill-red    { background:#2d1515; color:#ff7b72; border:1px solid #da3633; }
.pill-amber  { background:#1c1400; color:#e3b341; border:1px solid #9e6a03; }
.pill-green  { background:#0f2918; color:#56d364; border:1px solid #238636; }
.stTabs [data-baseweb="tab-list"] {
    background:#0d1117; border-radius:8px; padding:4px;
    border:1px solid #1e2530; gap:2px;
}
.stTabs [data-baseweb="tab"] { border-radius:6px; color:#8b949e; font-size:13px; padding:6px 14px; }
.stTabs [aria-selected="true"] { background:#1e2530 !important; color:#e6edf3 !important; }
hr { border-color:#1e2530 !important; }
.backend-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:#0d1117; border:1px solid; border-radius:20px;
    padding:4px 12px; font-size:0.75rem;
    font-family:'IBM Plex Mono',monospace;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════
BASE    = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "outputs"
DATA_P  = BASE / "data" / "risk_factors_cervical_cancer.csv"
def op(f): return OUTPUTS / f

# ═══════════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the FULL sklearn pipeline (imputer + scaler + model)."""
    path = op("final_pipeline.pkl")
    if path.exists():
        return joblib.load(path)

    alt_path = BASE / "frontend" / "outputs" / "final_pipeline.pkl"
    if alt_path.exists():
        return joblib.load(alt_path)

    cwd_path = Path.cwd() / "outputs" / "final_pipeline.pkl"
    if cwd_path.exists():
        return joblib.load(cwd_path)

    return None

@st.cache_data(show_spinner=False)
def load_test_data():
    xp, yp = op("X_test.csv"), op("y_test.csv")
    if xp.exists() and yp.exists():
        return pd.read_csv(xp), pd.read_csv(yp).squeeze()
    return None, None

@st.cache_data(show_spinner=False)
def load_raw():
    if DATA_P.exists():
        return pd.read_csv(DATA_P, na_values="?")
    return None


@st.cache_data(show_spinner=False)
def load_raw_feature_medians():
    """
    Column medians on the original UCI CSV (raw scale).
    outputs/X_test.csv is often StandardScaler output (z-scores) — do NOT use it as a baseline
    for sliders that collect raw ages/counts; mixing z-scores + raw values breaks predict_proba.
    """
    df = load_raw()
    if df is None:
        return None
    X = df.drop(columns=["Biopsy"], errors="ignore")
    return X.median(numeric_only=True)

@st.cache_data(show_spinner=False)
def load_model_name():
    p = op("best_model_name.pkl")
    return joblib.load(p) if os.path.exists(p) else "Pipeline"

def load_img(fname):
    p = op(fname)
    if os.path.exists(p) and PIL_OK:
        return Image.open(p)
    return None

pipeline   = load_pipeline()
X_test, y_test = load_test_data()
df_raw     = load_raw()
model_name = load_model_name()

# ─── Get the exact feature names the pipeline was trained on ───────
# This is the GROUND TRUTH — everything must match this list exactly
PIPELINE_FEATURES = None
if pipeline is not None and hasattr(pipeline, "feature_names_in_"):
    PIPELINE_FEATURES = list(pipeline.feature_names_in_)
elif X_test is not None:
    PIPELINE_FEATURES = list(X_test.columns)

# ═══════════════════════════════════════════════════════════════════
#  METRICS (computed against the pipeline directly)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_metrics(_pipeline, _X_test, _y_test):
    if _pipeline is None or _X_test is None or _y_test is None:
        return None
    try:
        from sklearn.metrics import (
            roc_auc_score, f1_score, recall_score, precision_score,
            accuracy_score, confusion_matrix, roc_curve,
            precision_recall_curve, average_precision_score,
        )
        # Align columns to what the pipeline was fit on (CSV may differ from training)
        expected = getattr(_pipeline, "feature_names_in_", None)
        if expected is not None:
            _X_test = _X_test.reindex(columns=list(expected), fill_value=0)
        # Pipeline takes RAW X_test (imputer + scaler + model inside)
        y_pred = _pipeline.predict(_X_test)
        y_prob = _pipeline.predict_proba(_X_test)[:, 1]
        fpr, tpr, _ = roc_curve(_y_test, y_prob)
        prec, rec, _ = precision_recall_curve(_y_test, y_prob)
        cm = confusion_matrix(_y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "auc": roc_auc_score(_y_test, y_prob),
            "f1": f1_score(_y_test, y_pred, zero_division=0),
            "recall": recall_score(_y_test, y_pred, zero_division=0),
            "precision": precision_score(_y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(_y_test, y_pred),
            "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fpr": fpr, "tpr": tpr,
            "prec_curve": prec, "rec_curve": rec,
            "ap": average_precision_score(_y_test, y_prob),
            "y_pred": y_pred, "y_prob": y_prob, "y_test": _y_test,
        }
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        return None

metrics = compute_metrics(pipeline, X_test, y_test)

# ═══════════════════════════════════════════════════════════════════
#  PREDICTION — uses final_pipeline.pkl with CORRECT feature names
# ═══════════════════════════════════════════════════════════════════
def build_input_df(patient: dict) -> pd.DataFrame:
    """
    Build a DataFrame with exactly the columns the pipeline was trained on.

    Strategy:
      1. Baseline = medians from the **raw** UCI CSV (same scale as training before the pipeline).
         Never use X_test.csv medians here if that file is z-scored (common after notebook export).
      2. Overwrite columns collected in the UI (raw ages, counts, 0/1 flags).
      3. Engineered columns only if present in PIPELINE_FEATURES.
    """
    if PIPELINE_FEATURES is None:
        raise RuntimeError("Cannot determine pipeline features. Check outputs/final_pipeline.pkl")

    raw_med = load_raw_feature_medians()
    if raw_med is not None:
        baseline = raw_med.reindex(PIPELINE_FEATURES).fillna(0.0)
    else:
        baseline = pd.Series(0.0, index=PIPELINE_FEATURES)

    row = {col: float(baseline[col]) for col in PIPELINE_FEATURES}

    # ── Step 2: map UI inputs → exact column names ───────────────────
    # These names must match what the model saw during fit EXACTLY
    t_first = (
        float(baseline["STDs: Time since first diagnosis"])
        if "STDs: Time since first diagnosis" in baseline.index
        else 0.0
    )
    t_last = (
        float(baseline["STDs: Time since last diagnosis"])
        if "STDs: Time since last diagnosis" in baseline.index
        else 0.0
    )

    ui_map = {
        "Age":                              float(patient.get("age", 25)),
        "Number of sexual partners":        float(patient.get("partners", 2)),
        "First sexual intercourse":         float(patient.get("first_sex", 18)),
        "Num of pregnancies":               float(patient.get("pregnancies", 0)),
        "Smokes":                           float(patient.get("smokes", 0)),
        "Smokes (years)":                   float(patient.get("smk_years", 0)),
        "Smokes (packs/year)":              float(patient.get("smk_packs", 0)),
        "Hormonal Contraceptives":          float(patient.get("hormonal", 0)),
        "Hormonal Contraceptives (years)":  float(patient.get("horm_years", 0)),
        "IUD":                              float(patient.get("iud", 0)),
        "IUD (years)":                      float(patient.get("iud_years", 0)),
        "STDs":                             float(patient.get("stds", 0)),
        "STDs (number)":                    float(patient.get("stds_num", 0)),
        "STDs: Number of diagnosis":        float(patient.get("stds_num", 0)),
        "Hinselmann":                       float(patient.get("hinselmann", 0)),
        "Schiller":                         float(patient.get("schiller", 0)),
        "Citology":                         float(patient.get("citology", 0)),
        "Dx:Cancer":                        float(patient.get("dx_cancer", 0)),
        "Dx:CIN":                           float(patient.get("dx_cin", 0)),
        "Dx:HPV":                           float(patient.get("dx_hpv", 0)),
        "Dx":                               float(patient.get("dx_any", 0)),
        "STDs: Time since first diagnosis": 0.0 if patient.get("stds", 0) == 0 else t_first,
        "STDs: Time since last diagnosis":  0.0 if patient.get("stds", 0) == 0 else t_last,
    }

    for col, val in ui_map.items():
        if col in row:
            row[col] = val

    # ── Step 3: engineered features — ONLY if the pipeline uses them ─
    a  = float(patient.get("age", 25))
    fs = float(patient.get("first_sex", 18))
    sm = float(patient.get("smokes", 0))
    sy = float(patient.get("smk_years", 0))
    sp = float(patient.get("smk_packs", 0))
    sn = float(patient.get("stds_num", 0))

    if "age_first_sex_gap" in row:
        row["age_first_sex_gap"] = a - fs
    if "smoke_exposure" in row:
        row["smoke_exposure"] = sm * sy * max(sp, 1.0)
    if "stds_score" in row:
        row["stds_score"] = sn

    # ── Step 4: return DataFrame with columns in the EXACT trained order
    return pd.DataFrame([row], columns=PIPELINE_FEATURES)


def run_prediction(patient: dict) -> dict | None:
    if pipeline is None:
        st.error("Pipeline not loaded. Check outputs/final_pipeline.pkl")
        return None
    try:
        df = build_input_df(patient)
        # Pipeline handles imputation + scaling internally
        prob = float(pipeline.predict_proba(df)[0][1])
        return {
            "probability": prob,
            "risk_level":  "HIGH" if prob >= 0.5 else "LOW",
            "source":      "final_pipeline.pkl",
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback; st.code(traceback.format_exc())
        return None
def compute_shap_values(input_df):
    import shap

    try:
        if pipeline is None:
            return None, None
        # Last step is the classifier / regressor (names vary across notebooks)
        model = pipeline.named_steps.get("model")
        if model is None:
            model = pipeline.steps[-1][1]

        X_transformed = pipeline[:-1].transform(input_df)

        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model, X_transformed)

        shap_values = explainer(X_transformed)

        return shap_values, input_df.columns

    except Exception as e:
        st.warning(f"SHAP error: {e}")
        return None, None

# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Patient data")
    st.markdown("---")

    age         = st.slider("Age", 13, 84, 25)
    partners    = st.slider("Sexual partners", 0, 28, 2)
    first_sex   = st.slider("Age first intercourse", 10, 32, 18)
    pregnancies = st.slider("Pregnancies", 0, 11, 0)

    st.markdown("**Smoking**")
    smokes    = st.radio("Smokes?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    smk_years = st.slider("Smoking years", 0.0, 37.0, 0.0, 0.5) if smokes else 0.0
    smk_packs = st.slider("Packs/year",    0.0, 20.0, 0.0, 0.5) if smokes else 0.0

    st.markdown("**Contraception**")
    hormonal   = st.radio("Hormonal contraceptives", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    horm_years = st.slider("Contraceptive years", 0.0, 30.0, 0.0, 0.5) if hormonal else 0.0
    iud        = st.radio("IUD", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    iud_years  = st.slider("IUD years", 0.0, 20.0, 0.0, 0.5) if iud else 0.0

    st.markdown("**STDs**")
    stds     = st.radio("STDs diagnosed", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    stds_num = st.slider("Number of STDs", 0, 8, 0) if stds else 0

    st.markdown("**Screening / prior diagnosis**")
    st.caption("These drive the model strongly — set if applicable.")
    hinselmann = schiller = citology = 0
    dx_cancer = dx_cin = dx_hpv = dx_any = 0
    with st.expander("Clinical tests & Dx (0/1)", expanded=False):
        hinselmann = st.radio("Hinselmann", [0, 1], format_func=lambda x: "Pos" if x else "Neg", horizontal=True, key="hin")
        schiller   = st.radio("Schiller", [0, 1], format_func=lambda x: "Pos" if x else "Neg", horizontal=True, key="sch")
        citology   = st.radio("Citology", [0, 1], format_func=lambda x: "Abn" if x else "Norm", horizontal=True, key="cit")
        dx_cancer  = st.radio("Dx: Cancer", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True, key="dxc")
        dx_cin     = st.radio("Dx: CIN", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True, key="dxci")
        dx_hpv     = st.radio("Dx: HPV", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True, key="dxhp")
        dx_any     = st.radio("Dx (any)", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True, key="dxa")

    st.markdown("---")
    predict_btn = st.button("🧬 Analyse risk", width="stretch", type="primary")

    st.markdown("""
    <div class='box-warn' style='font-size:0.75rem'>
        Academic use only. Not medical advice.
    </div>""", unsafe_allow_html=True)

    # Debug: show which features the pipeline uses
    if PIPELINE_FEATURES:
        with st.expander("🔍 Debug — pipeline features"):
            st.write(f"**{len(PIPELINE_FEATURES)} features:**")
            st.write(PIPELINE_FEATURES)

# ─── Patient dict ──────────────────────────────────────────────────
patient_dict = {
    "age": age, "partners": partners, "first_sex": first_sex,
    "pregnancies": pregnancies, "smokes": smokes,
    "smk_years": smk_years, "smk_packs": smk_packs,
    "hormonal": hormonal, "horm_years": horm_years,
    "iud": iud, "iud_years": iud_years,
    "stds": stds, "stds_num": stds_num,
    "hinselmann": hinselmann, "schiller": schiller, "citology": citology,
    "dx_cancer": dx_cancer, "dx_cin": dx_cin, "dx_hpv": dx_hpv, "dx_any": dx_any,
}

# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("""
    <h1 style='color:#e6edf3;font-size:2rem;margin-bottom:0;font-family:Syne,sans-serif'>
        🔬 Cervical Cancer Risk Predictor
    </h1>
    <p style='color:#8b949e;font-size:0.88rem;margin-top:4px'>
        Hadil Dhaya &nbsp;·&nbsp; 4th Year Data Science &nbsp;·&nbsp; Group 5 &nbsp;·&nbsp; 2026
    </p>""", unsafe_allow_html=True)
with c2:
    mc = "#3fb950" if pipeline is not None else "#da3633"
    mt = "● Pipeline loaded" if pipeline is not None else "● Pipeline missing"
    st.markdown(f"""
    <div style='text-align:right;padding-top:14px'>
        <span class='backend-badge' style='border-color:{mc};color:{mc}'>{mt}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
#  KPI ROW
# ═══════════════════════════════════════════════════════════════════
if metrics:
    cols = st.columns(5)
    kpis = [
        ("AUC-ROC",   f"{metrics['auc']:.4f}",      "ideal = 1.0"),
        ("F1-Score",  f"{metrics['f1']:.4f}",        "Precision × Recall"),
        ("Recall",    f"{metrics['recall']:.4f}",    "key metric"),
        ("Precision", f"{metrics['precision']:.4f}", "TP / (TP+FP)"),
        ("Accuracy",  f"{metrics['accuracy']:.4f}",  model_name[:18]),
    ]
    for col, (lbl, val, sub) in zip(cols, kpis):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-val'>{val}</div>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values(): s.set_edgecolor("#1e2530")
    return fig, ax

t1, t2, t3, t4, t5, t6 = st.tabs([
    "🧪 Prediction", "🤖 AI Assistant",
    "📊 Models", "🔍 SHAP", "📈 EDA", "📋 About",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────────────────────────────────────────
with t1:
    st.markdown("<div class='section-header'>Real-time prediction</div>", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Running prediction..."):
            result = run_prediction(patient_dict)

        if result:
            prob   = result["probability"]
            level  = result["risk_level"]
            source = result["source"]
            pred   = int(prob >= 0.5)

            # Risk factors
            factors = []
            if age > 40:                    factors.append(("🔴", "Age > 40",              "pill-red"))
            if smokes:                      factors.append(("🔴", f"Smoking {smk_years:.0f}y","pill-red"))
            if stds:                        factors.append(("🔴", f"STDs ({stds_num})",     "pill-red"))
            if first_sex < 16:              factors.append(("🟡", "1st intercourse < 16",   "pill-amber"))
            if partners > 5:                factors.append(("🟡", f"{partners} partners",   "pill-amber"))
            if pregnancies > 3:             factors.append(("🟡", f"{pregnancies} preg.",   "pill-amber"))
            if hormonal and horm_years > 5: factors.append(("🟡", f"Contraceptives {horm_years:.0f}y","pill-amber"))
            if not factors:                 factors.append(("🟢", "No major risk factors",  "pill-green"))

            st.session_state.last_prediction = {
                "probability": float(prob),
                "risk_level": level,
                "risk_factors": [l for _, l, _ in factors],
            }

            ca, cb = st.columns([1, 1], gap="large")
            with ca:
                if pred == 1:
                    st.markdown(f"""
                    <div class='risk-critical'>
                        <div class='risk-title' style='color:#ff7b72'>HIGH RISK</div>
                        <div class='risk-prob' style='color:#da3633'>{prob:.1%}</div>
                        <div style='color:#ff7b72;font-size:.9rem;font-weight:500'>Probability Biopsy = 1</div>
                        <div class='risk-note'>Significant risk factors detected.<br>Medical consultation recommended.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='risk-safe'>
                        <div class='risk-title' style='color:#3fb950'>LOW RISK</div>
                        <div class='risk-prob' style='color:#238636'>{prob:.1%}</div>
                        <div style='color:#56d364;font-size:.9rem;font-weight:500'>Probability Biopsy = 1</div>
                        <div class='risk-note'>No major signs detected.<br>Regular gynecological follow-up recommended.</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(min(float(prob), 1.0))
                st.markdown(f"""
                <p style='text-align:center;font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#8b949e'>
                    score={prob:.6f} · threshold=0.5 · source={source}
                </p>""", unsafe_allow_html=True)

                st.markdown("""
                <div class='box-info' style='margin-top:16px'>
                    Go to the <b>AI Assistant</b> tab to get a plain-language explanation of this result.
                </div>""", unsafe_allow_html=True)

            with cb:
                st.markdown("#### Risk factors identified")
                pills = "".join([f"<span class='factor-pill {c}'>{i} {l}</span>" for i, l, c in factors])
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Patient summary")
                # All string columns — mixed int/str breaks PyArrow when Streamlit serializes the dataframe
                _sum = pd.DataFrame({
                    "Parameter": ["Age", "Partners", "1st intercourse", "Pregnancies",
                                  "Smoking", "Contraceptives", "IUD", "STDs"],
                    "Value": [
                        str(int(age)),
                        str(int(partners)),
                        str(int(first_sex)),
                        str(int(pregnancies)),
                        f"Yes ({smk_years}y)" if smokes else "No",
                        f"Yes ({horm_years}y)" if hormonal else "No",
                        f"Yes ({iud_years}y)" if iud else "No",
                        f"Yes ({int(stds_num)})" if stds else "No",
                    ],
                }).astype({"Parameter": "object", "Value": "object"})
                st.dataframe(_sum, hide_index=True, width="stretch")
                st.markdown(
                    "<p style='font-size:0.72rem;color:#8b949e;margin-top:8px'>"
                    "The probability uses <b>all</b> pipeline inputs (sidebar, clinical expander, and default medians from the UCI CSV for columns not shown). "
                    "Two profiles can both be HIGH risk with different %.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("""
                <div class='box-danger'>
                    Model calibrated to maximise <b>Recall</b> — minimising missed cancers (FN).
                </div>""", unsafe_allow_html=True)

            st.markdown("### 🔍 Why this prediction? (Explainable AI)")
            input_df = build_input_df(patient_dict)
            shap_values, feature_names = compute_shap_values(input_df)

            if shap_values is not None and feature_names is not None:
                try:
                    fig_w, _ = plt.subplots()
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig_w, width="stretch")
                    plt.close(fig_w)
                except Exception:
                    pass

                st.markdown("#### 🔝 Top factors influencing prediction:")
                values = shap_values.values[0]
                impacts = sorted(
                    zip(feature_names, values),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                for name, val in impacts[:5]:
                    if val > 0:
                        st.write(f"🔴 **{name}** increases risk (+{val:.4f})")
                    else:
                        st.write(f"🟢 **{name}** decreases risk ({val:.4f})")
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#8b949e'>
            <div style='font-size:3rem'>🧬</div>
            <div style='font-family:Syne,sans-serif;font-size:1.2rem;margin:12px 0;color:#c9d1d9'>
                Fill in patient data in the sidebar
            </div>
            <div style='font-size:.88rem'>then click <b style='color:#58a6ff'>Analyse risk</b></div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — AI ASSISTANT (built-in, no external chatbot.py needed)
# ──────────────────────────────────────────────────────────────────
with t2:
    st.markdown("<div class='section-header'>AI medical assistant</div>", unsafe_allow_html=True)

    if st.session_state.last_prediction:
        p  = st.session_state.last_prediction
        bc = "#da3633" if p["risk_level"] == "HIGH" else "#238636"
        st.markdown(f"""
        <div style='background:#0d1117;border:1px solid {bc};border-radius:8px;
                    padding:10px 16px;font-size:0.82rem;margin-bottom:12px'>
            <span style='color:{bc};font-family:IBM Plex Mono,monospace'>
                Patient context loaded: {p["risk_level"]} risk ({p["probability"]:.1%})
            </span>
            <span style='color:#8b949e;font-size:0.75rem'>
                — The AI knows your prediction result
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='box-warn'>
            Run a prediction first (Tab 1) to give the assistant your result context.
        </div>""", unsafe_allow_html=True)

    # Quick prompt buttons
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("What does my result mean?"):
        st.session_state.chat_history.append({"role": "user", "content": "What does my result mean?"})
    if col_b.button("What should I do next?"):
        st.session_state.chat_history.append({"role": "user", "content": "What should I do next?"})
    if col_c.button("What risk factors affect my score?"):
        st.session_state.chat_history.append({"role": "user", "content": "Which risk factors affect my score the most?"})

    # Chat display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask about your results, risk factors, or next steps..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build context for the AI
        ctx = ""
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            ctx = (f"Patient assessed as {p['risk_level']} risk with probability "
                   f"{p['probability']:.1%}. Key factors: {', '.join(p['risk_factors'])}. ")

        # Try Gemini first (free), fallback to rule-based
        GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
        reply = ""

        if GEMINI_KEY:
            try:
                import google.generativeai as genai  # pyright: ignore[reportMissingImports]
                genai.configure(api_key=GEMINI_KEY)
                model_ai = genai.GenerativeModel("gemini-1.5-flash")
                system = (
                    "You are a compassionate medical assistant for a cervical cancer risk screening app. "
                    + ctx +
                    "Rules: be empathetic, explain in plain language, always recommend consulting a real doctor, "
                    "keep answers to 2-3 short paragraphs, never diagnose or prescribe."
                )
                response = model_ai.generate_content(system + "\n\nUser: " + user_input)
                reply = response.text
            except Exception as e:
                reply = f"[Gemini error: {e}] "

        if not reply:
            # Rule-based fallback — no API key needed
            u = user_input.lower()
            if any(w in u for w in ["high", "critical", "serious", "worried"]):
                reply = ("A high-risk result means the model detected several significant risk factors. "
                         "This is NOT a cancer diagnosis — it means you should see a gynaecologist soon (within a few weeks). "
                         "They may recommend a Pap smear or colposcopy. Early detection makes a huge difference.")
            elif any(w in u for w in ["low", "safe", "good", "fine"]):
                reply = ("A low-risk result is reassuring! Continue with routine annual Pap smear screenings. "
                         "The main preventive steps are regular check-ups, HPV vaccination if eligible, "
                         "and safe sexual practices.")
            elif any(w in u for w in ["mean", "result", "explain", "understand"]):
                if st.session_state.last_prediction:
                    p = st.session_state.last_prediction
                    reply = (f"Your result is {p['risk_level']} risk with a probability of {p['probability']:.1%}. "
                             f"The model identified these factors: {', '.join(p['risk_factors'])}. "
                             "The threshold is 50% — above that is high risk, below is low risk. "
                             "Always confirm with a healthcare professional.")
                else:
                    reply = "Please run a prediction first using the sidebar, then come back here to understand your result."
            elif any(w in u for w in ["next", "do", "step", "action", "recommend"]):
                reply = ("Next steps depend on your risk level. For HIGH risk: contact your gynaecologist this week. "
                         "For MEDIUM/LOW risk: schedule a routine Pap smear if you haven't had one recently. "
                         "In all cases: quit smoking if applicable, practice safe sex, and consider HPV vaccination.")
            elif any(w in u for w in ["hpv", "papilloma"]):
                reply = ("HPV is the main cause of cervical cancer. Most infections clear naturally, "
                         "but high-risk strains (HPV 16, 18) can cause cell changes over time. "
                         "The HPV vaccine is highly effective and recommended for ages 9–26. "
                         "Regular Pap smears catch early changes before they become cancer.")
            elif any(w in u for w in ["smok", "cigarette"]):
                reply = ("Smoking is a major cervical cancer risk factor. It weakens your immune system's "
                         "ability to fight HPV and introduces carcinogens directly to cervical cells. "
                         "Quitting smoking at any age reduces your risk significantly.")
            else:
                reply = ("I'm your cervical cancer risk assistant. I can explain your prediction result, "
                         "discuss risk factors (HPV, smoking, STDs, contraceptives), or suggest next steps. "
                         "Run a prediction first, then ask me anything about it! "
                         "Remember: always consult a real doctor for medical decisions.")

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

# ──────────────────────────────────────────────────────────────────
# TAB 3 — MODELS
# ──────────────────────────────────────────────────────────────────
with t3:
    st.markdown("<div class='section-header'>Model comparison</div>", unsafe_allow_html=True)
    if metrics is None:
        st.warning("Metrics not available. Check outputs/ folder.")
    else:
        img = load_img("roc_pr_curves.png")
        if img:
            st.image(img, width="stretch")
        else:
            ca, cb = st.columns(2)
            with ca:
                fig, ax = dark_fig(5, 4)
                ax.plot(metrics['fpr'], metrics['tpr'], color='#58a6ff', lw=2.5,
                        label=f"AUC={metrics['auc']:.3f}")
                ax.plot([0, 1], [0, 1], '--', color='#30363d', lw=1)
                ax.fill_between(metrics['fpr'], metrics['tpr'], alpha=.08, color='#58a6ff')
                ax.set_xlabel("FPR", color="#8b949e"); ax.set_ylabel("TPR", color="#8b949e")
                ax.set_title("ROC Curve", color="#e6edf3")
                ax.legend(fontsize=9, facecolor="#0d1117", labelcolor="white")
                st.pyplot(fig, width="stretch"); plt.close()
            with cb:
                fig, ax = dark_fig(5, 4)
                ax.plot(metrics['rec_curve'], metrics['prec_curve'], color='#3fb950', lw=2.5,
                        label=f"AP={metrics['ap']:.3f}")
                ax.fill_between(metrics['rec_curve'], metrics['prec_curve'], alpha=.08, color='#3fb950')
                ax.set_xlabel("Recall", color="#8b949e"); ax.set_ylabel("Precision", color="#8b949e")
                ax.set_title("Precision-Recall", color="#e6edf3")
                ax.legend(fontsize=9, facecolor="#0d1117", labelcolor="white")
                st.pyplot(fig, width="stretch"); plt.close()

        cm_img = load_img("confusion_matrices.png")
        if cm_img:
            st.markdown("<div class='section-header'>Confusion matrices</div>", unsafe_allow_html=True)
            st.image(cm_img, width="stretch")
        else:
            ca, cb = st.columns([1, 1])
            with ca:
                fig, ax = dark_fig(4.5, 3.5)
                sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Cancer', 'Cancer'],
                            yticklabels=['No Cancer', 'Cancer'],
                            linewidths=1, linecolor="#0d1117",
                            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
                ax.set_xlabel("Predicted", color="#8b949e")
                ax.set_ylabel("Actual", color="#8b949e")
                ax.set_title(f"Confusion — {model_name}", color="#e6edf3", fontsize=10)
                st.pyplot(fig, width="stretch"); plt.close()
            with cb:
                tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
                st.markdown(f"""
                <div class='box-info'>
                    <b>Error analysis</b><br><br>
                    TP = <b>{tp}</b> — cancers detected<br>
                    TN = <b>{tn}</b> — correct negatives<br>
                    FP = <b>{fp}</b> — false alarms<br>
                    FN = <b>{fn}</b> — MISSED cancers (critical)<br><br>
                    Sensitivity = {tp/(tp+fn+1e-9):.4f}<br>
                    Specificity = {tn/(tn+fp+1e-9):.4f}
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Performance table</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Model": model_name,
            "AUC-ROC": round(metrics['auc'], 4),
            "F1": round(metrics['f1'], 4),
            "Recall": round(metrics['recall'], 4),
            "Precision": round(metrics['precision'], 4),
            "Accuracy": round(metrics['accuracy'], 4),
        }]), hide_index=True, width="stretch")

# ──────────────────────────────────────────────────────────────────
# TAB 4 — SHAP
# ──────────────────────────────────────────────────────────────────
with t4:
    st.markdown("<div class='section-header'>Explainability — SHAP values</div>", unsafe_allow_html=True)
    shap_files = {
        "Feature Importance — global impact": "shap_importance.png",
        "Beeswarm — impact direction":        "shap_beeswarm.png",
        "Waterfall Plot":                     "shap_waterfall.png",
    }
    found = False
    for title, fname in shap_files.items():
        img = load_img(fname)
        if img:
            found = True
            st.markdown(f"#### {title}")
            st.image(img, width="stretch")
    if not found:
        st.warning("Run the training notebook to generate SHAP charts in outputs/.")

# ──────────────────────────────────────────────────────────────────
# TAB 5 — EDA
# ──────────────────────────────────────────────────────────────────
with t5:
    st.markdown("<div class='section-header'>Exploratory data analysis</div>", unsafe_allow_html=True)
    eda_imgs = {
        "Target distribution":   "target_distribution.png",
        "Feature distributions": "feature_distributions.png",
        "Missing values":        "missing_values.png",
        "Correlation matrix":    "correlation_matrix.png",
        "Boxplots":              "boxplots.png",
    }
    ecols = st.columns(2)
    ei = 0; found_eda = False
    for title, fname in eda_imgs.items():
        img = load_img(fname)
        if img:
            found_eda = True
            with ecols[ei % 2]:
                st.markdown(f"#### {title}")
                st.image(img, width="stretch")
            ei += 1

    if not found_eda and df_raw is not None:
        ea, eb = st.columns(2)
        with ea:
            fig, ax = dark_fig(5, 4)
            counts = df_raw['Biopsy'].value_counts()
            ax.pie(counts, labels=['No Cancer', 'Cancer'],
                   colors=['#58a6ff', '#da3633'], autopct='%1.1f%%',
                   wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
                   textprops={'color': 'white', 'fontsize': 10})
            ax.set_title("Class balance", color="#e6edf3")
            st.pyplot(fig, width="stretch"); plt.close()
        with eb:
            fig, ax = dark_fig(5, 4)
            for val, c, lbl in [(0, '#58a6ff', 'No Cancer'), (1, '#da3633', 'Cancer')]:
                df_raw[df_raw['Biopsy'] == val]['Age'].dropna().hist(
                    bins=20, ax=ax, alpha=.7, color=c, label=lbl)
            ax.set_xlabel("Age", color="#8b949e")
            ax.set_ylabel("Count", color="#8b949e")
            ax.set_title("Age distribution", color="#e6edf3")
            ax.legend(facecolor="#0d1117", labelcolor="white")
            st.pyplot(fig, width="stretch"); plt.close()

    if df_raw is not None:
        st.markdown("<div class='section-header'>Descriptive statistics</div>", unsafe_allow_html=True)
        nc = [c for c in ['Age', 'Number of sexual partners',
                           'First sexual intercourse', 'Num of pregnancies']
              if c in df_raw.columns]
        st.dataframe(df_raw[nc].describe().round(2).T, width="stretch")

# ──────────────────────────────────────────────────────────────────
# TAB 6 — ABOUT
# ──────────────────────────────────────────────────────────────────
with t6:
    st.markdown("<div class='section-header'>About this project</div>", unsafe_allow_html=True)
    pa, pb = st.columns([1, 1], gap="large")
    with pa:
        st.markdown("""
        <div class='box-info'><b>Author</b><br>Hadil Dhaya · 4th Year Data Science · Group 5 · 2026</div>
        <div class='box-info'>
            <b>Dataset</b><br>
            UCI Cervical Cancer Risk Factors<br>
            858 patients · 36 features · Target: Biopsy<br>
            <a href='https://archive.ics.uci.edu/dataset/383' style='color:#58a6ff'>UCI ML Repository</a>
        </div>
        <div class='box-info'>
            <b>Architecture</b><br>
            Streamlit → final_pipeline.pkl (sklearn Pipeline)<br>
            Imputer + Scaler + Model in one object
        </div>""", unsafe_allow_html=True)
    with pb:
        if PIPELINE_FEATURES:
            st.markdown(f"""
            <div class='box-info'>
                <b>Pipeline features ({len(PIPELINE_FEATURES)})</b><br><br>
                {"<br>".join(PIPELINE_FEATURES[:20])}
                {"<br>..." if len(PIPELINE_FEATURES) > 20 else ""}
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Methodological choices</div>", unsafe_allow_html=True)
    for q, a in [
        ("Why KNN Imputer?", "Preserves feature correlations in medical data."),
        ("Why SMOTE?", "Generates synthetic minority examples (~6% cancer). Better Recall than class_weight alone."),
        ("Why Recall > Accuracy?", "Missing a real cancer (false negative) is catastrophic."),
        ("Why sklearn Pipeline?", "Chains imputer + scaler + model into one object. Prevents data leakage and simplifies deployment."),
    ]:
        with st.expander(f"? {q}"):
            st.markdown(f"<div class='box-info'>{a}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#30363d;font-size:.76rem;font-family:IBM Plex Mono,monospace'>
    Cervical Cancer Risk Predictor · Hadil Dhaya · 2026 ·
    <a href='https://archive.ics.uci.edu/dataset/383' style='color:#30363d'>UCI ML Repository</a>
    · Academic use only
</p>""", unsafe_allow_html=True)
# Streamlit ML App - improved version