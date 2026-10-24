import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Stroke Prediction | KNN Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    :root {
        --primary: #ef4444;
        --primary-dark: #b91c1c;
        --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        --card-bg: rgba(255, 255, 255, 0.06);
        --card-border: rgba(255, 255, 255, 0.12);
        --text-main: #f8fafc;
        --text-muted: #cbd5e1;
    }
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background: var(--bg-gradient); background-attachment: fixed; color: var(--text-main); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stApp > header {display: none;}
    .navbar {
        position: fixed; top: 0; left: 0; right: 0; height: 70px;
        background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--card-border);
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 40px; z-index: 1000;
    }
    .navbar .logo {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.35rem; font-weight: 700; color: #fff; text-decoration: none;
    }
    .navbar .logo span { color: var(--primary); }
    .page-spacer { height: 90px; }
    .hero {
        padding: 60px 20px 30px; text-align: center;
        max-width: 900px; margin: 0 auto;
    }
    .hero h1 {
        font-size: 3rem; font-weight: 700; margin: 0 0 12px; letter-spacing: -1px;
        background: linear-gradient(to right, #fff, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.15rem; color: var(--text-muted); margin: 0 0 30px; line-height: 1.6;
    }
    .glass-card {
        background: var(--card-bg); border: 1px solid var(--card-border);
        border-radius: 20px; padding: 32px; backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .section-title {
        font-size: 1.5rem; font-weight: 700; margin-bottom: 24px; color: #fff;
        display: flex; align-items: center; gap: 10px;
    }
    .result-card {
        border-radius: 24px; padding: 32px; text-align: center;
        margin-top: 30px; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    .result-card.high {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .result-card.low {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .result-badge {
        display: inline-flex; align-items: center; gap: 10px;
        font-size: 1.75rem; font-weight: 700; margin-bottom: 12px;
    }
    .result-text { font-size: 1.15rem; color: #e2e8f0; line-height: 1.6; }
    .footer {
        margin-top: 80px; padding: 40px 20px; border-top: 1px solid var(--card-border);
        text-align: center; color: var(--text-muted); font-size: 0.9rem;
    }
    .footer a { color: var(--primary); text-decoration: none; font-weight: 600; }
    .glass-card .stSelectbox > div > div,
    .glass-card .stNumberInput > div > div,
    .glass-card .stSlider > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important; color: #fff !important;
    }
    .glass-card .stSelectbox label,
    .glass-card .stNumberInput label,
    .glass-card .stSlider label {
        color: var(--text-muted) !important; font-weight: 500 !important; font-size: 0.9rem !important;
    }
    .stButton > button {
        width: 100%; padding: 14px;
        background: linear-gradient(to right, var(--primary), var(--primary-dark));
        border: none; border-radius: 12px; color: #fff; font-size: 1rem;
        font-weight: 600; cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.35);
    }
    @media (max-width: 768px) {
        .navbar { padding: 0 20px; }
        .hero h1 { font-size: 2.2rem; }
        .glass-card { padding: 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def navbar():
    st.markdown(
        """
        <nav class="navbar">
            <div class="logo">❤️ <span>HeartCheck</span></div>
        </nav>
        """,
        unsafe_allow_html=True,
    )

def footer():
    st.markdown(
        """
        <div class="footer">
            <p>© 2026 <strong>Vivek Verma</strong> &bull; Built with Streamlit &bull; KNN Classifier</p>
            <p><a href="https://github.com/vivekcode12345/heart-stroke-prediction-knn" target="_blank">View on GitHub</a> &bull;
               <a href="https://vivekcode12345-heart-stroke-prediction-knn-app-o7rtko.streamlit.app/" target="_blank">Live Demo</a></p>
            <p style="margin-top:10px; font-size:0.8rem; opacity:0.7;">⚠️ For educational purposes only — not a substitute for professional medical advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def nav_buttons():
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.session_state.page = "Home"; st.rerun()
    with c2:
        if st.button("🔍 Predict", use_container_width=True, key="nav_predict"):
            st.session_state.page = "Predict"; st.rerun()
    with c3:
        if st.button("ℹ️ About", use_container_width=True, key="nav_about"):
            st.session_state.page = "About"; st.rerun()
def page_home():
    st.markdown('<div class="page-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>Heart Stroke <span>Prediction</span></h1>
            <p>Assess your cardiovascular risk instantly with a machine-learning powered KNN classifier. Enter your health metrics and get an immediate risk evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_buttons()
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="glass-card"><h4 style="color:#fff;margin:0 0 8px 0;">📊 Accurate</h4><p style="color:var(--text-muted);margin:0;">Trained on real clinical heart-disease dataset with KNN algorithm.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="glass-card"><h4 style="color:#fff;margin:0 0 8px 0;">⚡ Instant</h4><p style="color:var(--text-muted);margin:0;">Get your heart-risk result in seconds.</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="glass-card"><h4 style="color:#fff;margin:0 0 8px 0;">🔒 Private</h4><p style="color:var(--text-muted);margin:0;">Your data stays local — nothing is stored or shared.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
    if st.button("Start Assessment", use_container_width=True, key="start_assessment"):
        st.session_state.page = "Predict"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    footer()
def page_predict():
    st.markdown('<div class="page-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero" style="text-align:left; padding-top:20px;">'
        '<h1 style="font-size:2rem; background:none; -webkit-text-fill-color:#fff;">🔍 Heart Risk Assessment</h1>'
        '<p style="color:var(--text-muted);">Provide your health details below to check your heart stroke risk.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Patient Details</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 40, key="age")
            sex = st.selectbox("Sex", ["M", "F"], key="sex")
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], key="chest_pain")
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, key="resting_bp")
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, key="cholesterol")
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], key="fasting_bs")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="resting_ecg")
            max_hr = st.slider("Max Heart Rate", 60, 220, 150, key="max_hr")
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], key="exercise_angina")
            oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, key="oldpeak")
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="st_slope")
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🩺 Predict Heart Risk", key="predict_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    if predict_clicked:
        raw_input = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }
        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(
                '<div class="result-card high">'
                '<div class="result-badge">⚠️ High Risk</div>'
                '<div class="result-text">Based on your inputs, the model indicates a <strong>higher risk of heart disease</strong>. Please consult a qualified healthcare professional for further evaluation.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card low">'
                '<div class="result-badge">✅ Low Risk</div>'
                '<div class="result-text">Based on your inputs, the model indicates a <strong>lower risk of heart disease</strong>. Maintain a healthy lifestyle and regular check-ups.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    nav_buttons()
    footer()
def page_about():
    st.markdown('<div class="page-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero" style="text-align:left; padding-top:20px;">'
        '<h1 style="font-size:2rem; background:none; -webkit-text-fill-color:#fff;">ℹ️ About This Project</h1>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### 📖 Overview
        Heart disease is one of the leading causes of mortality worldwide. Early detection can significantly improve outcomes.
        This project provides an interactive tool where users input their health metrics and receive an instant prediction of whether
        they are at **high risk** or **low risk** of heart disease, powered by a trained **K-Nearest Neighbors (KNN)** classifier.

        ### 🛠️ Tech Stack
        - **Python 3** · **Streamlit** · **pandas** · **scikit-learn** · **joblib**

        ### 📊 Dataset Features
        Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

        ### 👨‍💻 Author
        **Vivek Verma** — [GitHub](https://github.com/vivekcode12345)

        ### 📄 License
        MIT License
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
    nav_buttons()
    footer()

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Predict":
    page_predict()
else:
    page_about()