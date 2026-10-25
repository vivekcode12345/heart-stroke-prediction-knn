import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HeartCheck | Heart Risk Prediction",
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


def go(page):
    st.session_state.page = page


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #FFFFFF;
        --tint: #FFF7F0;
        --border: #E5E7EB;
        --text: #0F172A;
        --muted: #64748B;
        --accent: #F97316;
        --accent-2: #FB923C;
        --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    }

    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(180deg, var(--tint) 0%, #FFFFFF 45%, #FFFFFF 75%, var(--tint) 100%);
        color: var(--text);
    }

    /* hide streamlit chrome */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
    .block-container { max-width: 1080px; padding-top: 1.1rem; padding-bottom: 3rem; }

    /* ---------- top nav ---------- */
    .st-key-nav {
        position: sticky;
        top: 0;
        z-index: 200;
        background: var(--bg);
        border-bottom: 1px solid var(--border);
        padding: 10px 0;
        margin-bottom: 6px;
    }
    .brand { font-size: 1.2rem; font-weight: 800; letter-spacing: -0.4px; color: var(--text); padding-top: 6px; }
    .brand b { color: var(--accent); }
    .st-key-nav .stButton > button {
        background: transparent; border: 1px solid transparent; color: var(--muted);
        font-weight: 600; font-size: 0.9rem; padding: 5px 12px; border-radius: 10px; box-shadow: none;
    }
    .st-key-nav .stButton > button:hover { background: #F3F4F6; border-color: var(--border); color: var(--text); }
    .st-key-nav .stButton > button[kind="primary"] {
        background: #FEF3E7; border: 1px solid #FDBA74; color: #B45309;
    }
    .st-key-nav .stButton > button[kind="primary"]:hover {
        background: #FDE6CB; border-color: #F59E0B; color: #92400E;
    }

    /* ---------- hero ---------- */
    .hero { text-align: center; padding: 52px 10px 26px; }
    .pill {
        display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
        color: #9A3412; background: #FEF3E7; border: 1px solid #FDBA74; margin-bottom: 16px;
    }
    .hero h1 {
        font-size: 3.1rem; line-height: 1.05; font-weight: 800; letter-spacing: -1.5px; margin: 0 0 14px;
        color: var(--text);
    }
    .hero h1 span {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 640px; margin: 0 auto; line-height: 1.75; }

    /* ---------- CTA ---------- */
    .st-key-cta .stButton > button, .st-key-predict_btn .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2)); border: none; color: #fff;
        font-weight: 700; font-size: 1rem; padding: 12px 26px; border-radius: 12px;
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }
    .st-key-cta .stButton > button:hover, .st-key-predict_btn .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(249,115,22,0.30);
        filter: brightness(1.04);
    }

    /* ---------- stat strip ---------- */
    .stats {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 34px 0 12px;
    }
    .stat { text-align: center; background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 20px 10px; box-shadow: var(--shadow); }
    .stat .n { font-size: 1.7rem; font-weight: 800; color: var(--text); }
    .stat .l { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

    /* ---------- feature cards ---------- */
    .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 10px; align-items: start; }
    .card {
        background: #fff; border: 1px solid var(--border); border-radius: 20px; padding: 22px;
        box-shadow: var(--shadow); display: flex; flex-direction: column; height: 100%;
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 14px 28px rgba(0,0,0,0.05); border-color: #F59E0B; }
    .card .ico { font-size: 1.75rem; margin-bottom: 12px; }
    .card h4 { margin: 0 0 8px; font-size: 1.05rem; color: var(--text); }
    .card p { margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.7; }

    /* ---------- form card ---------- */
    .st-key-form_card {
        background: #fff !important;
        border: 1px solid var(--border) !important;
        border-radius: 22px !important;
        box-shadow: var(--shadow);
    }
    .group-title {
        font-size: 0.78rem; font-weight: 800; color: var(--accent); text-transform: uppercase;
        letter-spacing: 0.12em; margin: 2px 0 4px; display: block;
    }
    .page-title { font-size: 1.95rem; font-weight: 800; letter-spacing: -0.5px; margin: 22px 0 4px; color: var(--text); }
    .page-sub { color: var(--muted); margin-bottom: 14px; }
    label, [data-testid="stWidgetLabel"] p { color: #334155 !important; font-weight: 600 !important; font-size: 0.92rem !important; }

    /* ---------- inputs ---------- */
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"],
    .stNumberInput div[data-baseweb="input"] > div {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        min-height: 44px !important;
        height: 44px !important;
    }
    .stSelectbox:focus-within div[data-baseweb="select"] > div,
    .stNumberInput:focus-within div[data-baseweb="input"] > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(249,115,22,0.12) !important;
    }
    .stSelectbox input, .stSelectbox div[data-baseweb="input"] input,
    .stNumberInput input { background: transparent !important; color: #FFFFFF !important; font-weight: 500 !important; }
    .stSelectbox svg, .stNumberInput svg { color: #CBD5E1 !important; fill: #CBD5E1 !important; }
    .stSelectbox [data-baseweb="menu"] { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 12px !important; margin-top: 6px !important; }
    .stSelectbox [data-baseweb="menu"] [role="option"] { color: #FFFFFF !important; background: #1E293B !important; }
    .stSelectbox [data-baseweb="menu"] [role="option"]:hover { background: rgba(249,115,22,0.2) !important; color: #FFFFFF !important; }
    .stSelectbox [data-baseweb="menu"] [data-active="true"], .stSelectbox [data-baseweb="menu"] li[aria-selected="true"] {
        background: #1E293B !important; color: #F97316 !important;
    }
    .stSelectbox input::placeholder, .stNumberInput input::placeholder { color: #94A3B8 !important; opacity: 1 !important; }
    .stSelectbox input[disabled], .stNumberInput input[disabled] { color: #94A3B8 !important; }
    [data-testid="stCaption"] p { color: #64748B !important; font-size: 0.82rem !important; }

    /* ---------- slider ---------- */
    .stSlider { margin: 6px 0 16px; }
    .stSlider [data-baseweb="slider"] { padding: 12px 0 4px; }
    .stSlider span.rfc-track, .stSlider span.rfc-slider-track {
        background: #E5E7EB !important; height: 5px !important; border-radius: 999px;
    }
    .stSlider span.rfc-range, .stSlider span.rfc-slider-range {
        background: #F97316 !important; height: 5px !important; border-radius: 999px;
    }
    .stSlider div[role="slider"], .stSlider span.rfc-thumb {
        background: #F97316 !important; border: 2px solid #F97316 !important; border-radius: 50% !important;
        width: 18px !important; height: 18px !important; box-shadow: 0 2px 6px rgba(249,115,22,0.35);
    }
    .stSlider div[role="slider"]:hover, .stSlider span.rfc-thumb:hover { border-color: #C2410C !important; }
    .stSlider [data-testid="stValueLabel"] { color: #F97316 !important; font-weight: 700 !important; }
    .stSlider [data-baseweb="slider"] span, .stSlider [data-testid="stCaption"] { color: #64748B !important; }

    /* ---------- result ---------- */
    .result {
        margin-top: 20px; padding: 26px; border-radius: 22px; display: flex; gap: 26px; align-items: center;
        flex-wrap: wrap; box-shadow: var(--shadow);
    }
    .result.high { background: #FFF1F2; border: 1px solid #FCA5A5; }
    .result.low  { background: #F0FDF4; border: 1px solid #86EFAC; }
    .result .big { font-size: 3.4rem; font-weight: 800; line-height: 1; min-width: 130px; color: var(--text); }
    .result.high .big { color: #B91C1C; } .result.low .big { color: #15803D; }
    .result.high h3 { margin: 0 0 6px; font-size: 1.3rem; color: #B91C1C; }
    .result.low h3  { margin: 0 0 6px; font-size: 1.3rem; color: #15803D; }
    .result p { margin: 0; color: var(--muted); line-height: 1.6; }
    .meter { height: 10px; border-radius: 99px; background: #E5E7EB; margin-top: 16px; overflow: hidden; width: 100%; max-width: 300px; }
    .meter > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #F97316, #EF4444); }

    /* ---------- about / footer ---------- */
    .about { padding: 26px; border-radius: 20px; background: #fff; border: 1px solid var(--border); line-height: 1.8; color: #475569; box-shadow: var(--shadow); }
    .about h3 { color: var(--text); margin-top: 16px; font-size: 1.05rem; }
    .about h3:first-child { margin-top: 0; }
    .about b, .about strong { color: var(--text); }
    .chips span { display: inline-block; margin: 4px 6px 0 0; padding: 5px 14px; border-radius: 999px; background: #F3F4F6; border: 1px solid var(--border); color: #374151; font-size: 0.85rem; font-weight: 500; }
    .foot { margin-top: 46px; padding-top: 20px; border-top: 1px solid var(--border); text-align: center; color: var(--muted); font-size: 0.85rem; }
    .foot a { color: var(--accent); text-decoration: none; font-weight: 600; }

    @media (max-width: 768px) {
        .hero h1 { font-size: 2.1rem; }
        .grid3, .stats { grid-template-columns: 1fr !important; }
        .st-key-form_card [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; }
        .block-container { padding-top: 0.7rem; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- SHARED UI
def navbar():
    with st.container(key="nav"):
        c0, c1, c2, c3 = st.columns([6, 1, 1, 1], vertical_alignment="center")
        c0.markdown('<div class="brand">❤️ Heart<b>Check</b></div>', unsafe_allow_html=True)
        for col, name in zip((c1, c2, c3), ("Home", "Predict", "About")):
            col.button(
                name, key=f"nav_{name}", width="stretch", on_click=go, args=(name,),
                type="primary" if st.session_state.page == name else "secondary",
            )


def footer():
    st.markdown(
        """
        <div class="foot">
            <p>© 2026 <strong>Vivek Verma</strong> · Built with Streamlit · KNN Classifier</p>
            <p><a href="https://github.com/vivekcode12345/heart-stroke-prediction-knn" target="_blank">GitHub</a> ·
               <a href="https://vivekcode12345-heart-stroke-prediction-knn-app-o7rtko.streamlit.app/" target="_blank">Live Demo</a></p>
            <p style="opacity:.7;font-size:.78rem">⚠️ Educational purposes only — not a substitute for professional medical advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- PAGES
def page_home():
    st.markdown(
        """
        <div class="hero">
            <div class="pill">● Machine-learning powered · KNN</div>
            <h1>Know your <span>heart risk</span><br>in seconds.</h1>
            <p>Enter a few health metrics and get an instant risk evaluation from a K-Nearest Neighbors
            model trained on a real clinical heart-disease dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([2, 1.2, 2])
    with mid:
        with st.container(key="cta"):
            st.button("Start Assessment →", key="start", width="stretch", on_click=go, args=("Predict",))

    st.markdown(
        """
        <div class="stats">
            <div class="stat"><div class="n">918</div><div class="l">Patient records</div></div>
            <div class="stat"><div class="n">11</div><div class="l">Health features</div></div>
            <div class="stat"><div class="n">KNN</div><div class="l">Classifier</div></div>
        </div>
        <div class="grid3">
            <div class="card"><div class="ico">📊</div><h4>Data-driven</h4>
                <p>Trained on a clinical heart-disease dataset using the K-Nearest Neighbors algorithm.</p></div>
            <div class="card"><div class="ico">⚡</div><h4>Instant</h4>
                <p>Fill the form, hit predict, and see your result in under a second.</p></div>
            <div class="card"><div class="ico">🔒</div><h4>Private</h4>
                <p>Your inputs are used only for this prediction — nothing is stored or shared.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    footer()


def page_predict():
    st.markdown('<div class="page-title">Heart Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Fill in the details below and click predict.</div>', unsafe_allow_html=True)

    with st.container(border=True, key="form_card"):
        st.markdown('<div class="group-title">Basic info</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        age = a.slider("Age", 18, 100, 40)
        sex = b.selectbox("Sex", ["M", "F"], format_func=lambda x: {"M": "Male", "F": "Female"}[x])
        chest_pain = c.selectbox(
            "Chest pain type", ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda x: {
                "ATA": "ATA · Atypical angina", "NAP": "NAP · Non-anginal pain",
                "TA": "TA · Typical angina", "ASY": "ASY · Asymptomatic",
            }[x],
        )

        st.markdown('<div class="group-title">Vitals & blood tests</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        resting_bp = a.number_input("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = b.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = c.selectbox(
            "Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

        st.markdown('<div class="group-title">ECG & exercise</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        resting_ecg = a.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = b.slider("Max heart rate", 60, 220, 150)
        exercise_angina = c.selectbox(
            "Exercise-induced angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No"
        )
        a, b, _ = st.columns(3)
        oldpeak = a.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = b.selectbox("ST slope", ["Up", "Flat", "Down"])

        st.write("")
        with st.container(key="predict_btn"):
            predict_clicked = st.button("Predict Heart Risk", width="stretch")

    if predict_clicked:
        raw_input = {
            "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "MaxHR": max_hr, "Oldpeak": oldpeak,
            "Sex_" + sex: 1, "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1, "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1,
        }
        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        risk_pct = int(round(model.predict_proba(scaled)[0][1] * 100))

        if prediction == 1:
            cls, title, text = "high", "Higher risk detected", (
                "The model indicates a <b>higher risk of heart disease</b> for these inputs. "
                "Please consult a qualified healthcare professional for proper evaluation."
            )
        else:
            cls, title, text = "low", "Lower risk", (
                "The model indicates a <b>lower risk of heart disease</b> for these inputs. "
                "Keep up a healthy lifestyle and regular check-ups."
            )
        st.markdown(
            f"""
            <div class="result {cls}">
                <div class="big">{risk_pct}%</div>
                <div style="flex:1;min-width:240px">
                    <h3>{title}</h3><p>{text}</p>
                    <div class="meter"><div style="width:{risk_pct}%"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Percentage = share of the nearest neighbours in the training data that had heart disease.")
    footer()


def page_about():
    st.markdown('<div class="page-title">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="about">
            <h3>Overview</h3>
            Heart disease is one of the leading causes of death worldwide, and early detection improves outcomes.
            HeartCheck takes your health metrics and predicts <b>high</b> or <b>low</b> risk using a trained
            <b>K-Nearest Neighbors</b> classifier.
            <h3>Tech stack</h3>
            <div class="chips"><span>Python</span><span>Streamlit</span><span>pandas</span><span>scikit-learn</span><span>joblib</span></div>
            <h3>Dataset features</h3>
            <div class="chips"><span>Age</span><span>Sex</span><span>ChestPainType</span><span>RestingBP</span><span>Cholesterol</span>
            <span>FastingBS</span><span>RestingECG</span><span>MaxHR</span><span>ExerciseAngina</span><span>Oldpeak</span><span>ST_Slope</span></div>
            <h3>Author</h3>
            <b>Vivek Verma</b> — <a href="https://github.com/vivekcode12345" target="_blank" style="color:#f43f5e">GitHub</a>
            · MIT License
        </div>
        """,
        unsafe_allow_html=True,
    )
    footer()


# ---------------------------------------------------------------- ROUTER
navbar()
{"Home": page_home, "Predict": page_predict, "About": page_about}[st.session_state.page]()