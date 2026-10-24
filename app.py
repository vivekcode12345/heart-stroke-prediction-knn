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


# ---------------------------------------------------------------- STYLES
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --surface: rgba(255,255,255,0.045);
        --border: rgba(255,255,255,0.10);
        --text: #f1f5f9;
        --muted: #94a3b8;
        --accent: #f43f5e;
        --accent-2: #fb923c;
    }

    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
            radial-gradient(900px 500px at 15% -10%, rgba(244,63,94,0.18), transparent 60%),
            radial-gradient(800px 500px at 90% 0%, rgba(99,102,241,0.18), transparent 60%),
            var(--bg);
        color: var(--text);
    }

    /* hide streamlit chrome */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
    .block-container { max-width: 1080px; padding-top: 1.2rem; padding-bottom: 3rem; }

    /* ---------- top nav (buttons inside .st-key-nav) ---------- */
    .st-key-nav {
        border-bottom: 1px solid var(--border);
        padding-bottom: 14px; margin-bottom: 8px;
    }
    .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.5px; padding-top: 6px; }
    .brand b { color: var(--accent); }
    .st-key-nav .stButton > button {
        background: transparent; border: 1px solid transparent; color: var(--muted);
        font-weight: 600; font-size: 0.92rem; padding: 6px 14px; border-radius: 10px; box-shadow: none;
    }
    .st-key-nav .stButton > button:hover { color: #fff; background: var(--surface); border-color: var(--border); }
    .st-key-nav .stButton > button[kind="primary"] { color: #fff; background: var(--surface); border-color: var(--border); }

    /* ---------- hero ---------- */
    .hero { text-align: center; padding: 64px 10px 28px; }
    .pill {
        display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        color: #fda4af; background: rgba(244,63,94,0.12); border: 1px solid rgba(244,63,94,0.3);
        margin-bottom: 22px;
    }
    .hero h1 {
        font-size: 3.4rem; line-height: 1.05; font-weight: 800; letter-spacing: -2px; margin: 0 0 18px;
        color: #fff;
    }
    .hero h1 span {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero p { color: var(--muted); font-size: 1.1rem; max-width: 640px; margin: 0 auto; line-height: 1.7; }

    /* ---------- CTA ---------- */
    .st-key-cta .stButton > button, .st-key-predict_btn .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2)); border: none; color: #fff;
        font-weight: 700; font-size: 1rem; padding: 12px 26px; border-radius: 12px;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .st-key-cta .stButton > button:hover, .st-key-predict_btn .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 12px 28px rgba(244,63,94,0.35); color: #fff;
    }

    /* ---------- stat strip + feature cards ---------- */
    .stats {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 40px 0 18px;
    }
    .stat { text-align: center; padding: 18px; border-radius: 16px; background: var(--surface); border: 1px solid var(--border); }
    .stat .n { font-size: 1.8rem; font-weight: 800; color: #fff; }
    .stat .l { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }

    .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 8px; }
    .card {
        padding: 24px; border-radius: 18px; background: var(--surface); border: 1px solid var(--border);
        transition: transform .2s ease, border-color .2s ease;
    }
    .card:hover { transform: translateY(-4px); border-color: rgba(244,63,94,0.45); }
    .card .ico { font-size: 1.6rem; margin-bottom: 10px; }
    .card h4 { margin: 0 0 6px; font-size: 1.05rem; color: #fff; }
    .card p { margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.6; }

    /* ---------- form card ---------- */
    .st-key-form_card {
        background: var(--surface); border: 1px solid var(--border) !important; border-radius: 20px; padding: 14px 10px;
    }
    .group-title {
        font-size: 0.78rem; font-weight: 700; color: var(--accent-2); text-transform: uppercase;
        letter-spacing: .1em; margin: 6px 0 2px;
    }
    .page-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -1px; margin: 28px 0 4px; }
    .page-sub { color: var(--muted); margin-bottom: 20px; }
    label, [data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; font-weight: 500 !important; }

    /* ---------- result ---------- */
    .result {
        margin-top: 24px; padding: 28px; border-radius: 20px; display: flex; gap: 24px; align-items: center;
        flex-wrap: wrap;
    }
    .result.high { background: linear-gradient(135deg, rgba(244,63,94,0.18), rgba(127,29,29,0.25)); border: 1px solid rgba(244,63,94,0.5); }
    .result.low  { background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(20,83,45,0.25)); border: 1px solid rgba(34,197,94,0.45); }
    .result .big { font-size: 3rem; font-weight: 800; line-height: 1; min-width: 130px; }
    .result.high .big { color: #fb7185; } .result.low .big { color: #4ade80; }
    .result h3 { margin: 0 0 6px; font-size: 1.35rem; color: #fff; }
    .result p { margin: 0; color: #cbd5e1; line-height: 1.6; }
    .meter { height: 8px; border-radius: 99px; background: rgba(255,255,255,0.1); margin-top: 14px; overflow: hidden; }
    .meter > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #4ade80, #facc15, #f43f5e); }

    /* ---------- about / footer ---------- */
    .about { padding: 28px; border-radius: 20px; background: var(--surface); border: 1px solid var(--border); line-height: 1.8; color: #cbd5e1; }
    .about h3 { color: #fff; margin-top: 18px; } .about h3:first-child { margin-top: 0; }
    .chips span { display: inline-block; margin: 4px 6px 0 0; padding: 4px 12px; border-radius: 999px;
        background: rgba(255,255,255,0.06); border: 1px solid var(--border); font-size: 0.85rem; }
    .foot { margin-top: 60px; padding-top: 24px; border-top: 1px solid var(--border); text-align: center; color: var(--muted); font-size: 0.85rem; }
    .foot a { color: var(--accent); text-decoration: none; font-weight: 600; }

    @media (max-width: 768px) {
        .hero h1 { font-size: 2.3rem; } .grid3, .stats { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- SHARED UI
def navbar():
    with st.container(key="nav"):
        c0, c1, c2, c3 = st.columns([5, 1, 1, 1], vertical_alignment="center")
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
            predict_clicked = st.button("🩺 Predict Heart Risk", width="stretch")

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