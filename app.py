import streamlit as st
import pandas as pd
import joblib

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Smart Heart Stroke Predictor",
    page_icon="🫀",
    layout="centered"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1320, #05080f 65%);
    color: #eaeaea;
    font-family: 'Inter', sans-serif;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 35px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    animation: slideUp 1.1s ease;
    max-width: 650px;
    margin: auto;
}

/* Vertical Spacing for inputs */
.stSelectbox, .stSlider, .stNumberInput {
    margin-bottom: 25px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 15px;
    color: #9aa4b2;
    margin-bottom: 25px;
}

/* Section label inside card */
.section-title {
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: #00f5ff;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* Button */
.stButton > button {
    width: 300px;
    background: linear-gradient(90deg, #ff2cdf, #00f5ff);
    color: #000;
    font-size: 18px;
    font-weight: 700;
    padding: 14px;
    border-radius: 14px;
    border: none;
    transition: all 0.35s ease;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 0 25px #00f5ff, 0 0 45px #ff2cdf;
}

/* Result cards */
.result-high {
    background: linear-gradient(135deg, rgba(255,45,90,0.18), rgba(255,45,90,0.05));
    border-left: 4px solid #ff2d5a;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    animation: pulse 1.4s infinite;
}

.result-low {
    background: linear-gradient(135deg, rgba(0,245,255,0.18), rgba(0,245,255,0.05));
    border-left: 4px solid #00f5ff;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    animation: fadeIn 1s ease;
}

/* Animations */
@keyframes slideUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(255,45,90,0.6);}
    70% {box-shadow: 0 0 0 22px rgba(255,45,90,0);}
    100% {box-shadow: 0 0 0 0 rgba(255,45,90,0);}
}
</style>
""", unsafe_allow_html=True)

# ==================== Load ML Assets ====================
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ==================== ONE-LINE CENTERED HEADING ====================
st.markdown("""
<div style="
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    font-size: 46px;
    font-weight: 800;
    white-space: nowrap;
    margin-bottom: 10px;
">
    <span>🫀</span>
    <span style="
        background: linear-gradient(90deg, #00f5ff, #ff2cdf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">
        Smart Heart Stroke Predictor
    </span>
    <span>⚕️</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">AI-powered stroke risk assessment using machine learning</p>',
    unsafe_allow_html=True
)

# ==================== Input Section (Centered Glass Card) ====================
st.markdown("""
<div class="glass-card">
    <div class="section-title">Patient Health Profile</div>
""", unsafe_allow_html=True)

# Inputs line by line
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==================== ✅ PERFECTLY CENTERED BUTTON ====================
left, center, right = st.columns([1, 2, 1])
with center:
    predict_clicked = st.button("🚀 Predict Stroke Risk")

# ==================== Prediction Logic ====================
if predict_clicked:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
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
            '<div class="result-high">⚠️ <b>High Risk of Heart Stroke</b><br>Please consult a medical professional immediately.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-low">✅ <b>Low Risk of Heart Stroke</b><br>Maintain a healthy lifestyle.</div>',
            unsafe_allow_html=True
        )