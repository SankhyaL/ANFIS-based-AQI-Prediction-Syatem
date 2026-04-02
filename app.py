import streamlit as st
import numpy as np
import pickle
import os

from anfis import ANFIS

# ------------------ LOAD MODEL ------------------
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
model_path = os.path.join(BASE_DIR, "model", "anfis_full_model.pkl")
print("Model path:", model_path)

with open(model_path, "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler_X = data["scaler_X"]
scaler_y = data["scaler_y"]

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #fff176, #ffb74d, #e53935);
        color: black;
    }

    h1 {
        text-align: center;
        color: #b71c1c;
        font-weight: bold;
    }

    .stButton>button {
        background-color: #e53935;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }

    .stButton>button:hover {
        background-color: #b71c1c;
    }

    .stNumberInput input {
        background-color: #fff3e0;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🌫️ AQI Prediction System")
st.markdown("### Enter pollutant values")

# ------------------ INPUTS ------------------
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
    pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
    no2  = st.number_input("NO2", 0.0, 200.0, 30.0)

with col2:
    so2  = st.number_input("SO2", 0.0, 200.0, 20.0)
    co   = st.number_input("CO", 0.0, 50.0, 1.0)
    o3   = st.number_input("O3", 0.0, 200.0, 60.0)

# ------------------ PREDICTION ------------------
if st.button("🚀 Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])

    X_scaled = scaler_X.transform(input_data)
    y_pred_norm, _, _ = model.forward(X_scaled)

    y_pred = scaler_y.inverse_transform(
        y_pred_norm.reshape(-1, 1)
    ).flatten()[0]

    # AQI Category Styling
    if y_pred <= 50:
        color = "#66bb6a"
        category = "Good 😊"
    elif y_pred <= 100:
        color = "#ffee58"
        category = "Moderate 😐"
    elif y_pred <= 150:
        color = "#ffb74d"
        category = "Unhealthy for Sensitive Groups 😷"
    elif y_pred <= 200:
        color = "#ff7043"
        category = "Unhealthy 😷"
    else:
        color = "#d32f2f"
        category = "Very Unhealthy 🚨"

    st.markdown(f"""
        <div class="result-box" style="background-color:{color};">
            Predicted AQI: {y_pred:.2f} <br>
            {category}
        </div>
    """, unsafe_allow_html=True)