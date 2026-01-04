# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------
# Streamlit page config
# ------------------------
st.set_page_config(
    page_title="🏎️ Sports Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>
/* Page background */
body, .main {
    background: linear-gradient(to right, #0b0b0b, #1c1c1c);
    color: #f0f0f0;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #ff0000;
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
}

/* Image banner */
img {
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Input boxes */
.stTextInput>div>input, .stNumberInput>div>input {
    border-radius: 12px;
    padding: 10px;
    background-color: #1e1e1e;
    color: #00ffcc;   /* changed text color */
    border: 2px solid #ff0000;
}

/* Card style select boxes */
.selectbox-container {
    background-color: #1e1e1e;
    border: 2px solid #ff0000;
    border-radius: 15px;
    padding: 10px;
    color: #00ffcc;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff0000, #ff7f00);
    color: white;
    font-size: 18px;
    border-radius: 15px;
    padding: 12px 20px;
    transition: transform 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Success price card */
.stSuccess {
    background: linear-gradient(to right, #ff0000, #ffcc00);
    color: #000;
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 0 20px #ffcc00;
}

/* Error message */
.stError {
    background-color: #ff4b4b;
    color: white;
    font-size: 1.2rem;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Title and hero image
# ------------------------
st.markdown("<h1>🏎️ Sports Car Price Predictor</h1>", unsafe_allow_html=True)
st.image("https://cdn.pixabay.com/photo/2017/09/14/10/43/ferrari-2748582_1280.jpg", use_column_width=True)

# ------------------------
# Load model
# ------------------------
model_path = "simple_car_price_model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
else:
    st.error(f"❌ Model file not found. Upload '{model_path}' to repo.")
    st.stop()

# ------------------------
# Input fields
# ------------------------
st.subheader("Enter Car Details:")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1980, max_value=2026, value=2022)
    mileage = st.number_input("Mileage (km)", min_value=0, value=5000)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=3.0, step=0.1)

with col2:
    # Fuel Type as card-style selectbox
    st.markdown('<div class="selectbox-container">Fuel Type</div>', unsafe_allow_html=True)
    fuel_type = st.selectbox("", ["Petrol", "Diesel", "Electric", "Hybrid"], key="fuel_type")

    # Transmission as card-style selectbox
    st.markdown('<div class="selectbox-container">Transmission</div>', unsafe_allow_html=True)
    transmission = st.selectbox("", ["Manual", "Automatic"], key="transmission")

    brand = st.text_input("Brand", value="Ferrari")

with col3:
    color = st.text_input("Color", value="Red")
    doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=2)

# ------------------------
# Predict button
# ------------------------
if st.button("Predict Price"):
    input_dict = {
        "year": [year],
        "mileage": [mileage],
        "engine_size": [engine_size],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "brand": [brand],
        "color": [color],
        "doors": [doors]
    }

    input_df = pd.DataFrame(input_dict)
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align input with model features
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_encoded.columns
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Prediction
    try:
        price = model.predict(input_encoded)[0]
        st.success(f"💰 Estimated Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
