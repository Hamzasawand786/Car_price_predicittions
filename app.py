# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="🏎️ Supercar Price Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ------------------------
# Custom CSS for sporty interface
# ------------------------
st.markdown("""
<style>
/* Body and background */
body, .main {
    background: linear-gradient(to bottom right, #0a0a0a, #1a1a1a);
    color: #f0f0f0;
    font-family: 'Orbitron', sans-serif;
}

/* Title */
h1 {
    color: #ff0000;
    text-align: center;
    font-size: 3.5rem;
    font-weight: bold;
    text-shadow: 0 0 15px #ff0000;
    margin-bottom: 10px;
}

/* Hero image */
.hero-img {
    border-radius: 20px;
    box-shadow: 0 0 30px #ff0000;
    margin-bottom: 30px;
}

/* Input cards */
.input-card {
    background-color: #1e1e1e;
    border: 2px solid #ff0000;
    border-radius: 20px;
    padding: 15px;
    margin-bottom: 15px;
    transition: transform 0.2s;
}
.input-card:hover {
    transform: scale(1.05);
    border-color: #ff7f00;
}

/* Input text */
input, .stNumberInput>div>input {
    background-color: #2a2a2a;
    color: #00ffff;
    border-radius: 10px;
    padding: 8px;
    border: 1px solid #ff0000;
}

/* Selectbox container */
.selectbox-card {
    background-color: #1e1e1e;
    border-radius: 15px;
    border: 2px solid #ff0000;
    padding: 10px;
    color: #00ffff;
    margin-bottom: 10px;
}

/* Predict button */
.stButton>button {
    background: linear-gradient(90deg, #ff0000, #ff7f00);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 20px;
    padding: 15px 30px;
    transition: transform 0.2s ease;
    box-shadow: 0 0 20px #ff7f00;
}
.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 30px #ff0000;
}

/* Price card */
.price-card {
    background: radial-gradient(circle, #ff0000 0%, #ffcc00 70%);
    color: black;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 25px;
    box-shadow: 0 0 30px #ffcc00;
    margin-top: 20px;
}

/* Error */
.stError {
    background-color: #ff4b4b;
    color: white;
    font-size: 1.3rem;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Title and hero image
# ------------------------
st.markdown("<h1>🏎️ Supercar Price Dashboard</h1>", unsafe_allow_html=True)
st.image("https://cdn.pixabay.com/photo/2018/03/01/10/05/car-3190192_1280.jpg", use_column_width=True, caption="High-End Supercar", output_format="auto")

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
# Input fields in cards
# ------------------------
st.subheader("Enter Car Specs:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-card">Year</div>', unsafe_allow_html=True)
    year = st.number_input("", min_value=1980, max_value=2026, value=2022, key="year")

    st.markdown('<div class="input-card">Mileage (km)</div>', unsafe_allow_html=True)
    mileage = st.number_input("", min_value=0, value=5000, key="mileage")

    st.markdown('<div class="input-card">Engine Size (L)</div>', unsafe_allow_html=True)
    engine_size = st.number_input("", min_value=0.5, max_value=8.0, value=3.0, step=0.1, key="engine")

with col2:
    st.markdown('<div class="selectbox-card">Fuel Type</div>', unsafe_allow_html=True)
    fuel_type = st.selectbox("", ["Petrol", "Diesel", "Electric", "Hybrid"], key="fuel")

    st.markdown('<div class="selectbox-card">Transmission</div>', unsafe_allow_html=True)
    transmission = st.selectbox("", ["Manual", "Automatic"], key="transmission")

    st.markdown('<div class="input-card">Brand</div>', unsafe_allow_html=True)
    brand = st.text_input("", value="Ferrari", key="brand")

with col3:
    st.markdown('<div class="input-card">Color</div>', unsafe_allow_html=True)
    color = st.text_input("", value="Red", key="color")

    st.markdown('<div class="input-card">Number of Doors</div>', unsafe_allow_html=True)
    doors = st.number_input("", min_value=2, max_value=5, value=2, key="doors")

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

    try:
        price = model.predict(input_encoded)[0]
        st.markdown(f'<div class="price-card">💰 ${price:,.2f}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
