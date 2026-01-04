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
# Custom CSS for sporty theme
# ------------------------
st.markdown("""
<style>
body {
    background-color: #0f0f0f;
    color: #f0f0f0;
}
h1 {
    color: #ff4b4b;
    text-align: center;
    font-size: 3rem;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px 20px;
    margin-top: 10px;
}
.stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div {
    border-radius: 8px;
    padding: 8px;
    background-color: #1e1e1e;
    color: white;
}
.stAlert {
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Title and image
# ------------------------
st.markdown("<h1>🏎️ Sports Car Price Predictor</h1>", unsafe_allow_html=True)
st.image("https://cdn.pixabay.com/photo/2017/09/14/10/43/ferrari-2748582_1280.jpg", use_column_width=True)

# ------------------------
# Load trained model
# ------------------------
model_path = "simple_car_price_model.pkl"  # Must be in the same folder

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
else:
    st.error(f"❌ Model file not found. Upload '{model_path}' to the repo.")
    st.stop()

# ------------------------
# Input fields
# ------------------------
st.subheader("Enter Your Car Details:")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1980, max_value=2026, value=2022)
    mileage = st.number_input("Mileage (km)", min_value=0, value=5000)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=3.0, step=0.1)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    brand = st.text_input("Brand", value="Ferrari")

with col3:
    color = st.text_input("Color", value="Red")  # optional feature for dummy variable
    doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=2)

# ------------------------
# Predict button
# ------------------------
if st.button("Predict Price"):
    # Create a dataframe matching the training features
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

    # One-hot encode categorical features (matching training)
    # This assumes you trained model with pd.get_dummies(drop_first=True)
    all_features = model.coef_.shape[0]  # number of features in model
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align input columns to model columns (fill missing with 0)
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_encoded.columns
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Prediction
    try:
        price = model.predict(input_encoded)[0]
        st.success(f"💰 Estimated Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

