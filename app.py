# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load your trained model
model = joblib.load("car_price_model.pkl")

# Page config
st.set_page_config(
    page_title="🔥 Sports Car Predictor 🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for sporty UI ---
st.markdown("""
<style>
/* Background gradient */
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Card style */
.card {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
    margin-bottom: 20px;
}

/* Big buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(45deg, #ff4b2b, #ff416c);
    transform: scale(1.05);
}

/* Headers */
h1, h2, h3 {
    color: #ffcc00;
    text-shadow: 2px 2px 5px #000;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>🏎️ Sports Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter your car details to get a turbo-charged prediction!</h3>", unsafe_allow_html=True)

# --- Inputs in columns ---
col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", ["Ferrari", "Lamborghini", "Porsche", "McLaren"])
    year = st.slider("Year of Manufacture", 2000, 2026, 2018)

with col2:
    horsepower = st.number_input("Horsepower (HP)", min_value=100, max_value=1500, value=500)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=10000)

with col3:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])

# --- Predict button ---
if st.button("🚀 Predict Price"):
    # Example preprocessing (update to your model's preprocessing)
    df = pd.DataFrame({
        "brand": [brand],
        "year": [year],
        "horsepower": [horsepower],
        "mileage": [mileage],
        "fuel_type": [fuel_type],
        "transmission": [transmission]
    })

    # If your model has a pipeline with preprocessing, just do:
    prediction = model.predict(df)[0]

    st.markdown(f"<h2>💰 Predicted Price: ${prediction:,.0f}</h2>", unsafe_allow_html=True)

# --- Extra sporty touches ---
st.markdown("<hr style='border:2px solid #ff416c;'>", unsafe_allow_html=True)
st.markdown("<h3>🔥 Top Sports Cars Insights 🔥</h3>", unsafe_allow_html=True)

# Example visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Dummy data for visualization
cars = ["Ferrari", "Lamborghini", "Porsche", "McLaren"]
prices = [250000, 300000, 200000, 350000]

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=cars, y=prices, palette="flare", ax=ax)
ax.set_ylabel("Average Price ($)")
ax.set_xlabel("Brand")
st.pyplot(fig)
