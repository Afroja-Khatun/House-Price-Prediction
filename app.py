import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# --- MODEL LOADING ---
# MUST match the filename created by train_models.py
MODEL_FILE = "house_price_model.pkl"

def load_model():
    if os.path.exists(MODEL_FILE):
        return pickle.load(open(MODEL_FILE, "rb"))
    return None

model = load_model()

st.title("🏠 House Price Prediction")

# Error handling if model is missing
if model is None:
    st.error(f"⚠️ Model file '{MODEL_FILE}' not found! Please run 'python train_models.py' first.")
    st.stop()

st.write("Enter House Details:")

# Inputs (Order must match training features)
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0.0, value=3.0)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0)
    sqft_living = st.number_input("Sqft Living", min_value=0.0, value=2000.0)
    sqft_lot = st.number_input("Sqft Lot", min_value=0.0, value=5000.0)
    floors = st.number_input("Floors", min_value=0.0, value=1.0)
    waterfront = st.selectbox("Waterfront (0=No, 1=Yes)", [0, 1])
    view = st.slider("View (0-4)", 0, 4, 0)
    condition = st.slider("Condition (1-5)", 1, 5, 3)

with col2:
    grade = st.slider("Grade (1-13)", 1, 13, 7)
    sqft_basement = st.number_input("Sqft Basement", min_value=0.0, value=0.0)
    lat = st.number_input("Latitude", format="%.6f", value=47.5112)
    long = st.number_input("Longitude", format="%.6f", value=-122.2570)
    sqft_living15 = st.number_input("Sqft Living15", min_value=0.0, value=1800.0)
    sqft_lot15 = st.number_input("Sqft Lot15", min_value=0.0, value=5000.0)
    house_age = st.number_input("House Age", min_value=0.0, value=20.0)
    is_renovated = st.selectbox("Is Renovated (0=No, 1=Yes)", [0, 1])

if st.button("Predict Price"):
    input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                            waterfront, view, condition, grade, sqft_basement,
                            lat, long, sqft_living15, sqft_lot15,
                            house_age, is_renovated]])

    prediction = model.predict(input_data)
    
    price = float(prediction[0])
    st.success(f"### Estimated House Price: ${price:,.2f}")