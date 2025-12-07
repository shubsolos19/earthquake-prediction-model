import streamlit as st
import os
import numpy as np

# Prefer joblib for scikit-learn models
from joblib import load

# Get model path relative to this file
model_path = os.path.join(os.path.dirname(__file__), "../model/earthquake_model.pkl")

# Load the model
model = load(model_path)

# Streamlit UI
st.title("🌍 Earthquake Depth Prediction")
st.write("Enter the earthquake parameters below:")

# Inputs
magnitude = st.number_input("Magnitude", min_value=0.0, max_value=10.0, value=5.0)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.0)

# Predict button
if st.button("Predict Depth"):
    X = np.array([[magnitude, latitude, longitude]])
    depth = model.predict(X)[0]

    st.success(f"Predicted Earthquake Depth: **{depth:.2f} km**")
