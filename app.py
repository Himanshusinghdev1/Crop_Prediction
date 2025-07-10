import streamlit as st
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load("crop_yield_model.pkl")

# Streamlit app UI
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")
st.title("🌱 Crop Yield Prediction System")
st.markdown("""
Welcome to the Crop Yield Prediction System!  
Provide environmental and soil parameters to predict the **most suitable crop**.
""")

# Input form
with st.form("input_form"):
    st.subheader("Enter Soil & Environmental Parameters:")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Himanshu Singh")
