
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# 1. SETUP
st.title("Predictive Analytics: Tourism Packages")

@st.cache_resource
def load_model_from_hub():
    try:
        model_path = hf_hub_download(repo_id="Abhik19/tourism-prediction-model", filename="best_model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model_from_hub()

if model:
    st.success("✅ Model Activated")

# 2. INPUTS
st.subheader("Customer Details")
age = st.number_input("Age", 18, 90, 30)
income = st.number_input("Monthly Income", 1000, 100000, 20000)
pitch = st.number_input("Duration of Pitch (minutes)", 5, 120, 15)
passport = st.selectbox("Has Passport?", [0, 1])

# 3. PREDICTION LOGIC
if st.button("Predict"):
    # Fix: Set vector size to 19 to match model expectation (includes Unnamed: 0)
    input_data = [0] * 19

    # Mapping based on typical Pandas Column order after 'Unnamed: 0' inclusion:
    # 0: Unnamed: 0
    # 1: Age
    # 4: DurationOfPitch
    # 13: Passport
    # 18: MonthlyIncome

    input_data[1] = age          # Age
    input_data[18] = income      # MonthlyIncome
    input_data[4] = pitch        # DurationOfPitch
    input_data[13] = passport    # Passport

    # Create DataFrame
    df_input = pd.DataFrame([input_data])

    # Predict
    try:
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        st.divider()
        if prediction == 1:
            st.success(f"### Prediction: Will Purchase (Prob: {probability:.2f})")
        else:
            st.warning(f"### Prediction: No Purchase (Prob: {probability:.2f})")

    except ValueError as e:
        st.error(f"Shape Error: {e}")
        st.info("Debug Info: Input shape is " + str(df_input.shape))
