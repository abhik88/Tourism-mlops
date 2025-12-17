
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
import sys
import traceback

# Configuration
hf_username = "Abhik19"
model_repo_id = f"{hf_username}/tourism-prediction-model"
encoders_filename = "encoders.joblib"
model_filename = "best_model.joblib"

# --- Setup ---
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("🗺️ Wellness Tourism Package Prediction")
st.markdown("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Add startup indicator
with st.spinner("🔄 Loading prediction model..."):
    print(f"[STARTUP] App starting up...", file=sys.stderr)
    print(f"[STARTUP] Attempting to load from repo: {model_repo_id}", file=sys.stderr)

@st.cache_resource
def load_artifacts_from_hub():
    print(f"[LOAD] Starting artifact loading...", file=sys.stderr)
    try:
        print(f"[LOAD] Downloading model from {model_repo_id}...", file=sys.stderr)
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        print(f"[LOAD] Model downloaded to: {model_path}", file=sys.stderr)
        model = joblib.load(model_path)
        print(f"[LOAD] Model loaded successfully!", file=sys.stderr)

        print(f"[LOAD] Downloading encoders from {model_repo_id}...", file=sys.stderr)
        encoders_path = hf_hub_download(repo_id=model_repo_id, filename=encoders_filename)
        print(f"[LOAD] Encoders downloaded to: {encoders_path}", file=sys.stderr)
        encoders = joblib.load(encoders_path)
        print(f"[LOAD] Encoders loaded successfully!", file=sys.stderr)

        return model, encoders
    except Exception as e:
        error_msg = f"Failed to load model or encoders: {str(e)}"
        traceback_msg = traceback.format_exc()
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        print(f"[ERROR] Traceback: {traceback_msg}", file=sys.stderr)
        st.error(f"❌ {error_msg}")
        st.info("🔍 Ensure the model and encoders are pushed to Hugging Face Hub correctly.")
        st.code(traceback_msg, language="python")
        return None, None

model, encoders = load_artifacts_from_hub()

if model and encoders:
    st.success("✅ Prediction System Activated: Model and Encoders Loaded")
    print(f"[SUCCESS] All artifacts loaded. App ready!", file=sys.stderr)
else:
    st.warning("⚠️ App Running in Demo Mode - Model Not Loaded")
    st.info('''**Possible Issues:**
1. Model repository might not exist: `{model_repo_id}`
2. Model files (best_model.joblib, encoders.joblib) not uploaded
3. Repository might be private and needs authentication

**To fix:**
- Ensure the model training pipeline has run successfully
- Verify files exist at: https://huggingface.co/{model_repo_id}
''')
    st.stop()

# Define feature names in the exact order the model expects
# This order is derived from X = df.drop(['ProdTaken', 'CustomerID'], axis=1).columns
feature_names = [
    'Unnamed: 0', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
    'Occupation', 'Gender', 'NumberOfPersonVisiting', 'ProductPitched',
    'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation',
    'MonthlyIncome'
]

# --- Input Fields ---
st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 90, 30)
    monthly_income = st.number_input("Monthly Income", 5000.0, 100000.0, 20000.0, step=100.0)
    number_of_person_visiting = st.slider("Number of People Visiting", 1, 10, 1)
    number_of_trips = st.slider("NumberOfTrips Annually", 0, 20, 2)
    number_of_children_visiting = st.slider("Number of Children Visiting (under 5)", 0, 5, 0)
    own_car = st.selectbox("Owns a Car?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No") # Store 0/1

with col2:
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    preferred_property_star = st.slider("Preferred Property Star (1-5)", 1, 5, 3)
    passport = st.selectbox("Has a Passport?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No") # Store 0/1
    city_tier = st.selectbox("City Tier", [1, 2, 3])

with col3:
    # Categorical features requiring encoding - use the categories seen during training
    # These options should ideally come from `le.classes_` from the loaded encoders
    # For now, hardcode based on common sense/data description
    typeof_contact_options = ['Company Invited', 'Self Inquiry', 'Unknown']
    typeof_contact = st.selectbox("Type of Contact", typeof_contact_options)

    occupation_options = ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Student', 'Unemployed']
    occupation = st.selectbox("Occupation", occupation_options)

    gender_options = ['Male', 'Female', 'Fe Male']
    gender = st.selectbox("Gender", gender_options)

    product_pitched_options = ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King']
    product_pitched = st.selectbox("Product Pitched", product_pitched_options)

    marital_status_options = ['Married', 'Single', 'Divorced']
    marital_status = st.selectbox("Marital Status", marital_status_options)

    designation_options = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
    designation = st.selectbox("Designation", designation_options)

# --- Prediction Logic ---
if st.button("Predict Purchase Likelihood"):
    # Create a dictionary for raw inputs
    raw_inputs = {
        'Unnamed: 0': 0, # This was likely an index column, setting to 0 for prediction
        'Age': age,
        'TypeofContact': typeof_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation,
        'MonthlyIncome': monthly_income
    }

    # Transform categorical features using loaded encoders
    transformed_inputs = raw_inputs.copy()
    for col, le in encoders.items():
        if col in transformed_inputs:
            # Convert to string for LabelEncoder
            input_val = str(transformed_inputs[col])
            # Handle potential unseen labels gracefully
            try:
                if input_val in le.classes_:
                    transformed_inputs[col] = le.transform([input_val])[0]
                else:
                    # Use the first class as default for unseen categories
                    st.warning(f"Warning: Unseen category '{input_val}' for feature '{col}'. Using default encoding.")
                    transformed_inputs[col] = 0
            except Exception as e:
                st.warning(f"Warning: Error encoding '{col}': {e}. Using default value 0.")
                transformed_inputs[col] = 0

    # Create DataFrame for prediction, ensuring correct order
    df_input = pd.DataFrame([transformed_inputs], columns=feature_names)

    try:
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1] # Probability of '1' (purchase)

        st.divider()
        st.subheader("Prediction Results")
        if prediction == 1:
            st.success(f"### This customer is likely to purchase! (Probability: {probability:.2f})")
        else:
            st.info(f"### This customer is not likely to purchase. (Probability of Purchase: {probability:.2f})")
        st.metric("Purchase Probability", f"{probability:.2%}")

        st.write("---")
        st.write("Input Data for Prediction:")
        st.dataframe(df_input)

    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        st.info("Debug Info: Input shape is " + str(df_input.shape))
        st.dataframe(df_input) # Show the malformed input DataFrame for debugging
