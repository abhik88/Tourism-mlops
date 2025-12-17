
import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import HfApi, login

print("🔍 STARTING MODEL TRAINING SCRIPT...")

hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token)
    print("✅ Logged in to Hugging Face.")
else:
    print("❌ ERROR: HF_TOKEN is missing! Check GitHub Secrets.")
    exit(1)

# Load Data
try:
    df = pd.read_csv('data/tourism_cleaned.csv')
    print("✅ Cleaned data loaded locally.")
except Exception as e:
    print(f"❌ Failed to load tourism_cleaned.csv: {e}")
    print("⚠️ Creating MOCK data for demonstration.")
    # Replicate the structure of the processed data as much as possible for the fallback
    df = pd.DataFrame({   'Unnamed: 0':[0,1,2], 'CustomerID':[200000, 200001, 200002], 'ProdTaken':[0,1,0], 'Age':[25,30,35], 
    'TypeofContact':[0,1,0], 'CityTier':[1,2,3], 'DurationOfPitch':[10,15,20], 'Occupation':[0,1,2], 
    'Gender':[0,1,0], 'NumberOfPersonVisiting':[1,2,1], 'ProductPitched':[0,1,0], 'PreferredPropertyStar':[3,4,3], 
    'MaritalStatus':[0,1,0], 'NumberOfTrips':[1,2,1], 'Passport':[0,1,0], 'PitchSatisfactionScore':[3,4,3], 
    'OwnCar':[0,1,0], 'NumberOfChildrenVisiting':[0,1,0], 'Designation':[0,1,2], 'MonthlyIncome':[20000,30000,40000]})

# Load Encoders
try:
    encoders = joblib.load('data/encoders.joblib')
    print("✅ Encoders loaded locally.")
except Exception as e:
    print(f"❌ Failed to load encoders.joblib: {e}")
    print("⚠️ Proceeding without encoders (might affect deployment).")
    encoders = {}

# Define features and target
X = df.drop(['ProdTaken', 'CustomerID'], axis=1, errors='ignore')
y = df['ProdTaken']

# Split data (using a small subset for quick CI/CD runs)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model for CI/CD demonstration
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate (basic metrics for CI/CD) - Full evaluation in main notebook
preds = model.predict(X_test)
f1 = f1_score(y_test, preds, average='weighted')
print(f"✅ Model F1-Score (weighted): {f1:.4f}")

# Save Artifacts to 'model_dir'
os.makedirs('model_dir', exist_ok=True)
model_path = 'model_dir/best_model.joblib'
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Save encoders to 'model_dir' for deployment alongside the model
encoders_deploy_path = 'model_dir/encoders.joblib'
joblib.dump(encoders, encoders_deploy_path)
print(f"✅ Encoders saved to {encoders_deploy_path} for deployment.")

# Push to HF Model Hub
api = HfApi()
repo_id = f"Abhik19/tourism-prediction-model"

print(f"🚀 Pushing model and encoders to {repo_id}...")
try:
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path="model_dir",
        repo_id=repo_id,
        repo_type="model",
        commit_message="CI/CD: Updated model and encoders"
    )
    print("✅ Model and Encoders Pushed successfully to Hugging Face Model Hub!")
except Exception as e:
    print(f"❌ PUSH FAILED: {e}")
    exit(1)
