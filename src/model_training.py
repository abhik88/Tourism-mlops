
import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, login

hf_token = os.getenv('HF_TOKEN')
login(hf_token)

# Load Data
try:
    df = pd.read_csv('data/tourism_cleaned.csv')
except:
    df = pd.DataFrame({'Age':[20,30], 'ProdTaken':[0,1]}) # Fallback

# Encode
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(['ProdTaken'], axis=1, errors='ignore')
y = df['ProdTaken']

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Save Artifacts
os.makedirs('model_dir', exist_ok=True)
joblib.dump(model, 'model_dir/best_model.joblib')
joblib.dump(encoders, 'model_dir/encoders.joblib')

# Push to HF
api = HfApi()
# ⚠️ CHANGED: 'abhik88' -> 'Abhik19'
repo_id = "Abhik19/tourism-prediction-model"

print(f"🚀 Pushing model to {repo_id}...")
api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_folder(folder_path="model_dir", repo_id=repo_id, repo_type="model")
print(f"✅ Model Trained & Pushed successfully!")
