
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
df = pd.read_csv('data/tourism_cleaned.csv')

# Encode
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split
X = df.drop(['ProdTaken'], axis=1)
# Handle potential missing columns in mock runs
if 'CustomerID' in X.columns: X = X.drop('CustomerID', axis=1)
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Metrics
preds = model.predict(X_test)
metrics = {"accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds)}

# Save Artifacts
os.makedirs('model_dir', exist_ok=True)
joblib.dump(model, 'model_dir/best_model.joblib')
joblib.dump(encoders, 'model_dir/encoders.joblib')
with open('model_dir/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Push to HF
api = HfApi()
repo_id = "abhik88/tourism-prediction-model"
api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_folder(folder_path="model_dir", repo_id=repo_id, repo_type="model")
print(f"✅ Model Trained & Pushed to {repo_id}")
