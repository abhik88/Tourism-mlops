
import pandas as pd
from datasets import load_dataset
import os
from sklearn.preprocessing import LabelEncoder
import joblib

print("🔍 STARTING DATA PREPARATION SCRIPT...")

hf_username = "Abhik19"
dataset_name = "tourism-wellness-data"

try:
    dataset = load_dataset(f"{hf_username}/{dataset_name}")
    df = dataset['train'].to_pandas()
    print("✅ Raw data loaded from Hugging Face Hub.")
except Exception as e:
    print(f"❌ Failed to load dataset from HF: {e}")
    print("⚠️ Creating MOCK data for demonstration.")
    df = pd.DataFrame({'Age': [25, 30, 35], 'MonthlyIncome': [20000, 30000, 40000], 'ProdTaken': [0, 1, 0], 'TypeofContact':['Company Invited', 'Self Inquiry', 'Unknown'], 'DurationOfPitch': [10.0, 15.0, 20.0], 'Occupation':['Salaried', 'Freelancer', 'Student'], 'Gender':['Male', 'Female', 'Fe Male'], 'ProductPitched':['Basic', 'Deluxe', 'Standard'], 'MaritalStatus':['Married', 'Single', 'Divorced'], 'Designation':['Executive', 'Manager', 'VP']}))

# 1. Data Cleaning: Filling missing values
print("⚙️ Cleaning data...")
if 'Age' in df.columns: df['Age'] = df['Age'].fillna(df['Age'].median())
if 'MonthlyIncome' in df.columns: df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
if 'DurationOfPitch' in df.columns: df['DurationOfPitch'] = df['DurationOfPitch'].fillna(df['DurationOfPitch'].median())
if 'TypeofContact' in df.columns: df['TypeofContact'] = df['TypeofContact'].fillna('Unknown')

# 2. Feature Engineering: Encoding categorical features
print("⚙️ Encoding categorical features...")
encoders = {}
cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
            'MaritalStatus', 'Designation']

for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save processed data
processed_data_path = 'data/tourism_cleaned.csv'
df.to_csv(processed_data_path, index=False)
print(f"✅ Cleaned and encoded data saved to {processed_data_path}")

# Save encoders
encoders_path = 'data/encoders.joblib'
joblib.dump(encoders, encoders_path)
print(f"✅ Encoders saved to {encoders_path}")
