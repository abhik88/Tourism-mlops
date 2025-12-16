
import pandas as pd
from datasets import load_dataset
import os

# Load from HF (Simulating the 'pull' from the previous step)
hf_username = "abhik88"
dataset_name = "tourism-wellness-data"
dataset = load_dataset(f"{hf_username}/{dataset_name}")
df = dataset['train'].to_pandas()

# Clean
df['Age'] = df['Age'].fillna(df['Age'].median())
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

# Save locally for the next step
os.makedirs('data', exist_ok=True)
df.to_csv('data/tourism_cleaned.csv', index=False)
print("✅ Data Prepared and Saved Locally")
