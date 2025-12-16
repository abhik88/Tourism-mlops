
import pandas as pd
from datasets import load_dataset
import os

hf_username = "Abhik19"
dataset_name = "tourism-wellness-data"
dataset = load_dataset(f"{hf_username}/{dataset_name}")
df = dataset['train'].to_pandas()

# Clean
if 'Age' in df.columns: df['Age'] = df['Age'].fillna(df['Age'].median())
if 'MonthlyIncome' in df.columns: df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

os.makedirs('data', exist_ok=True)
df.to_csv('data/tourism_cleaned.csv', index=False)
print("✅ Data Prepared and Saved Locally")
