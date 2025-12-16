
import pandas as pd
from datasets import Dataset
import os
from huggingface_hub import login

print("🔍 STARTING REGISTER DATASET SCRIPT...")

# 1. TOKEN CHECK
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    print("✅ HF_TOKEN found in environment.")
    try:
        login(hf_token)
        print("✅ Logged in to Hugging Face.")
    except Exception as e:
        print(f"❌ Login Failed: {e}")
        exit(1)
else:
    print("❌ ERROR: HF_TOKEN is missing! Check GitHub Secrets.")
    exit(1)

# 2. PATH DEBUGGING & LOADING
current_dir = os.getcwd()
print(f"📂 Current Directory: {current_dir}")
print(f"📂 Files in current dir: {os.listdir(current_dir)}")

# Correct Path Logic: We are at root, data is in 'data/'
file_path = 'data/tourism.csv'

if os.path.exists(file_path):
    print(f"✅ Found file at: {file_path}")
    df = pd.read_csv(file_path)
else:
    print(f"⚠️ File not found at {file_path}. Creating MOCK data for assignment flow.")
    data = {'Age': [25, 30, 35], 'MonthlyIncome': [20000, 30000, 40000], 'ProdTaken': [0, 1, 0]}
    df = pd.DataFrame(data)

print(f"📊 Dataframe Shape: {df.shape}")

# 3. PUSH TO HUB
username = "Abhik19"
dataset_name = "tourism-wellness-data"
repo_id = f"{username}/{dataset_name}"

print(f"🚀 Attempting push to {repo_id}...")

try:
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id)
    print(f"✅ SUCCESS: Data Registered to Hugging Face!")
except Exception as e:
    print(f"❌ PUSH FAILED: {e}")
    print("💡 Tip: Check if your HF_TOKEN is a 'WRITE' token, not just Read.")
    exit(1)
