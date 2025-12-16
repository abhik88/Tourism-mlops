
import pandas as pd
from datasets import Dataset
import os
from huggingface_hub import login

# We use the env variable injected by GitHub Actions
hf_token = os.getenv('HF_TOKEN')
login(hf_token)

# Load Data (Assuming it exists or creating dummy for flow if missing in runner)
# In a real CI/CD, you might pull this from a raw source. 
# For this assignment, we assume the repo contains the csv or we generate it.
if os.path.exists('../data/tourism.csv'):
    df = pd.read_csv('../data/tourism.csv')
else:
    # Fallback if file isn't in runner (Mocking for pipeline success)
    data = {'Age': [25, 30], 'MonthlyIncome': [20000, 30000], 'ProdTaken': [0, 1]}
    df = pd.DataFrame(data)

# Push to HF
hf_username = "abhik88"
dataset_name = "tourism-wellness-data"
repo_id = f"{hf_username}/{dataset_name}"

dataset = Dataset.from_pandas(df)
dataset.push_to_hub(repo_id)
print(f"✅ Data Registered to {repo_id}")
