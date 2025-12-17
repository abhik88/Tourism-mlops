
from huggingface_hub import HfApi, login
import os

print("🔍 STARTING DEPLOYMENT SCRIPT...")

hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token)
    print("✅ Logged in to Hugging Face.")
else:
    print("❌ ERROR: HF_TOKEN is missing! Check GitHub Secrets.")
    exit(1)

api = HfApi()
space_id = "Abhik19/tourism-prediction-app"
deploy_folder = "deployment"

print(f"🚀 Deploying to {space_id}...")
try:
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=deploy_folder, repo_id=space_id, repo_type="space")
    print("✅ Deployment Successful")
except Exception as e:
    print(f"❌ Error during deployment: {e}")
    exit(1)
