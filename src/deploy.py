
from huggingface_hub import HfApi, login
import os

hf_token = os.getenv('HF_TOKEN')
login(hf_token)

api = HfApi()
space_id = "abhik88/tourism-prediction-app"
deploy_folder = "deployment"

print(f"🚀 Deploying to {space_id}...")
try:
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=deploy_folder, repo_id=space_id, repo_type="space")
    print("✅ Deployment Successful")
except Exception as e:
    print(f"❌ Error: {e}")
