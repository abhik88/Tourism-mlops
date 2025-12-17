
from huggingface_hub import HfApi, login
import os
from datetime import datetime

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

# Generate timestamp for unique commit
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
commit_msg = f"CI/CD Deployment: Updated app at {timestamp}"

print(f"🚀 Deploying to {space_id}...")
try:
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    # Upload with commit message that includes timestamp to force update
    api.upload_folder(
        folder_path=deploy_folder,
        repo_id=space_id,
        repo_type="space",
        commit_message=commit_msg,
        delete_patterns=["*.py", "*.toml", "*.txt", "*.md", "Dockerfile"]  # Delete old files first
    )
    print("✅ Deployment Successful")
    print(f"📝 Commit: {commit_msg}")
except Exception as e:
    print(f"❌ Error during deployment: {e}")
    import traceback
    print(traceback.format_exc())
    exit(1)
