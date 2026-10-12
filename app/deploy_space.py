# uploads the app to a hugging face space
# run `hf auth login` first (with a write token), then: python app/deploy_space.py
import os
import shutil
import tempfile

from huggingface_hub import HfApi

SPACE_ID = "DivineItu/pidgin-sentiment-app"
HERE = os.path.dirname(os.path.abspath(__file__))

api = HfApi()
api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", exist_ok=True)

# the space needs everything in one folder: the app, the cleaning code and a readme with the space settings
with tempfile.TemporaryDirectory() as tmp:
    shutil.copy(os.path.join(HERE, "app.py"), tmp)
    shutil.copy(os.path.join(HERE, "requirements.txt"), tmp)
    shutil.copy(os.path.join(HERE, "SPACE_README.md"), os.path.join(tmp, "README.md"))
    shutil.copy(os.path.join(HERE, "..", "src", "preprocess.py"), tmp)
    api.upload_folder(folder_path=tmp, repo_id=SPACE_ID, repo_type="space", commit_message="update app")

print(f"done: https://huggingface.co/spaces/{SPACE_ID}")
