# uploads the app to a hugging face space
# gradio spaces need a paid plan now, static spaces are free, so the space is a static page (static/index.html)
# that runs the model in the browser. export_onnx.py has to be run once before this.
# run `hf auth login` first (with a write token), then from the app folder: python deploy_space.py
import os
import shutil
import tempfile

from huggingface_hub import HfApi

SPACE_ID = "DivineItu/pidgin-sentiment-app"
HERE = os.path.dirname(os.path.abspath(__file__))

api = HfApi()
api.create_repo(SPACE_ID, repo_type="space", space_sdk="static", exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    shutil.copy(os.path.join(HERE, "static", "index.html"), tmp)
    shutil.copy(os.path.join(HERE, "SPACE_README.md"), os.path.join(tmp, "README.md"))
    api.upload_folder(folder_path=tmp, repo_id=SPACE_ID, repo_type="space", commit_message="update app")

print(f"done: https://huggingface.co/spaces/{SPACE_ID}")
