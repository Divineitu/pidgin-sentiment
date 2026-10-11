# downloads the nigerian pidgin (pcm) part of afrisenti from hugging face
# the data is cc by-nc-sa so i don't keep it in the repo, run this first
import os
import urllib.request

BASE_URL = "https://huggingface.co/datasets/masakhane/afrisenti/resolve/main/data/pcm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ["train", "dev", "test"]:
        out_path = os.path.join(OUT_DIR, f"{split}.tsv")
        if os.path.exists(out_path):
            print(f"{split}.tsv already there, skipping")
            continue
        urllib.request.urlretrieve(f"{BASE_URL}/{split}.tsv", out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    download()
