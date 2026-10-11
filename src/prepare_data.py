# downloads the raw data and makes the cleaned splits used by every model notebook
# (same steps as the end of notebooks/01_data_and_eda.ipynb, as a script so colab can run it)
import os

from download_data import download
from preprocess import load_split, dedupe_train

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def prepare():
    download()
    raw = os.path.join(DATA_DIR, "raw")
    train = load_split(os.path.join(raw, "train.tsv"))
    dev = load_split(os.path.join(raw, "dev.tsv"))
    test = load_split(os.path.join(raw, "test.tsv"))

    seen_in_train = set(train["text"])
    train_clean, n_conflict = dedupe_train(train)
    # dev and clean test only keep tweets the model never saw in training
    dev_clean = dev[~dev["text"].isin(seen_in_train)].drop_duplicates("text").reset_index(drop=True)
    test_clean = test[~test["text"].isin(seen_in_train)].reset_index(drop=True)

    out = os.path.join(DATA_DIR, "processed")
    os.makedirs(out, exist_ok=True)
    cols = ["text", "label"]
    train_clean[cols].to_csv(os.path.join(out, "train.csv"), index=False)
    dev_clean[cols].to_csv(os.path.join(out, "dev.csv"), index=False)
    test[cols].to_csv(os.path.join(out, "test_official.csv"), index=False)
    test_clean[cols].to_csv(os.path.join(out, "test_clean.csv"), index=False)

    print(f"dropped {n_conflict} tweets with conflicting labels")
    print(f"train {len(train_clean)} | dev {len(dev_clean)} | test official {len(test)} | test clean {len(test_clean)}")


if __name__ == "__main__":
    prepare()
