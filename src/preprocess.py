# text cleaning used by the notebooks and the app, so both clean text the same way
import re

import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# leftover invisible bits from emojis that were removed (variation selector, zero width joiner etc.)
INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u2060-\u2064\ufe0e\ufe0f]")
# 3 or more of the same letter in a row, e.g. "sweeeet" or "lmaooooo"
REPEATED_CHARS = re.compile(r"(\w)\1{2,}")


def clean_text(text):
    text = str(text).lower()
    text = INVISIBLE_CHARS.sub("", text)
    text = REPEATED_CHARS.sub(r"\1\1", text)  # keep 2 so "sweeeet" -> "sweet"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(path):
    df = pd.read_csv(path, sep="\t")
    df["text"] = df["tweet"].apply(clean_text)
    return df


def dedupe_train(train):
    # same tweet with different labels = annotators disagreed, we can't trust it so drop it
    n_labels = train.groupby("text")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    train = train[~train["text"].isin(conflicting)]
    # exact copies with the same label, keep one
    train = train.drop_duplicates(subset="text").reset_index(drop=True)
    return train, len(conflicting)
