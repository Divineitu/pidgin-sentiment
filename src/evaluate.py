# metrics, confusion matrix and the experiment log, shared by all the model notebooks
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from preprocess import LABELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOG_PATH = os.path.join(RESULTS_DIR, "experiments.csv")


def get_scores(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        # the afrisenti paper reports weighted f1, so i log it too to compare
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }
    for label, f1 in zip(LABELS, per_class):
        scores[f"f1_{label}"] = f1
    return {k: round(float(v), 4) for k, v in scores.items()}


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))


def plot_confusion(y_true, y_pred, title, save_name=None):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    if save_name:
        os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, "figures", save_name), dpi=150)
    plt.show()


def log_result(exp_id, model, split, scores, notes=""):
    # one row per experiment per split; if i rerun an experiment its old row gets replaced
    row = {"exp_id": exp_id, "model": model, "split": split, **scores, "notes": notes}
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        log = log[~((log["exp_id"] == exp_id) & (log["split"] == split))]
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(LOG_PATH, index=False)
