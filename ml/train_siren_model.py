"""
Train and evaluate the siren classifier (Phase 10).

Pipeline:  audio -> MFCC features -> Random Forest classifier -> siren ON/OFF

Trained on the small synthetic dataset in data/audio/{siren,no_siren}/
(see ml/generate_siren_dataset.py docstring for the honesty disclosure
about this dataset being synthetic, not real recordings).
"""

import os
import glob
import json

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.config import SIREN_DATA_DIR, SIREN_MODEL_PATH, DATA_FOLDER
from audio.feature_extractor import extract_mfcc_features

METRICS_PATH = os.path.join(DATA_FOLDER, "audio", "siren_model_metrics.json")


def build_feature_matrix():
    X, y = [], []

    siren_files = glob.glob(os.path.join(SIREN_DATA_DIR, "siren", "*.wav"))
    no_siren_files = glob.glob(os.path.join(SIREN_DATA_DIR, "no_siren", "*.wav"))

    for f in siren_files:
        X.append(extract_mfcc_features(f))
        y.append(1)  # siren ON

    for f in no_siren_files:
        X.append(extract_mfcc_features(f))
        y.append(0)  # siren OFF

    return np.array(X), np.array(y)


def train_and_evaluate():
    X, y = build_feature_matrix()

    if len(X) < 10:
        raise RuntimeError(
            "Not enough audio samples. Run: python -m ml.generate_siren_dataset"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["no_siren", "siren"], output_dict=True)

    os.makedirs(os.path.dirname(SIREN_MODEL_PATH), exist_ok=True)
    joblib.dump(clf, SIREN_MODEL_PATH)

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "note": "Trained on a small SYNTHETIC dataset (see ml/generate_siren_dataset.py). "
                "Not validated against real ambulance/fire-truck siren recordings.",
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = train_and_evaluate()
    print(json.dumps(m, indent=2))
