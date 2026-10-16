"""
Train and evaluate the green-time regression models (Phase 6).

Trains BOTH a Linear Regression baseline and a Random Forest Regressor
on the simulation-based dataset (Phase 5), evaluates both with
MAE / RMSE / R^2, saves comparison + feature-importance + actual-vs-
predicted plots, and saves the Random Forest (the primary model per the
project spec) as models/traffic_model.pkl for use in Phase 7. NOTE: the
labels come from a LINEAR rule plus noise, so Linear Regression can score
equal or slightly higher; results are reported as they come out, unmodified.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed — we only save PNG files
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import (
    TRAFFIC_DATASET_PATH, TRAFFIC_MODEL_PATH, TRAFFIC_METRICS_PATH,
    TRAFFIC_MODEL_FEATURES, DATA_FOLDER,
)
from ml.dataset_generator import generate_dataset, save_dataset

PLOTS_DIR = os.path.join(DATA_FOLDER, "traffic", "plots")


def _evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


def train_and_evaluate():
    # Step 1: ensure the dataset exists (regenerate if missing)
    if not os.path.exists(TRAFFIC_DATASET_PATH):
        save_dataset(generate_dataset())

    df = pd.read_csv(TRAFFIC_DATASET_PATH)
    X = df[TRAFFIC_MODEL_FEATURES]
    y = df["optimal_green_time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Linear Regression baseline ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = _evaluate(y_test, lr_pred)

    # --- Random Forest (primary model) ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = _evaluate(y_test, rf_pred)

    # Random Forest is our primary model per project spec (section 16).
    # We still report both for the comparison requirement (section 19).
    joblib.dump(rf, TRAFFIC_MODEL_PATH)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Actual vs Predicted plot (Random Forest) ---
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, rf_pred, alpha=0.5, color="#4ade80")
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual green time (sec)")
    plt.ylabel("Predicted green time (sec)")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=110)
    plt.close()

    # --- Feature importance plot ---
    importances = pd.Series(rf.feature_importances_, index=TRAFFIC_MODEL_FEATURES)
    importances = importances.sort_values()
    plt.figure(figsize=(6, 4))
    importances.plot(kind="barh", color="#4ade80")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=110)
    plt.close()

    metrics = {
        "linear_regression": lr_metrics,
        "random_forest": rf_metrics,
        "features": TRAFFIC_MODEL_FEATURES,
        "feature_importance": importances.round(4).to_dict(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    with open(TRAFFIC_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = train_and_evaluate()
    print(json.dumps(m, indent=2))
