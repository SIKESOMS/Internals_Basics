"""
Task 1 — Experiment Tracking & Model Comparison
Trains Lasso and RandomForest, logs to MLflow, saves results/step1_s1.json
"""

import json
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "swimsync-lap-time-seconds"
FEATURES = ["stroke_rate", "drag_coefficient", "turn_time_ms", "pool_length_m"]
TARGET = "lap_time_seconds"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(mae, 4), round(rmse, 4)

def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("domain", "aquatic_sports")
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae, rmse = compute_metrics(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # save locally too
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}.pkl"))
        print(f"  {model_name:15s}  MAE={mae:.4f}  RMSE={rmse:.4f}  run_id={run.info.run_id}")
        return mae, rmse, run.info.run_id

def main():
    print("=== Task 1: Experiment Tracking & Model Comparison ===")
    X_train, X_test, y_train, y_test = load_data()

    models_cfg = [
        (Lasso(alpha=1.0, random_state=42),   "Lasso",        {"alpha": 1.0}),
        (RandomForestRegressor(n_estimators=100, random_state=42),
         "RandomForest", {"n_estimators": 100, "random_state": 42}),
    ]

    records = []
    for model, name, params in models_cfg:
        mae, rmse, _ = train_and_log(model, name, params, X_train, X_test, y_train, y_test)
        records.append({"name": name, "mae": mae, "rmse": rmse})

    best = min(records, key=lambda r: r["rmse"])
    result = {
        "experiment_name": EXPERIMENT_NAME,
        "models": records,
        "best_model": best["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best["rmse"],
    }

    out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()