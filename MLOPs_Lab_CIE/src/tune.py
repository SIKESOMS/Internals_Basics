"""
Task 2 — Hyperparameter Tuning
Random search + 3-fold CV on RandomForest, nested MLflow runs, saves results/step2_s2.json
"""

import json
import os
import random
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "swimsync-lap-time-seconds"
FEATURES = ["stroke_rate", "drag_coefficient", "turn_time_ms", "pool_length_m"]
TARGET = "lap_time_seconds"

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4],
}
N_ITER = 6          # random sample 6 combos out of 18
N_FOLDS = 3
PARENT_RUN_NAME = "tuning-swimsync"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def random_search_configs(grid, n_iter, seed=42):
    random.seed(seed)
    configs = []
    for _ in range(n_iter):
        cfg = {k: random.choice(v) for k, v in grid.items()}
        configs.append(cfg)
    return configs

def main():
    print("=== Task 2: Hyperparameter Tuning ===")
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(EXPERIMENT_NAME)
    configs = random_search_configs(PARAM_GRID, N_ITER)

    best_rmse = float("inf")
    best_params = {}
    best_mae = 0.0
    best_cv_mae = 0.0

    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
        mlflow.set_tag("domain", "aquatic_sports")

        for i, cfg in enumerate(configs):
            with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
                model = RandomForestRegressor(random_state=42, **cfg)

                # 3-fold CV MAE on training set
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=N_FOLDS,
                    scoring="neg_mean_absolute_error"
                )
                cv_mae = round(-cv_scores.mean(), 4)

                # fit on full train, eval on test
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = round(mean_absolute_error(y_test, preds), 4)
                rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)

                mlflow.log_params(cfg)
                mlflow.log_metric("cv_mae", cv_mae)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)

                print(f"  trial {i+1}: {cfg}  cv_mae={cv_mae}  rmse={rmse}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = cfg
                    best_mae = mae
                    best_cv_mae = cv_mae
                    # save best model
                    joblib.dump(model, os.path.join(MODELS_DIR, "best_tuned_model.pkl"))
                    mlflow.sklearn.log_model(model, artifact_path="best_tuned_rf")

        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    result = {
        "search_type": "random",
        "n_folds": N_FOLDS,
        "total_trials": N_ITER,
        "best_params": best_params,
        "best_mae": best_mae,
        "best_cv_mae": best_cv_mae,
        "parent_run_name": PARENT_RUN_NAME,
    }

    out_path = os.path.join(RESULTS_DIR, "step2_s2.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()