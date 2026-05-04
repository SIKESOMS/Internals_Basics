"""
Task 3 — CLI Predictor (used inside Docker container)
Usage:
  python predict_cli.py --stroke_rate 46.7 --drag_coefficient 0.1 \
                        --turn_time_ms 583.3 --pool_length_m 37
"""

import argparse
import json
import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_model():
    # prefer tuned model, fall back to base RandomForest
    for fname in ("best_tuned_model.pkl", "RandomForest.pkl"):
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("No trained model found in models/. Run train.py first.")

def parse_args():
    parser = argparse.ArgumentParser(description="SwimSync lap-time predictor")
    parser.add_argument("--stroke_rate",      type=float, required=True)
    parser.add_argument("--drag_coefficient", type=float, required=True)
    parser.add_argument("--turn_time_ms",     type=float, required=True)
    parser.add_argument("--pool_length_m",    type=float, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    model = load_model()

    import pandas as pd
    X = pd.DataFrame([[args.stroke_rate, args.drag_coefficient,
                       args.turn_time_ms, args.pool_length_m]],
                     columns=["stroke_rate", "drag_coefficient",
                              "turn_time_ms", "pool_length_m"])
    prediction = round(float(model.predict(X)[0]), 4)

    result = {
        "image_name": "swimsync-predictor",
        "image_tag": "v1",
        "base_image": "python:3.10-slim",
        "test_input": {
            "stroke_rate": args.stroke_rate,
            "drag_coefficient": args.drag_coefficient,
            "turn_time_ms": args.turn_time_ms,
            "pool_length_m": args.pool_length_m,
        },
        "prediction": prediction,
    }

    # Save to results/ if the directory exists (won't exist inside Docker by default)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "step3_s3.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()