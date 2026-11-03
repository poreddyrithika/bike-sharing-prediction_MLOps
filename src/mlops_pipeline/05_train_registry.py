"""Stage 5 - Train the model, track the run in MLflow and register it as a new model version."""
import json
import os
import sys

import joblib
import mlflow
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from common import load_config, regression_metrics
from mlflow_utils import configure_mlflow, log_sklearn_model


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    config = load_config()

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = config["data"]["processed_dir"]
    try:
        X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Run 03_preprocessing.py first.")
        return False

    metadata_path = os.path.join(processed_dir, "dataset_metadata.json")
    dataset_meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            dataset_meta = json.load(f)

    params = dict(config["model"])

    with mlflow.start_run(run_name="GradientBoosting") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "GradientBoostingRegressor")
        mlflow.log_param("pipeline", "ML Pipeline (Lab 7)")
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("test_rows", len(y_test))
        if dataset_meta:
            mlflow.log_param("dataset_rows", dataset_meta.get("raw_dataset_rows"))
            mlflow.log_param("dataset_sha256", dataset_meta.get("raw_dataset_sha256"))
            mlflow.log_param("train_period", " .. ".join(dataset_meta.get("train_period", [])))
        if os.getenv("GITHUB_RUN_ID"):
            mlflow.log_param("github_run_id", os.getenv("GITHUB_RUN_ID"))
        if os.getenv("GITHUB_SHA"):
            mlflow.log_param("github_commit", os.getenv("GITHUB_SHA"))

        print("[INFO] Training GradientBoostingRegressor...")
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating on the fixed hold-out set...")
        metrics = regression_metrics(y_test, model.predict(X_test))
        mlflow.log_metrics(metrics)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, config["paths"]["model"])

        preprocessor_path = config["paths"]["preprocessor"]
        if os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        log_sklearn_model(model, name="gradient_boosting_model", registered_model_name=config["registry"]["model_name"])

        print(f"[SUCCESS] RMSE={metrics['rmse']:.2f} | MAE={metrics['mae']:.2f} | R2={metrics['r2_score']:.4f}")
        print(f"[SUCCESS] Model saved to {config['paths']['model']}")
        print(f"[SUCCESS] Model registered as '{config['registry']['model_name']}'")
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")
    return True


if __name__ == "__main__":
    sys.exit(0 if train_and_register_model() else 1)
