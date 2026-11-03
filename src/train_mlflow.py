import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow_utils import configure_mlflow, log_sklearn_model


def train_and_track(run_name="GradientBoosting_Baseline", params=None):
    if params is None:
        params = {"random_state": 42}

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # 1. Load Data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")

    configure_mlflow()
    mlflow.set_experiment("Bike_Sharing_Demand")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", "GradientBoosting")

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, predictions),
            "mse": mean_squared_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "r2_score": r2_score(y_test, predictions),
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: RMSE = {metrics['rmse']:.4f} | R2 = {metrics['r2_score']:.4f}")

        # Diagnostic plot: actual vs predicted
        os.makedirs("artifacts", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, predictions, label="Predictions")
        lo, hi = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="Perfect Prediction")
        ax.set_xlabel("Actual Bike Rentals")
        ax.set_ylabel("Predicted Bike Rentals")
        ax.set_title(f"Actual vs Predicted - {run_name}")
        ax.legend()
        plot_path = "artifacts/actual_vs_predicted.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        log_sklearn_model(model, name="model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/gradient_boosting_model.pkl")
        print(f"Run '{run_name}' successfully tracked!")


if __name__ == "__main__":
    train_and_track(run_name="GradientBoosting")
