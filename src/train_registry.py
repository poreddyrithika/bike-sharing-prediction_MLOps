import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow_utils import configure_mlflow, log_sklearn_model


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    # 1. Set up MLflow
    configure_mlflow()
    mlflow.set_experiment("Bike_Sharing_Demand")

    # 2. Load Processed Data
    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        return

    # 3. Define Model Parameters
    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="GradientBoosting_Registry_V1") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "GradientBoosting")

        # 4. Train Model
        print("[INFO] Training GradientBoosting model...")
        gb_model = GradientBoostingRegressor(**params)
        gb_model.fit(X_train, y_train)

        # 5. Evaluate
        print("[INFO] Evaluating model...")
        y_pred = gb_model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)

        # 6. Log preprocessor pipeline (lineage)
        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")

        # 7. Log and register the model
        print("[INFO] Pushing model to MLflow Registry...")
        log_sklearn_model(
            gb_model,
            name="gradient_boosting_model",
            registered_model_name="Bike_Sharing_Production_Model",
        )

        print(f"[SUCCESS] Metrics logged: RMSE = {metrics['rmse']:.4f} | R2 = {metrics['r2_score']:.4f}")
        print("[SUCCESS] Model successfully registered under name: 'Bike_Sharing_Production_Model'")


if __name__ == "__main__":
    train_and_register_model()
