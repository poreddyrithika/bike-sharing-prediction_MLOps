"""Stage 9 - Production inference with the Production model AND its own preprocessor.

The preprocessor is downloaded from the Production model's MLflow run, not read from
models/preprocessor.pkl, because the local file belongs to the most recently trained
candidate, which may not be the model that is currently in Production.
"""
import os
import sys
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from common import clean_dataframe, get_production_version, load_config
from mlflow_utils import configure_mlflow


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")
    config = load_config()
    configure_mlflow(config)

    model_name = config["registry"]["model_name"]
    raw_data_path = config["data"]["raw_path"]
    client = MlflowClient()

    try:
        production = get_production_version(client, model_name)
        if production is None:
            raise RuntimeError("no model is in the Production stage")
        print(f"[INFO] Loading Production model: {model_name} v{production.version}")
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{production.version}")
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor_file = client.download_artifacts(production.run_id, "preprocessing_pipeline/preprocessor.pkl", dst_path=tmp)
            preprocessor = joblib.load(preprocessor_file)
    except Exception as exc:
        print(f"[ERROR] Could not load the Production model/preprocessor from MLflow: {exc}")
        print("[ERROR] Run 08_automate_lifecycle.py first.")
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    df = pd.read_csv(raw_data_path)
    sample = df.sample(n=5, random_state=7).sort_values(config["data"]["id_column"])
    predictions = np.clip(model.predict(preprocessor.transform(clean_dataframe(sample, config))), 0, None)

    result = sample[[config["data"]["date_column"]] + list(config["features"]["numerical"]) + list(config["features"]["categorical"])].copy()
    result["actual_cnt"] = sample[config["data"]["target_column"]].to_numpy()
    result["predicted_cnt"] = np.round(predictions, 1)
    result["abs_error"] = (result["actual_cnt"] - result["predicted_cnt"]).abs().round(1)
    result["model_version"] = int(production.version)

    print("\n--- ML Pipeline Production Inference Results (demonstration sample) ---")
    print(result[[config["data"]["date_column"], "actual_cnt", "predicted_cnt", "abs_error"]].to_string(index=False))
    print("------------------------------------------------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    result.to_csv(config["paths"]["inference_output"], index=False)
    print(f"[SUCCESS] Predictions saved to {config['paths']['inference_output']}")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_inference_demo() else 1)
