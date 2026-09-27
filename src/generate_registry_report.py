import os
import json
from mlflow.tracking import MlflowClient


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    client = MlflowClient()
    model_name = "Bike_Sharing_Production_Model"

    try:
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)

        if not champion:
            print("[ERROR] No model found in Production stage!")
            return

        run = client.get_run(champion.run_id)

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source,
            },
            "performance_metrics": {
                "rmse": run.data.metrics.get("rmse"),
                "mae": run.data.metrics.get("mae"),
                "mse": run.data.metrics.get("mse"),
                "r2_score": run.data.metrics.get("r2_score"),
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl",
        }

        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")


if __name__ == "__main__":
    generate_registry_report()
