"""Register V2 in the MLflow registry as a STAGING candidate (never straight to Production).

08_automate_lifecycle.py then decides promotion with the same rule as every other candidate.
"""
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from common import get_production_version, load_config, write_json
from mlflow_utils import configure_mlflow, log_sklearn_model

warnings.filterwarnings("ignore", message=".*transition_model_version_stage.*")
warnings.filterwarnings("ignore", message=".*Model registry stages will be removed.*")


def main():
    config = load_config()
    configure_mlflow(config)
    client = MlflowClient()
    name = config["registry"]["model_name"]
    retrain = json.loads(Path("outputs/retraining/retraining_report.json").read_text(encoding="utf-8"))
    comparison = json.loads(Path("outputs/retraining/model_comparison_report.json").read_text(encoding="utf-8"))
    prod_before = get_production_version(client, name)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="Lab12_Registry_V2") as run:
        mlflow.log_params({"candidate": "V2", "source_retraining_run_id": retrain["mlflow_run_id"],
                           "training_dataset": retrain["training_dataset"], "training_rows": retrain["training_rows"],
                           "previous_production_version": prod_before.version if prod_before else "none"})
        mlflow.log_metrics(retrain["metrics"])  # rmse/mae/mse/r2_score on the fixed hold-out (lifecycle metric)
        mlflow.log_metrics({f"evolved_{k}": v for k, v in retrain["evolved_metrics"].items()})
        log_sklearn_model(joblib.load("models/retrained_model_v2.pkl"), "retrained_gradient_boosting_v2", registered_model_name=name)
        # Same artifact path/name as every other model so 11_prepare_deployment can package V2 if promoted.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "preprocessor.pkl"
            shutil.copy("models/retrained_preprocessor_v2.pkl", target)
            mlflow.log_artifact(str(target), artifact_path="preprocessing_pipeline")
        run_id = run.info.run_id

    version = next(v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id)
    for k, v in {"candidate": "V2", "source_retraining_run_id": retrain["mlflow_run_id"],
                 "quality_gate": "PASSED" if comparison["v2_quality_gate_passed"] else "FAILED",
                 "better_on_holdout": str(comparison["v2_better_on_holdout"]),
                 "better_on_evolved": str(comparison["v2_better_on_evolved"])}.items():
        client.set_model_version_tag(name, version.version, k, v)
    client.transition_model_version_stage(name, version.version, "Staging", archive_existing_versions=False)

    prod_after = get_production_version(client, name)
    unchanged = (prod_before.version if prod_before else None) == (prod_after.version if prod_after else None)
    write_json("outputs/retraining/mlflow_registry_update_report.json", {
        "model_name": name, "v2_version": int(version.version), "v2_stage": "Staging", "v2_run_id": run_id,
        "production_before": prod_before.version if prod_before else None,
        "production_after": prod_after.version if prod_after else None, "production_unchanged": unchanged})
    print(f"[SUCCESS] V2 registered as version {version.version} in Staging. Production unchanged: {unchanged}")
    print("[INFO] Run src/mlops_pipeline/08_automate_lifecycle.py to let the lifecycle decide promotion.")


if __name__ == "__main__":
    main()
