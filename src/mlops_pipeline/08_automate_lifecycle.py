"""Stage 8 - Automated champion/challenger lifecycle: None -> Staging -> Production / Archived.

The best Staging candidate (lowest RMSE) replaces the Production model only if it
beats it by more than registry.min_improvement. Metrics come from each model's own
MLflow run, all measured on the same fixed hold-out set.
"""
import sys
import warnings

import mlflow
from mlflow.tracking import MlflowClient

from common import get_production_version, load_config, write_json
from mlflow_utils import configure_mlflow

# MLflow marks registry *stages* as deprecated in favour of aliases. The course uses stages.
warnings.filterwarnings("ignore", message=".*transition_model_version_stage.*")
warnings.filterwarnings("ignore", message=".*Model registry stages will be removed.*")


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    config = load_config()
    configure_mlflow(config)
    client = MlflowClient()

    reg = config["registry"]
    model_name = reg["model_name"]
    metric = reg["decision_metric"]
    lower_is_better = reg.get("direction", "lower_is_better") == "lower_is_better"
    min_improvement = float(reg.get("min_improvement", 0.0))
    report_path = config["paths"]["lifecycle_report"]

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric,
        "direction": reg.get("direction", "lower_is_better"),
        "candidate_version": None,
        "candidate_run_id": None,
        "candidate_metric": None,
        "production_version_before": None,
        "production_metric_before": None,
        "promoted": False,
        "decision": None,
        "production_version_after": None,
    }

    def is_better(candidate, incumbent):
        return (incumbent - candidate) > min_improvement if lower_is_better else (candidate - incumbent) > min_improvement

    def metric_of(model_version):
        value = client.get_run(model_version.run_id).data.metrics.get(metric)
        if value is None:
            raise RuntimeError(f"Run {model_version.run_id} (version {model_version.version}) has no '{metric}' metric.")
        return float(value)

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")

        for mv in [v for v in all_versions if v.current_stage == "None"]:
            client.transition_model_version_stage(name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False)
            print(f"[INFO] Version {mv.version} moved to Staging.")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging = [v for v in all_versions if v.current_stage == "Staging"]
        production = get_production_version(client, model_name)

        production_score = None
        if production:
            production_score = metric_of(production)
            report["production_version_before"] = int(production.version)
            report["production_metric_before"] = production_score
            print(f"[INFO] Production: version {production.version} | {metric}={production_score:.4f}")

        if not staging:
            report["decision"] = "NO_CANDIDATE"
            report["production_version_after"] = report["production_version_before"]
            write_json(report_path, report)
            print("[INFO] No models in Staging. Nothing to promote.")
            return True

        best, best_score = None, None
        for mv in staging:
            score = metric_of(mv)
            print(f"[INFO] Staging version {mv.version} | {metric}={score:.4f}")
            if best is None or (score < best_score if lower_is_better else score > best_score):
                best, best_score = mv, score

        report.update(candidate_version=int(best.version), candidate_run_id=best.run_id, candidate_metric=best_score)

        if production is None:
            promote = True
            print("[INFO] No Production model exists. The best Staging model becomes Production.")
        else:
            promote = is_better(best_score, production_score)
            report["improvement"] = round(abs(production_score - best_score), 6)
            print("[SUCCESS] Candidate beats Production." if promote else "[INFO] Candidate does not beat Production. Production unchanged.")

        if promote:
            client.transition_model_version_stage(name=model_name, version=best.version, stage="Production", archive_existing_versions=False)
            if production:
                client.transition_model_version_stage(name=model_name, version=production.version, stage="Archived", archive_existing_versions=False)
            report.update(promoted=True, decision="PROMOTED", production_version_after=int(best.version))
            print(f"[SUCCESS] Version {best.version} is now Production.")
        else:
            client.transition_model_version_stage(name=model_name, version=best.version, stage="Archived", archive_existing_versions=False)
            report.update(promoted=False, decision="ARCHIVED", production_version_after=report["production_version_before"])
            print(f"[INFO] Version {best.version} archived because it did not improve.")

        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Staging":
                client.transition_model_version_stage(name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False)
                print(f"[INFO] Version {mv.version} archived (lost the competition).")

        write_json(report_path, report)
        print("[SUCCESS] Automated Model Lifecycle execution complete!")
        print(f"[INFO] Lifecycle report saved to {report_path}")
        return True

    except Exception as exc:
        report.update(status="failed", decision="ERROR", error=str(exc))
        write_json(report_path, report)
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(0 if automate_model_lifecycle() else 1)
