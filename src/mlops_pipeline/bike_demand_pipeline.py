"""Orchestrator for the Bike Sharing Demand MLOps pipeline (Labs 7-9, 13).

Stages run as separate Python processes with fail-fast behaviour.

    python src/mlops_pipeline/bike_demand_pipeline.py --force   # run every stage
    python src/mlops_pipeline/bike_demand_pipeline.py           # run only if a NEW batch was ingested

Without --force the pipeline stops after stage 1 when no new data was ingested
(``no_new_data`` / ``already_ingested``). That is expected, not a failure: it stops
monitoring-triggered runs from retraining on unchanged data.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("Data Loading", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow + Model Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Deployment Update", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference", "src/mlops_pipeline/09_inference.py"),
]


def load_data_load_report():
    report_path = PROJECT_ROOT / "artifacts" / "data_load_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Data load report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(force=False):
    os.chdir(PROJECT_ROOT)
    print("[INFO] ==================================================")
    print("[INFO] Starting Bike Sharing Demand MLOps Pipeline")
    print("[INFO] ==================================================")

    run_log = {
        "pipeline": "Bike Sharing Demand MLOps",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "force": force,
        "stages": [],
    }
    overall = None

    for index, (stage_name, script_path) in enumerate(STAGES):
        print("\n" + "=" * 70)
        print(f"RUNNING: {stage_name}")
        print("=" * 70)

        started = time.perf_counter()
        result = subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT)
        status = "PASSED" if result.returncode == 0 else "FAILED"
        run_log["stages"].append({
            "stage": stage_name,
            "status": status,
            "exit_code": result.returncode,
            "seconds": round(time.perf_counter() - started, 2),
        })
        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print("[HALT] Pipeline stopped because this stage failed.")
            overall = "FAILED"
            break

        if index == 0 and not force:
            data_status = load_data_load_report().get("status")
            if data_status in {"already_ingested", "no_new_data"}:
                print(f"[INFO] Data load status: {data_status}. No new dataset was ingested.")
                print("[INFO] Retraining and downstream stages will not run (use --force to run them).")
                overall = "NO_NEW_DATA"
                break
            if data_status != "ingested":
                print(f"[ERROR] Unexpected data load status: {data_status}")
                overall = "FAILED"
                break
            print("[INFO] New dataset was successfully ingested. Continuing with the ML pipeline.")

    if overall is None:
        all_passed = len(run_log["stages"]) == len(STAGES) and all(s["status"] == "PASSED" for s in run_log["stages"])
        overall = "PASSED" if all_passed else "FAILED"

    run_log["overall_status"] = overall
    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()

    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    log_path = PROJECT_ROOT / "logs" / "bike_demand_pipeline_run_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    if overall == "PASSED":
        print("[SUCCESS] Bike Sharing Demand MLOps Pipeline completed successfully.")
    elif overall == "NO_NEW_DATA":
        print("[INFO] No new dataset available. Pipeline finished without retraining.")
    else:
        print("[ERROR] Bike Sharing Demand MLOps Pipeline failed.")
    print(f"[INFO] Run log: {log_path}")
    print("=" * 70)

    return overall in {"PASSED", "NO_NEW_DATA"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bike Sharing Demand MLOps pipeline.")
    parser.add_argument("--force", action="store_true", help="Run all ML stages even when no new dataset was ingested.")
    args = parser.parse_args()
    sys.exit(0 if run_pipeline(force=args.force) else 1)
