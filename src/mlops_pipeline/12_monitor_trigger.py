"""Monitoring + retraining trigger (Lab 13).

Refreshes the drift reports, decides whether drift exists, and only when drift AND an
incoming batch are present calls the main pipeline. Writes outputs/monitoring/monitor_trigger_report.json.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT, load_config, write_json

REFRESH = ["12_data_drift.py", "12_prediction_drift.py", "12_concept_drift.py", "12_drift_analysis.py"]
DRIFT_REPORT = Path("outputs/drift/drift_analysis_report.csv")
PIPELINE = "src/mlops_pipeline/bike_demand_pipeline.py"
TRIGGER_REPORT = Path("outputs/monitoring/monitor_trigger_report.json")
DATA_LOAD_REPORT = Path("artifacts/data_load_report.json")


def run(script):
    return subprocess.run([sys.executable, script], cwd=PROJECT_ROOT).returncode == 0


def save(status, **fields):
    base = {"status": status, "drift_detected": False, "new_data_present": False, "pipeline_called": False,
            "new_data_ingested": False, "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()}
    base.update(fields)
    write_json(TRIGGER_REPORT, base)


def main():
    config = load_config()
    new_data = Path(config["data"]["new_batch_path"])
    print("=" * 70 + "\nMONITORING + RETRAINING TRIGGER\n" + "=" * 70)

    for script in REFRESH:
        print(f"[INFO] Running {script}...")
        if not run(f"src/mlops_pipeline/{script}"):
            save("drift_analysis_failed", new_data_present=new_data.exists())
            print(f"[ERROR] {script} failed.")
            return False

    report = pd.read_csv(DRIFT_REPORT)
    drift = bool((report["status"].astype(str).str.upper() == "DRIFT").any())
    present = new_data.exists()
    print(f"[INFO] Drift detected: {drift} | incoming batch present: {present}")

    if not drift:
        save("no_drift", new_data_present=present)
        print("[INFO] No drift detected. Waiting for future monitoring cycles.")
        return True
    if not present:
        save("drift_waiting_for_data", drift_detected=True)
        print(f"[INFO] Drift detected, but {new_data} is missing. Waiting for data.")
        return True

    print("[INFO] Drift detected and a batch is available. Triggering the MLOps pipeline...")
    result = subprocess.run([sys.executable, PIPELINE], cwd=PROJECT_ROOT)
    load_status = None
    if DATA_LOAD_REPORT.exists():
        load_status = json.loads(DATA_LOAD_REPORT.read_text(encoding="utf-8")).get("status")
    ingested = load_status == "ingested"
    retrained = ingested and result.returncode == 0
    save("retraining_completed" if retrained else "no_new_data_retraining_skipped",
         drift_detected=True, new_data_present=True, pipeline_called=True,
         pipeline_exit_code=result.returncode, pipeline_status="passed" if result.returncode == 0 else "failed",
         data_load_status=load_status, new_data_ingested=ingested, retraining_executed=retrained)
    if result.returncode != 0:
        print("[ERROR] The MLOps pipeline failed.")
        return False
    print("[SUCCESS] New data ingested and retraining completed." if ingested
          else "[INFO] Pipeline finished but no new data was ingested; retraining skipped.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
