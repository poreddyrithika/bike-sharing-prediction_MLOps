"""Lab 12: create the drift datasets, run every drift/performance report, then the combined analysis."""
import subprocess
import sys

SCRIPTS = [
    "12_create_evolved_dataset.py", "12_create_concept_drift_dataset.py",
    "12_data_drift.py", "12_prediction_drift.py", "12_concept_drift.py", "12_drift_analysis.py",
    "12_performance_monitoring.py", "12_performance_trend.py",
]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n[INFO] ---> {script}")
        if subprocess.run([sys.executable, f"src/mlops_pipeline/{script}"]).returncode != 0:
            print(f"[ERROR] {script} failed.")
            sys.exit(1)
    print("\n[SUCCESS] Lab 12 monitoring reports generated in outputs/drift/")
