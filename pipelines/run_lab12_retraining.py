"""Lab 12: retrain on post-drift data, compare V1 vs V2, register V2 in Staging, let the lifecycle decide."""
import subprocess
import sys

SCRIPTS = ["12_create_updated_dataset.py", "12_retrain_model.py", "12_model_comparison.py",
           "12_update_mlflow_registry.py", "08_automate_lifecycle.py"]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n[INFO] ---> {script}")
        if subprocess.run([sys.executable, f"src/mlops_pipeline/{script}"]).returncode != 0:
            print(f"[ERROR] {script} failed.")
            sys.exit(1)
    print("\n[SUCCESS] Lab 12 retraining flow finished. Check reports/lifecycle_report.json for the decision.")
