"""Lab 7-9: run every stage of the MLOps pipeline (forced, so it also works without a new batch)."""
import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, "src/mlops_pipeline/bike_demand_pipeline.py", "--force"]).returncode)
