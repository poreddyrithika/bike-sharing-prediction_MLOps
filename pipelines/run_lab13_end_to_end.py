"""Lab 13: monitoring trigger -> (drift + new batch) -> full pipeline -> promotion-gated deployment."""
import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, "src/mlops_pipeline/12_monitor_trigger.py"]).returncode)
