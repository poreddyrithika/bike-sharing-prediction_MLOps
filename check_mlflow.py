"""Print the tracking URI, every experiment and the number of runs in each."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_utils import configure_mlflow


def main():
    print("Tracking URI:", configure_mlflow())
    client = MlflowClient()
    for exp in client.search_experiments():
        runs = client.search_runs([exp.experiment_id])
        print(f"- {exp.name} (id={exp.experiment_id}) -> {len(runs)} run(s)")


if __name__ == "__main__":
    main()
