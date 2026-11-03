"""Usage: python inspect_mlflow_run.py <run_id>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mlflow.tracking import MlflowClient

from mlflow_utils import configure_mlflow


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    configure_mlflow()
    client = MlflowClient()
    run = client.get_run(sys.argv[1])
    print("Run name   :", run.info.run_name)
    print("Status     :", run.info.status)
    print("Params     :", run.data.params)
    print("Metrics    :", run.data.metrics)
    print("Artifacts  :")
    for item in client.list_artifacts(run.info.run_id):
        print("  -", item.path)


if __name__ == "__main__":
    main()
