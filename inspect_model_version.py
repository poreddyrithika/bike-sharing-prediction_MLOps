"""Usage: python inspect_model_version.py <version> [model_name]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import yaml
from mlflow.tracking import MlflowClient

from mlflow_utils import configure_mlflow


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    configure_mlflow(config)
    name = sys.argv[2] if len(sys.argv) > 2 else config["registry"]["model_name"]
    mv = MlflowClient().get_model_version(name, sys.argv[1])
    print("Name    :", mv.name)
    print("Version :", mv.version)
    print("Stage   :", mv.current_stage)
    print("Run ID  :", mv.run_id)
    print("Source  :", mv.source)
    print("Tags    :", mv.tags)


if __name__ == "__main__":
    main()
