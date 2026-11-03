"""List every registered model version with its stage and decision metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.tracking import MlflowClient

from mlflow_utils import configure_mlflow


def main():
    print("Tracking URI:", configure_mlflow())
    client = MlflowClient()
    models = client.search_registered_models()
    if not models:
        print("No registered models found. Run the training stage first.")
        return
    for rm in models:
        print(f"\nModel: {rm.name}")
        versions = sorted(client.search_model_versions(f"name='{rm.name}'"), key=lambda v: int(v.version))
        for mv in versions:
            metrics = client.get_run(mv.run_id).data.metrics
            print(
                f"  v{mv.version:<3} stage={mv.current_stage:<11} "
                f"rmse={metrics.get('rmse', float('nan')):.2f}  "
                f"mae={metrics.get('mae', float('nan')):.2f}  "
                f"r2={metrics.get('r2_score', float('nan')):.4f}  run_id={mv.run_id}"
            )


if __name__ == "__main__":
    main()
