"""Build the deployment bundle (model + matching preprocessor + metadata) from the Production model.

Only the generated bundle files are replaced. The directory also holds
deploy_production.py, which must survive.
"""
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy
import sklearn
from mlflow.tracking import MlflowClient

from common import PROJECT_ROOT, clean_dataframe, get_production_version, load_config
from mlflow_utils import configure_mlflow

BUNDLE_FILES = ["model.pkl", "preprocessor.pkl", "model_metadata.json", "requirements-model.txt"]


def prepare_deployment():
    config = load_config()
    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    deployment_dir = PROJECT_ROOT / "deployment"
    deployment_dir.mkdir(exist_ok=True)

    client = MlflowClient()
    production = get_production_version(client, model_name)
    if production is None:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    version, run_id = production.version, production.run_id
    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")

    print("[INFO] Downloading Production model...")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")

    print("[INFO] Downloading matching preprocessor...")
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = client.download_artifacts(run_id, "preprocessing_pipeline/preprocessor.pkl", dst_path=tmp)
        preprocessor = joblib.load(downloaded)

    # Smoke test: the pair must transform + predict real rows before it is packaged.
    x_test_path = Path(config["data"]["processed_dir"]) / "X_test_raw.csv"
    if x_test_path.exists():
        sample = clean_dataframe(pd.read_csv(x_test_path).head(10), config)
        predictions = model.predict(preprocessor.transform(sample))
        if not np.isfinite(predictions).all():
            raise RuntimeError("Smoke test failed: non-finite predictions from the model/preprocessor pair.")
        print(f"[SUCCESS] Smoke test passed on {len(sample)} rows.")
    else:
        print("[WARN] X_test_raw.csv not found - smoke test skipped.")

    for name in BUNDLE_FILES:
        target = deployment_dir / name
        if target.exists():
            target.unlink()

    joblib.dump(model, deployment_dir / "model.pkl")
    joblib.dump(preprocessor, deployment_dir / "preprocessor.pkl")

    run_metrics = client.get_run(run_id).data.metrics
    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": production.current_stage,
        "run_id": run_id,
        "model_source": production.source,
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "metrics": {k: round(float(v), 4) for k, v in run_metrics.items()},
        "source": "MLflow",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    with open(deployment_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(deployment_dir / "requirements-model.txt", "w", encoding="utf-8") as f:
        f.write("# Written by 11_prepare_deployment.py: exact versions the model was trained with.\n")
        f.write(f"scikit-learn=={sklearn.__version__}\nnumpy=={np.__version__}\npandas=={pd.__version__}\njoblib=={joblib.__version__}\n")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    for name in ("model.pkl", "preprocessor.pkl", "model_metadata.json", "requirements-model.txt"):
        print(f"  - {name}")
    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    try:
        prepare_deployment()
    except Exception as exc:
        print(f"[ERROR] Deployment preparation failed: {exc}")
        sys.exit(1)
