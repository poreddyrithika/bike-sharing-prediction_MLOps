# MLflow tracking
The URI is read from `MLFLOW_TRACKING_URI`, else `mlflow.tracking_uri` in `config.yaml`, else MLflow's local default
(`sqlite:///mlflow.db` in the project root; the pipeline scripts always switch to the project root first).
Shared server: `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` and
`$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"`. A GitHub-hosted runner cannot reach a server on your laptop.
