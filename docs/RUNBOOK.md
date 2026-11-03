# Runbook - Bike Sharing Demand MLOps

## Before running
1. Activate the venv, work from the project root.
2. For deployment steps start Docker Desktop with Kubernetes and check `docker info`, `kubectl get nodes`.
3. For GitHub automation start the runner (`C:\actions-runner\run.cmd`) and set the repo variable `BIKE_PROJECT_DIR`.

## Monitoring report (`outputs/monitoring/monitor_trigger_report.json`)
| status | meaning |
|---|---|
| `no_drift` | no drift; pipeline not called |
| `drift_waiting_for_data` | drift found, `data/raw/new_bike.csv` missing |
| `no_new_data_retraining_skipped` | batch already ingested (hash in lineage) |
| `retraining_completed` | batch ingested and pipeline succeeded; read `reports/lifecycle_report.json` for promotion |
| `drift_analysis_failed` | an upstream drift script failed; see the log |

## Key files
`artifacts/data_load_report.json`, `data/raw/archive/dataset_lineage.json`, `reports/evaluation_report.json`,
`reports/quality_gate_report.json`, `reports/lifecycle_report.json`, `reports/deployment_automation_report.json`,
`outputs/drift/drift_analysis_report.csv`, `logs/bike_demand_pipeline_run_log.json`.

## Inspect MLflow
`python src\mlops_pipeline\check_mlflow_registry.py`, `python check_mlflow.py`,
`python inspect_mlflow_run.py <run_id>`, `python inspect_model_version.py <version>`, or `python -m mlflow ui`.

## Repeat the Lab 13 test
Batches are identified by SHA-256, so a used batch is skipped. For a new test: regenerate with a different seed
(`monitoring.random_seed`) or delete `data/raw/archive/`, restore `data/raw/day.csv` from git, and delete
`data/processed/test_instants.csv` only if you also want a new hold-out (not recommended).
