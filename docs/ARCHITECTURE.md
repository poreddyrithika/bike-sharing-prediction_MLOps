# Architecture - Bike Sharing Demand MLOps

```text
GitHub Actions (manual / daily)  ->  self-hosted runner  ->  12_monitor_trigger.py
   refresh: 12_data_drift -> 12_prediction_drift -> 12_concept_drift -> 12_drift_analysis
   drift?  no -> report "no_drift"        drift + no batch -> "drift_waiting_for_data"
   drift + data/raw/new_bike.csv -> bike_demand_pipeline.py
        01 load/ingest -> 02 validate -> 03 preprocess -> 04 validate outputs -> 05 train+register
        -> 06 evaluate -> 07 quality gate (R2 >= 0.60) -> 08 lifecycle (RMSE must improve)
        -> deploy_if_promoted (bundle -> docker build -> kubectl rollout) -> 09 inference
```

```text
bike-sharing/
  .github/workflows/  lab8_ci, lab9_failure_tests, lab13_monitoring, lab13_runner_test
  api/                FastAPI app (main, schemas, model_loader, preprocessing, monitoring_logger)
  artifacts/ logs/ outputs/ reports/ models/    generated runtime outputs
  data/raw            day.csv, new_bike.csv.staged (simulated batch), archive/ (versions + lineage)
  data/processed      npy/csv splits, test_instants.csv (fixed hold-out), dataset_metadata.json
  data/monitoring/drift   simulated drift datasets
  deployment/         deploy_production.py + generated model bundle
  docs/ k8s/ lab11_commands/ notebooks/ pipelines/ tests/
  src/                Lab 3-6 scripts, mlflow_utils.py, mlops_pipeline/ (Labs 7-13)
  config.yaml  Dockerfile  requirements*.txt
```

* **Lifecycle:** `None -> Staging -> Production | Archived`. Production changes only if the candidate's RMSE is lower
  (by more than `registry.min_improvement`). All RMSEs are measured on the same fixed hold-out.
* **Serving:** the API loads `deployment/` (model + matching preprocessor + metadata) at start-up. `/health` returns 503
  until both are loaded, so Kubernetes readiness probes work.
* **Lineage:** every training run logs dataset row count and SHA-256; ingested batches are recorded in
  `data/raw/archive/dataset_lineage.json` with the archived previous version.
