# Lab 13 - End-to-end integration
Flow: monitor trigger -> drift analysis -> (drift + `data/raw/new_bike.csv`) -> pipeline -> quality gate -> lifecycle -> promotion-gated deployment.

Evidence from the build environment (simulated data, MLflow 3.17):
* Drift on the simulated evolved data: 5 drifted features (temp, atemp, hum, windspeed, weathersit), prediction KS 0.306, concept RMSE +39%, R2 0.67 -> 0.28.
* Batch of 90 simulated days ingested: 731 -> 821 rows, previous data archived as `day_v01.csv`, lineage recorded.
* Candidate v3: RMSE 1089.10 (R2 0.6625) passed the gate but did not beat Production v1 (1076.10) -> archived, deployment skipped.
* Re-running the trigger skipped the same batch (`no_new_data_retraining_skipped`).
* Promotion path: a better candidate (v4, RMSE 1039.52) became Production and produced a valid bundle; `deploy_production.py --dry-run` printed the build/rollout commands.
Not verified: Docker build, Kubernetes rollout, GitHub Actions (no such services in the build sandbox).
Limitations: simulated batch and drift, interactive runner must stay online, MLflow stages are deprecated, fixed hold-out cannot reflect a new regime.
