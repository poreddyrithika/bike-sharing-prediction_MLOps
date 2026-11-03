# Bike Sharing Demand Prediction - End-to-End MLOps Platform

Predicts **daily bike rentals (`cnt`)** from calendar and weather data (UCI Bike Sharing, `day.csv`, 731 days, 2011-2012)
and wraps the model in an MLOps workflow: validation, reproducible preprocessing, MLflow tracking and registry, a quality
gate, champion/challenger lifecycle, dataset versioning, FastAPI serving, Docker/Kubernetes deployment, drift monitoring,
retraining and GitHub Actions automation. This is a **regression** project (the churn reference is classification), so
metrics are RMSE / MAE / R2 and "better" means a **lower RMSE**.

Labs 1-2 are in `notebooks/project_implementation.ipynb`; Labs 3-6 are the scripts in `src/` + `pipelines/`; Labs 7-13 are
`src/mlops_pipeline/`, `api/`, `deployment/`, `k8s/`, `tests/`, `.github/workflows/`.

## Quick start (Windows PowerShell, run from this folder)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Labs 3-6
python pipelines\run_lab3_baseline.py
python pipelines\run_lab4_tracking.py
python pipelines\run_lab5_pipeline.py
python pipelines\run_lab6_registry.py

# Lab 7-9: full MLOps pipeline (first run needs --force: there is no new batch yet)
python src\mlops_pipeline\bike_demand_pipeline.py --force
```

Set `$env:SKIP_DEPLOYMENT="1"` if Docker/Kubernetes are not running; the pipeline then builds the deployment bundle but
skips `docker`/`kubectl`. Without it, a promoted model triggers a real deployment and the stage fails if Docker is down.

| Lab | Run |
|---|---|
| 7 pipeline | `python src\mlops_pipeline\bike_demand_pipeline.py --force` |
| 8 CI / quality gate | `.github/workflows/lab8_ci.yml`; change `config.yaml -> quality_gate` to see the gate fail |
| 9 failure handling | `python src\mlops_pipeline\10_failure_injection.py` |
| 10 API | `python -m uvicorn api.main:app --reload` then open http://127.0.0.1:8000/docs ; `python -m pytest tests -q` |
| 11 Docker / K8s | see `docs/lab11_containerization_report.md` and `lab11_commands/` |
| 12 drift + retraining | `python pipelines\run_lab12_monitoring.py` then `python pipelines\run_lab12_retraining.py` |
| 13 end to end | copy `data\raw\new_bike.csv.staged` to `data\raw\new_bike.csv`, then `python src\mlops_pipeline\12_monitor_trigger.py` |

## Pipeline stages (`src/mlops_pipeline/`)

| # | Script | Responsibility |
|---|---|---|
| 1 | `01_load_data.py` | Load data; ingest `data/raw/new_bike.csv` if present and not already ingested (SHA-256 + lineage) |
| 2 | `02_validate_data.py` | Pandera schema, ranges, uniqueness, `casual + registered == cnt` |
| 3 | `03_preprocessing.py` | Fixed chronological hold-out, median-impute + scale, fixed-category one-hot |
| 4 | `04_validate_outputs.py` | NaN/finite/shape/width/target checks |
| 5 | `05_train_registry.py` | GradientBoostingRegressor, MLflow run, registry version |
| 6 | `06_model_evaluation.py` | RMSE/MAE/MSE/R2, residual summary, plots |
| 7 | `07_quality_gate.py` | `r2_score >= 0.60` (config) else the pipeline stops |
| 8 | `08_automate_lifecycle.py` | None -> Staging -> Production/Archived; candidate must beat Production RMSE |
| 9 | `deploy_if_promoted.py` | Bundle + Docker + Kubernetes only if promoted |
| 10 | `09_inference.py` | Predict with the Production model **and its own preprocessor** |

Without `--force` the pipeline stops after stage 1 when nothing new was ingested (`no_new_data` / `already_ingested`).
That is intentional (no retraining on unchanged data), not a failure.

## Design decisions you should know

* **Fixed hold-out.** Test set = last 20% of the original 731 days (instant 585-731, 2012-08-07..2012-12-31), saved in
  `data/processed/test_instants.csv`. Ingested data only extends the training set so every model version is compared on the
  same days. Trade-off: the hold-out cannot show how a model handles a *new regime*; in production you would refresh it
  with recent labelled data.
* **Leakage.** `casual` and `registered` sum to `cnt` and are never features (also rejected by the API).
* **Stable feature space.** One-hot categories are fixed in `config.yaml` (including `yr=2` for 2013), so all model
  versions share the same input width.
* **Serving bundle = model + its own preprocessor**, built from the Production run, never from "whatever was trained last".
* **MLflow skops fix.** New MLflow refuses to save tree models unless `sklearn.tree._tree.Tree` is trusted;
  `src/mlflow_utils.py` handles this (this was breaking Lab 4 and Lab 6 on current MLflow).
* **Model stages** (`None/Staging/Production/Archived`) are deprecated in MLflow but used by the course; the code silences
  that warning. A future migration would use model aliases.

## Simulated data (read this)

The public dataset ends on 2012-12-31, so **everything about 2013 and drift is simulated**:
`data/raw/new_bike.csv.staged` (90 days, Jan-Mar 2013, `create_new_batch.py`, seeded) and the files in
`data/monitoring/drift/` (seeded perturbations of the hold-out: warmer/more humid/windier/rainier days; working-day demand
-22%, weekend +12%, very hot days -10%). They are never presented as real observations.

## Verified in the build environment

Python 3.12, MLflow 3.17, scikit-learn 1.8, pandas 3.0: Labs 3-6 pipelines, the Lab 7 pipeline (first run: RMSE 1076.10,
MAE 913.52, R2 0.6705, v1 promoted), 41 pytest tests, 3x6 failure scenarios, drift analysis, V2 retraining
(correctly archived: better on drifted data, worse on the hold-out), new-batch ingestion (731 -> 821 rows, archive +
lineage), duplicate-batch skip, and a promoted candidate (v4, RMSE 1039.52) producing a valid deployment bundle plus a
`deploy_production.py --dry-run`.
**Not run here (no Docker/Kubernetes/GitHub in the sandbox):** the Docker build, the `kubectl` rollout and the GitHub
Actions workflows. They follow the churn reference but you must run them once yourself.

## Repository layout

See the folder tree in `docs/ARCHITECTURE.md`. Runtime folders (`artifacts/`, `logs/`, `outputs/`, `models/*.pkl`,
`mlflow.db`, `mlruns/`) are generated and git-ignored.

## Troubleshooting

* `untrusted types` MLflow error -> use the scripts in this repo (they call `log_sklearn_model`).
* `No Production version` -> run the pipeline with `--force`.
* Pipeline stops after "Data loading" -> expected without a new batch; add `--force`.
* `docker`/`kubectl` not found in the deployment stage -> start Docker Desktop (Kubernetes enabled) or set `SKIP_DEPLOYMENT=1`.
* scikit-learn pickle warnings in the container -> the image installs `deployment/requirements-model.txt`, which pins the
  training versions; rebuild the image after retraining.
* API tests skipped -> no `deployment/model.pkl`; run the pipeline first.

## Security

Never commit tokens or `mlflow.db`. A self-hosted runner executes workflow code on your machine; use a private repo.
