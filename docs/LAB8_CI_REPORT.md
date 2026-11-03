# Lab 8 - Pipeline automation and CI
`.github/workflows/lab8_ci.yml` installs `requirements.txt`, runs the pipeline with `--force` (hosted runners have no cluster,
so `SKIP_DEPLOYMENT=1`), runs `pytest`, and uploads artifacts. Quality gate (`config.yaml`): `r2_score >= 0.60`; below it the
pipeline exits before lifecycle promotion. Repeated-run demo: change `model.n_estimators` / `max_depth` in `config.yaml`,
push, and compare the new runs in MLflow experiment `Bike_Sharing_Demand_Pipeline`. For shared history across CI runs set the
`MLFLOW_TRACKING_URI` secret (see `MLFLOW_CI_SETUP.md`). GitHub Actions vs Jenkins: Actions is YAML-native to the repo with
hosted runners; Jenkins needs self-managed infrastructure. Not executed in the build sandbox.
