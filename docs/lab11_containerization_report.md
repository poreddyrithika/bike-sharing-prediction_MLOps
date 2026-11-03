# Lab 11 - Containerization and Kubernetes (procedure; not executed in the build sandbox)
```powershell
python src\mlops_pipeline\bike_demand_pipeline.py --force        # creates deployment/ bundle (SKIP_DEPLOYMENT=1 if no Docker)
docker build -t bike-sharing-api:1.0 .
docker run -p 8000:8000 bike-sharing-api:1.0                      # http://localhost:8000/docs
python -m pytest tests\test_local_vs_container.py -s              # local vs container predictions must match
kubectl apply -f k8s\deployment.yaml ; kubectl apply -f k8s\service.yaml
kubectl get deployments ; kubectl get pods ; kubectl get service bike-api-service
python deployment\deploy_production.py                            # versioned image bike-sharing-api:model-vN + rollout
```
Image: `python:3.12-slim`, serving libs from `requirements-api.txt`, exact numpy/pandas/scikit-learn/joblib from
`deployment/requirements-model.txt` (written from the training environment). Probes: `/health` (503 until the model loads).
Evidence to capture for your report: `docker images`, container logs, `kubectl get pods`, test_local_vs_container output.
`python deployment\deploy_production.py --dry-run` prints the exact commands without running them.
