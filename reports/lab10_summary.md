# Lab 10 - Model serving with FastAPI
Endpoints: `GET /`, `GET /health` (503 if model missing), `GET /metadata`, `POST /predict`, `POST /predict/batch` (1-1000 days).
Input (`api/schemas.py`): season 1-4, yr 0-2, mnth 1-12, holiday 0/1, weekday 0-6, workingday 0/1, weathersit 1-4, temp/atemp/hum/windspeed in [0,1],
optional `dteday`; unknown fields (e.g. `casual`) and `holiday=1 & workingday=1` are rejected. Output: `{"predicted_rentals": float}` (never negative).
Errors share one shape `{status, message, details[{field,message}]}` for 422/503/500/404. Every request is logged to `logs/inference/inference_log.jsonl`.
Tests: `python -m pytest tests -q` (build environment: 41 passed, 1 skipped = container test).
