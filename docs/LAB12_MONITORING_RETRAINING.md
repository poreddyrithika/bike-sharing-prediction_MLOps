# Lab 12 - Monitoring, drift and continuous training
`python pipelines\run_lab12_monitoring.py`: creates the simulated drift sets and writes `outputs/drift/*.csv`
(data drift KS/JS, prediction drift KS, concept drift RMSE/MAE/R2 with thresholds from `config.yaml -> monitoring`).
`python pipelines\run_lab12_retraining.py`: post-drift batch -> retrain V2 -> compare V1/V2 on the hold-out **and** the evolved data -> register V2 in Staging -> lifecycle decides.
Observed: V2 hold-out RMSE 1169.58 (worse), evolved-data RMSE 1328.01 vs V1 1495.55 (better). The lifecycle keeps V1 because decisions use the fixed hold-out.
`12_prediction_monitoring.py`, `12_request_history.py`, `12_api_performance_monitoring.py` read `logs/inference/inference_log.jsonl` (call the API first).
