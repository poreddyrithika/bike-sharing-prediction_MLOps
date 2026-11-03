# Lab 7 - ML Pipeline Engineering
Stages 01-06, 08, 09 are chained by `src/mlops_pipeline/bike_demand_pipeline.py` (Lab 8 adds `07_quality_gate.py`, Lab 13 adds
the deployment stage). Model: `GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)` registered as
`Bike_Sharing_Production_Model`. Metrics: RMSE (decision metric), MAE, MSE, R2. Run: `python src\mlops_pipeline\bike_demand_pipeline.py --force`.
Observed in the build environment (first run): RMSE 1076.10, MAE 913.52, R2 0.6705; version 1 promoted to Production.
Replace with the numbers from your own run when you submit evidence.
