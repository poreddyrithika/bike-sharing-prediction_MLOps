# Lab 9 - Failure handling and recovery
`python src\mlops_pipeline\10_failure_injection.py` (repetitions: `lab9.repetitions` = 3) writes `reports/failure_report.json`.
| Scenario | Expected |
|---|---|
| Missing raw dataset | HALT |
| Invalid datatype (`temp="INVALID_NUMBER"`) | HALT |
| Schema mismatch (column `weathersit` removed) | HALT |
| Out-of-range value (`hum=5.0`) | HALT |
| Corrupted processed output (NaN in `X_train_final.npy`) | HALT |
| Missing local model | detect failure, restore the Production model from MLflow, evaluation succeeds |
Observed in the build environment: all 6 scenarios x 3 repetitions PASSED. Every test restores the files it touched.
Also see `src/simulate_data_corruption.py` (7 injected faults, all detected).
