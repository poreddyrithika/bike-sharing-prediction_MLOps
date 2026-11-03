"""Lab 5 extra: confirm that two identical training runs give identical results."""
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def run_deterministic_test():
    print("Running Reproducibility Validation...")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}

    model_1 = GradientBoostingRegressor(**params).fit(X_train, y_train)
    rmse_1 = float(np.sqrt(mean_squared_error(y_test, model_1.predict(X_test))))

    model_2 = GradientBoostingRegressor(**params).fit(X_train, y_train)
    rmse_2 = float(np.sqrt(mean_squared_error(y_test, model_2.predict(X_test))))

    is_reproducible = rmse_1 == rmse_2

    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_rmse": rmse_1,
        "execution_2_rmse": rmse_2,
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED",
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 RMSE: {rmse_1:.6f}")
    print(f"Execution 2 RMSE: {rmse_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        print("FAILED: Pipeline is non-deterministic.")
    return is_reproducible


if __name__ == "__main__":
    raise SystemExit(0 if run_deterministic_test() else 1)
