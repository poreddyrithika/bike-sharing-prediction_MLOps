"""Failure handling and recovery tests: break the pipeline on purpose and verify it fails safely."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, load_config
from mlflow_utils import configure_mlflow


def run_script(script):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT).returncode


def restore(backup, original):
    if os.path.exists(original):
        os.remove(original)
    os.replace(backup, original)


def _data_scenario(config, name, mutate):
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup = Path(str(data_path) + ".lab9_backup")
    shutil.copy2(data_path, backup)
    try:
        df = pd.read_csv(data_path)
        df = mutate(df)
        df.to_csv(data_path, index=False)
        halted = run_script("src/mlops_pipeline/02_validate_data.py") != 0
        return {"scenario": name, "expected": "HALT", "actual": "HALT" if halted else "CONTINUED", "passed": halted}
    finally:
        restore(backup, data_path)


def test_missing_data(config):
    print("\n[LAB 9] Scenario 1: Missing data")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup = Path(str(data_path) + ".lab9_backup")
    os.replace(data_path, backup)
    try:
        halted = run_script("src/mlops_pipeline/02_validate_data.py") != 0
        return {"scenario": "missing_data", "expected": "HALT", "actual": "HALT" if halted else "CONTINUED", "passed": halted}
    finally:
        restore(backup, data_path)


def test_invalid_datatype(config):
    print("\n[LAB 9] Scenario 2: Invalid datatype")
    def mutate(df):
        df["temp"] = df["temp"].astype(object)
        df.loc[0, "temp"] = "INVALID_NUMBER"
        return df
    return _data_scenario(config, "invalid_datatype", mutate)


def test_schema_mismatch(config):
    print("\n[LAB 9] Scenario 3: Schema mismatch")
    return _data_scenario(config, "schema_mismatch", lambda df: df.drop(columns=["weathersit"]))


def test_out_of_range(config):
    print("\n[LAB 9] Scenario 4: Out-of-range value")
    def mutate(df):
        df.loc[0, "hum"] = 5.0
        return df
    return _data_scenario(config, "out_of_range_value", mutate)


def test_preprocessing_inconsistency(config):
    print("\n[LAB 9] Scenario 5: Preprocessing inconsistency")
    path = PROJECT_ROOT / config["data"]["processed_dir"] / "X_train_final.npy"
    if not path.exists():
        print("[ERROR] Processed training output is missing. Run the normal pipeline first.")
        return {"scenario": "preprocessing_inconsistency", "expected": "HALT", "actual": "NOT_TESTED", "passed": False}
    backup = Path(str(path) + ".lab9_backup")
    shutil.copy2(path, backup)
    try:
        X = np.load(path)
        X[0, 0] = np.nan
        np.save(path, X)
        halted = run_script("src/mlops_pipeline/04_validate_outputs.py") != 0
        return {"scenario": "preprocessing_inconsistency", "expected": "HALT", "actual": "HALT" if halted else "CONTINUED", "passed": halted}
    finally:
        restore(backup, path)


def test_missing_model_with_recovery(config):
    print("\n[LAB 9] Scenario 6: Missing model artifact + recovery")
    model_path = PROJECT_ROOT / config["paths"]["model"]
    if not model_path.exists():
        print("[ERROR] Local model is missing before the test. Run the normal pipeline first.")
        return {"scenario": "missing_model_artifact", "expected": "RECOVER", "actual": "NOT_TESTED", "passed": False}
    backup = Path(str(model_path) + ".lab9_backup")
    shutil.copy2(model_path, backup)
    os.remove(model_path)
    try:
        detected = run_script("src/mlops_pipeline/06_model_evaluation.py") != 0
        configure_mlflow(config)
        model = mlflow.sklearn.load_model(f"models:/{config['registry']['model_name']}/Production")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        recovered = run_script("src/mlops_pipeline/06_model_evaluation.py") == 0
        return {"scenario": "missing_model_artifact", "expected": "RECOVER",
                "actual": "RECOVERED" if detected and recovered else "RECOVERY_FAILED",
                "failure_detected": detected, "recovery_succeeded": recovered, "passed": detected and recovered}
    finally:
        restore(backup, model_path)


def run_failure_tests():
    print("[INFO] Failure Tests: Pipeline Failure Handling and Recovery")
    config = load_config()
    repetitions = int(config["lab9"]["repetitions"])
    tests = [test_missing_data, test_invalid_datatype, test_schema_mismatch, test_out_of_range,
             test_preprocessing_inconsistency, test_missing_model_with_recovery]
    runs = []
    for rep in range(1, repetitions + 1):
        print(f"\n{'=' * 70}\n[INFO] Repetition {rep}/{repetitions}\n{'=' * 70}")
        results = [t(config) for t in tests]
        runs.append({"repetition": rep, "results": results, "passed": all(r["passed"] for r in results)})

    report = {"stage": "Failure Handling and Recovery", "repetitions": repetitions, "runs": runs,
              "overall_status": "PASSED" if all(r["passed"] for r in runs) else "FAILED"}
    report_path = PROJECT_ROOT / config["paths"]["failure_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\n[{'SUCCESS' if report['overall_status'] == 'PASSED' else 'ERROR'}] Failure tests {report['overall_status']}. Report: {report_path}")
    return report["overall_status"] == "PASSED"


if __name__ == "__main__":
    sys.exit(0 if run_failure_tests() else 1)
