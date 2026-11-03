"""Stage 4 - Validate the processed outputs before any model is trained."""
import json
import os
import sys

import numpy as np
import pandas as pd

from common import expected_transformed_width, feature_columns, load_config, write_json


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")
    config = load_config()
    processed_dir = config["data"]["processed_dir"]

    try:
        X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError as exc:
        print(f"[ERROR] Could not find processed data: {exc}")
        return False

    errors = []

    for name, array in (("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)):
        if np.isnan(array).sum() > 0:
            errors.append(f"NaNs detected in {name} after preprocessing.")
        if not np.isfinite(array).all():
            errors.append(f"Non-finite values detected in {name}.")

    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    expected_width = expected_transformed_width(config)
    if X_train.shape[1] != expected_width:
        errors.append(f"Unexpected transformed width {X_train.shape[1]} (expected {expected_width}).")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")
    if (y_train < 0).any() or (y_test < 0).any():
        errors.append("Negative values detected in target (cnt) - invalid for rental counts.")

    ids_path = os.path.join(processed_dir, "test_instants.csv")
    if not os.path.exists(ids_path):
        errors.append(f"Fixed test-set definition missing: {ids_path}")
    elif len(pd.read_csv(ids_path)) != len(y_test):
        errors.append("Hold-out size no longer matches test_instants.csv.")

    expected_columns = feature_columns(config)
    for required in ("X_train_raw.csv", "X_test_raw.csv"):
        path = os.path.join(processed_dir, required)
        if not os.path.exists(path):
            errors.append(f"Required raw split artifact missing: {path}")
        elif list(pd.read_csv(path, nrows=1).columns) != expected_columns:
            errors.append(f"{path} columns differ from the configured feature list.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "errors": errors,
    }
    write_json("artifacts/preprocessing_summary_report.json", report)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for error in errors:
            print(f"  - {error}")
        return False

    print("[SUCCESS] Output Validation PASSED.")
    print("[INFO] Summary report saved to artifacts/preprocessing_summary_report.json")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_preprocessing_outputs() else 1)
