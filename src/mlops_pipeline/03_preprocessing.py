"""Stage 3 - Preprocessing with a permanent chronological hold-out.

The test set is the last 20% of the ORIGINAL dataset (instant 585..731). Its ids
are written once to data/processed/test_instants.csv and reused forever, so
ingested batches only ever extend the training set and model versions stay
comparable.
"""
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd

from common import (build_preprocessor, clean_dataframe, feature_columns, file_sha256,
                    load_config)


def fixed_test_ids(df, config, processed_dir):
    id_col = config["data"]["id_column"]
    ids_path = os.path.join(processed_dir, "test_instants.csv")

    if os.path.exists(ids_path):
        print(f"[INFO] Using fixed test set from {ids_path}")
        return set(pd.read_csv(ids_path)[id_col].astype(int))

    baseline = df.sort_values(id_col).head(int(config["split"]["baseline_rows"]))
    split_index = int(math.floor(len(baseline) * (1 - float(config["split"]["test_fraction"])) + 1e-9))
    test_ids = baseline[id_col].iloc[split_index:].astype(int).tolist()

    pd.DataFrame({id_col: test_ids}).to_csv(ids_path, index=False)
    print(f"[SUCCESS] Fixed test set created: {len(test_ids)} days ({test_ids[0]}..{test_ids[-1]}) -> {ids_path}")
    return set(test_ids)


def split_raw_data(df, config, processed_dir):
    id_col = config["data"]["id_column"]
    target = config["data"]["target_column"]

    df = df.sort_values(id_col).reset_index(drop=True)
    test_ids = fixed_test_ids(df, config, processed_dir)

    mask = df[id_col].isin(test_ids)
    if mask.sum() != len(test_ids):
        raise ValueError(
            f"Only {int(mask.sum())} of {len(test_ids)} fixed test days exist in the dataset. "
            "The raw data no longer contains the original hold-out period."
        )

    train_df, test_df = df.loc[~mask], df.loc[mask]
    if train_df.empty or train_df[id_col].min() >= test_df[id_col].min():
        raise ValueError("Training data must start before the hold-out period - unexpected dataset layout.")

    X_train, X_test = clean_dataframe(train_df, config), clean_dataframe(test_df, config)
    y_train = train_df[target].to_numpy(dtype=np.float64)
    y_test = test_df[target].to_numpy(dtype=np.float64)

    print(f"[INFO] Training days : {len(X_train)} ({train_df[config['data']['date_column']].iloc[0]} .. {train_df[config['data']['date_column']].iloc[-1]})")
    print(f"[INFO] Hold-out days : {len(X_test)} ({test_df[config['data']['date_column']].iloc[0]} .. {test_df[config['data']['date_column']].iloc[-1]})")
    return X_train, X_test, y_train, y_test, train_df, test_df


def run_preprocessing():
    print("[INFO] Starting Preprocessing...")
    config = load_config()

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test, train_df, test_df = split_raw_data(raw_df, config, processed_dir)

    preprocessor = build_preprocessor(config)
    print("[INFO] Fitting preprocessing pipeline on training data only...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    joblib.dump(preprocessor, config["paths"]["preprocessor"])

    date_col = config["data"]["date_column"]
    metadata = {
        "dataset_name": "Bike Sharing Demand (daily)",
        "target_column": config["data"]["target_column"],
        "raw_dataset_rows": int(len(raw_df)),
        "raw_dataset_sha256": file_sha256(data_path),
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "train_period": [str(train_df[date_col].iloc[0]), str(train_df[date_col].iloc[-1])],
        "test_period": [str(test_df[date_col].iloc[0]), str(test_df[date_col].iloc[-1])],
        "numerical_features": config["features"]["numerical"],
        "categorical_features": config["features"]["categorical"],
        "split_strategy": "chronological fixed hold-out",
        "fixed_test_set": True,
        "test_instants_path": os.path.join(processed_dir, "test_instants.csv"),
    }
    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {config['paths']['preprocessor']}")
    print(f"[INFO] Transformed feature width: {X_train_final.shape[1]} (features used: {len(feature_columns(config))} raw)")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_preprocessing() else 1)
