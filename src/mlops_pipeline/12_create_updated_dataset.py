"""Build the post-drift training batch and the updated training dataset used for retraining."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config

X_TRAIN, Y_TRAIN, X_TEST = Path("data/processed/X_train_raw.csv"), Path("data/processed/y_train.npy"), Path("data/processed/X_test_raw.csv")
NEW_BATCH = Path("data/monitoring/drift/post_drift_training_data.csv")
UPDATED = Path("data/processed/updated_training_data.csv")
SUMMARY = Path("outputs/retraining/updated_dataset_summary.csv")


def main():
    config = load_config()
    mon = config["monitoring"]
    rng = np.random.default_rng(mon["random_seed"])
    X, y, X_test = pd.read_csv(X_TRAIN), np.load(Y_TRAIN), pd.read_csv(X_TEST)
    idx = rng.choice(len(X), size=mon["post_drift_batch_size"], replace=True)
    new = X.iloc[idx].copy().reset_index(drop=True)
    y_new = y[idx].astype(float)

    # same environment change as the evolved dataset ...
    new["temp"] = (new["temp"] + rng.normal(0.12, 0.05, len(new))).clip(0.05, 0.98)
    new["atemp"] = (new["temp"] * 0.95 + rng.normal(0, 0.02, len(new))).clip(0.05, 0.98)
    new["hum"] = (new["hum"] + rng.normal(0.10, 0.05, len(new))).clip(0.1, 0.99)
    new["windspeed"] = (new["windspeed"] * rng.normal(1.35, 0.10, len(new))).clip(0.02, 0.6)
    # ... and the same changed demand behaviour (working days down, weekends up, very hot days down)
    factor = np.where(new["workingday"] == 1, 0.78, 1.12) * np.where(new["temp"] > 0.75, 0.90, 1.0)
    new["cnt"] = np.clip(np.round(y_new * factor * rng.normal(1.0, 0.05, len(new))), 0, None).astype(int)

    if list(new.drop(columns=["cnt"]).columns) != list(X_test.columns):
        raise ValueError("Feature schema mismatch between new batch and test feature schema.")
    if new.isnull().any().any() or (new["cnt"] < 0).any():
        raise ValueError("Post-drift batch contains nulls or negative targets.")
    overlap = set(pd.util.hash_pandas_object(new.drop(columns=["cnt"]), index=False)) & set(pd.util.hash_pandas_object(X_test, index=False))
    if overlap:
        raise ValueError(f"{len(overlap)} post-drift rows exactly match hold-out rows.")

    orig = X.copy()
    orig["cnt"] = y.astype(int)
    updated = pd.concat([orig, new], ignore_index=True)
    NEW_BATCH.parent.mkdir(parents=True, exist_ok=True)
    UPDATED.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(NEW_BATCH, index=False)
    updated.to_csv(UPDATED, index=False)
    summary = pd.DataFrame([{"dataset": n, "rows": len(d), "mean_cnt": round(float(d["cnt"].mean()), 2)}
                            for n, d in (("original_training", orig), ("post_drift_batch", new), ("updated_training", updated))])
    summary.to_csv(SUMMARY, index=False)
    print("=" * 70 + "\nUPDATED TRAINING DATASET CREATION\n" + "=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSaved: {NEW_BATCH}, {UPDATED}")


if __name__ == "__main__":
    main()
