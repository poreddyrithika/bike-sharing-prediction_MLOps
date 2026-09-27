import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    data_path = "data/raw/day.csv"
    df = pd.read_csv(data_path)
    print("Shape of dataset:", df.shape)

    df["dteday"] = pd.to_datetime(df["dteday"])

    # 2. Define target and drop leakage / non-feature columns
    target = "cnt"
    columns_to_drop = ["cnt", "casual", "registered", "instant", "dteday"]

    X = df.drop(columns=columns_to_drop)
    y = df[target]

    # 3. Chronological train/test split (this is time-series data)
    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    # 4. Feature groups
    categorical_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
    numerical_features = ["temp", "atemp", "hum", "windspeed"]

    # 5. One-hot encode categoricals (fit on train, align test columns)
    X_train = pd.get_dummies(X_train, columns=categorical_features, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_features, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print("Features after one-hot encoding:", X_train.shape[1])

    # 6. Scale numerical features (fit on train only)
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # 7. Save processed artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")
    joblib.dump(categorical_features, "models/categorical_features.pkl")
    joblib.dump(numerical_features, "models/numerical_features.pkl")

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
