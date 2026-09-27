import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load the raw data
    data_path = "data/raw/day.csv"
    df = pd.read_csv(data_path)
    df["dteday"] = pd.to_datetime(df["dteday"])

    # 2. Define target and drop leakage / non-feature columns
    target = "cnt"
    columns_to_drop = ["cnt", "casual", "registered", "instant", "dteday"]
    X = df.drop(columns=columns_to_drop)
    y = df[target]

    # 3. Chronological train/test split (time-series data, no shuffling)
    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # 4. Feature groups
    categorical_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
    numerical_features = ["temp", "atemp", "hum", "windspeed"]

    # 5. Build modular scikit-learn pipelines
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ])

    # 6. Execute pipeline
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 7. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.float64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.float64))

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    metadata = {
        "dataset_name": "Bike Sharing Demand",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
