"""Helpers shared by the Lab 7-13 pipeline stages.

Importing this module (or calling ``load_config``) switches the working
directory to the project root, so every relative path in ``config.yaml`` and the
default local MLflow store (``mlflow.db``) resolve identically no matter where
the script was launched from (terminal, IDE run button, CI, subprocess).
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

# Make src/ importable (mlflow_utils, validate_data) from any stage script.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_config():
    os.chdir(PROJECT_ROOT)
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=str)


def feature_columns(config):
    return list(config["features"]["numerical"]) + list(config["features"]["categorical"])


def clean_dataframe(df, config):
    """Keep exactly the model features, in a fixed order and dtype.

    Drops the id, the date, the target and the leakage columns
    (casual / registered) if they are present.
    """
    numerical = list(config["features"]["numerical"])
    categorical = list(config["features"]["categorical"])
    missing = [c for c in numerical + categorical if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    cleaned = df[numerical + categorical].copy()
    for column in numerical:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype(float)
    for column in categorical:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype("Int64")
    return cleaned.reset_index(drop=True)


def build_preprocessor(config):
    """Single canonical preprocessing pipeline (numeric: median+scale, categorical: one-hot)."""
    numerical = list(config["features"]["numerical"])
    categorical = list(config["features"]["categorical"])
    categories = [list(config["features"]["categories"][c]) for c in categorical]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical),
        ("cat", cat_pipeline, categorical),
    ])


def expected_transformed_width(config):
    return len(config["features"]["numerical"]) + sum(
        len(config["features"]["categories"][c]) for c in config["features"]["categorical"]
    )


def regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred)),
    }


def get_production_version(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        return None
    if len(production) > 1:
        raise RuntimeError(f"Multiple Production versions found for '{model_name}'.")
    return production[0]
