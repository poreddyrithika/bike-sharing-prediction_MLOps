import pandas as pd

NUMERICAL = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
FEATURES = NUMERICAL + CATEGORICAL


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly the model features (drops dteday / instant / target / leakage columns if sent)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    cleaned = df[FEATURES].copy()
    for col in NUMERICAL:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype(float)
    for col in CATEGORICAL:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype("Int64")
    return cleaned.reset_index(drop=True)
