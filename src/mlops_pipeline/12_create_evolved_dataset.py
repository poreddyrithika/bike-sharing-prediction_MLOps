"""Simulate DATA drift: warmer, more humid, windier, rainier days (features only)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config

SOURCE = Path("data/processed/X_test_raw.csv")
OUTPUT = Path("data/monitoring/drift/evolved_data.csv")


def main():
    config = load_config()
    rng = np.random.default_rng(config["monitoring"]["random_seed"])
    df = pd.read_csv(SOURCE)
    df["temp"] = (df["temp"] + rng.normal(0.12, 0.05, len(df))).clip(0.05, 0.98)
    df["atemp"] = (df["temp"] * 0.95 + rng.normal(0, 0.02, len(df))).clip(0.05, 0.98)
    df["hum"] = (df["hum"] + rng.normal(0.10, 0.05, len(df))).clip(0.1, 0.99)
    df["windspeed"] = (df["windspeed"] * rng.normal(1.35, 0.10, len(df))).clip(0.02, 0.6)
    df["weathersit"] = rng.choice([1, 2, 3], size=len(df), p=[0.35, 0.40, 0.25])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"[SUCCESS] Evolved dataset created: {OUTPUT} {df.shape}")
    print("Modified: temp, atemp, hum, windspeed, weathersit (calendar columns unchanged)")


if __name__ == "__main__":
    main()
