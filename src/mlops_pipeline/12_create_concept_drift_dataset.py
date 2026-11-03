"""Simulate CONCEPT drift: same kind of inputs, but demand responds differently.

Working-day demand falls (e.g. remote work), weekend/holiday demand rises, and very
hot days suppress rentals - the feature -> target relationship changes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config

FEATURES = Path("data/monitoring/drift/evolved_data.csv")
LABELS = Path("data/processed/y_test.npy")
OUTPUT = Path("data/monitoring/drift/evolved_labeled_data.csv")


def apply_concept_shift(X, y, rng):
    factor = np.where(X["workingday"] == 1, 0.78, 1.12)
    factor = factor * np.where(X["temp"] > 0.75, 0.90, 1.0)
    return np.clip(np.round(y * factor * rng.normal(1.0, 0.05, len(y))), 0, None).astype(int)


def main():
    config = load_config()
    rng = np.random.default_rng(config["monitoring"]["random_seed"])
    X = pd.read_csv(FEATURES)
    y = np.load(LABELS)
    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch: X={len(X)}, y={len(y)}")
    out = X.copy()
    out["cnt"] = apply_concept_shift(X, y, rng)
    out.to_csv(OUTPUT, index=False)
    print(f"[SUCCESS] Concept-drift dataset created: {OUTPUT} {out.shape}")
    print(f"Mean demand baseline={y.mean():.1f} -> evolved={out['cnt'].mean():.1f}")


if __name__ == "__main__":
    main()
