"""Performance of the deployed model on the fixed hold-out set."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.model_loader import ModelLoader  # noqa: E402
from api.preprocessing import clean_prediction_data  # noqa: E402
from common import load_config, regression_metrics  # noqa: E402


def main():
    config = load_config()
    X, y = pd.read_csv("data/processed/X_test_raw.csv"), np.load("data/processed/y_test.npy")
    loader = ModelLoader()
    loader.load()
    pred = np.clip(loader.model.predict(loader.preprocessor.transform(clean_prediction_data(X))), 0, None)
    m = regression_metrics(y, pred)
    print("=" * 60 + "\nMODEL PERFORMANCE MONITORING\n" + "=" * 60)
    print(f"Model {loader.model_name} v{loader.model_version} | test days: {len(X)}")
    for k, v in m.items():
        print(f"{k:<10}: {v:.4f}")
    gate = config["quality_gate"]
    print(f"\nQuality gate ({gate['metric']} >= {gate['minimum_value']}): {m[gate['metric']] >= gate['minimum_value']}")


if __name__ == "__main__":
    main()
