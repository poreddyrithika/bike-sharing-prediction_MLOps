"""Performance across consecutive time windows of the hold-out (is accuracy decaying over time?)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.model_loader import ModelLoader  # noqa: E402
from api.preprocessing import clean_prediction_data  # noqa: E402
from common import load_config, regression_metrics  # noqa: E402


def main():
    load_config()
    X, y = pd.read_csv("data/processed/X_test_raw.csv"), np.load("data/processed/y_test.npy")
    loader = ModelLoader()
    loader.load()
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(X)), 3), start=1):
        xw, yw = X.iloc[idx], y[idx]
        pred = np.clip(loader.model.predict(loader.preprocessor.transform(clean_prediction_data(xw))), 0, None)
        rows.append({"window": f"Window {i}", "days": len(xw), **regression_metrics(yw, pred)})
    df = pd.DataFrame(rows)
    print("=" * 70 + "\nMODEL PERFORMANCE TREND ANALYSIS\n" + "=" * 70)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    change = df["rmse"].iloc[-1] - df["rmse"].iloc[0]
    print(f"\nRMSE change Window 1 -> 3: {change:+.2f} ({'errors increased' if change > 0 else 'errors decreased'})")


if __name__ == "__main__":
    main()
