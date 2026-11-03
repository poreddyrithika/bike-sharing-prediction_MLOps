"""Concept drift: performance of the Production model on baseline vs labelled evolved data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.model_loader import ModelLoader  # noqa: E402
from api.preprocessing import clean_prediction_data  # noqa: E402
from common import load_config, regression_metrics  # noqa: E402

BASELINE_X = Path("data/processed/X_test_raw.csv")
BASELINE_Y = Path("data/processed/y_test.npy")
EVOLVED = Path("data/monitoring/drift/evolved_labeled_data.csv")
REPORT = Path("outputs/drift/concept_drift_report.csv")


def evaluate(loader, X, y):
    pred = np.clip(loader.model.predict(loader.preprocessor.transform(clean_prediction_data(X))), 0, None)
    return regression_metrics(y, pred)


def main():
    config = load_config()
    mon = config["monitoring"]
    loader = ModelLoader()
    loader.load()
    base_X, base_y = pd.read_csv(BASELINE_X), np.load(BASELINE_Y)
    evo = pd.read_csv(EVOLVED)
    base_m = evaluate(loader, base_X, base_y)
    evo_m = evaluate(loader, evo.drop(columns=["cnt"]), evo["cnt"].to_numpy())

    rows = []
    for metric in ("rmse", "mae", "r2_score"):
        b, e = base_m[metric], evo_m[metric]
        if metric == "r2_score":
            degraded = (b - e) >= mon["concept_r2_drop"]
            direction = "higher_is_better"
        else:
            degraded = (e - b) / b * 100 >= mon["concept_rmse_increase_pct"]
            direction = "lower_is_better"
        rows.append({"metric": metric, "baseline": b, "evolved": e, "change": e - b,
                     "relative_change_pct": (e - b) / abs(b) * 100, "direction": direction,
                     "status": "DRIFT" if degraded else "OK"})
    report = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    print("=" * 70 + f"\nCONCEPT DRIFT ANALYSIS (Production model v{loader.model_version})\n" + "=" * 70)
    print(report.to_string(index=False))
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
