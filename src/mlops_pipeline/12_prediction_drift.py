"""Prediction drift: compare the distribution of Production-model predictions (baseline vs evolved)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.model_loader import ModelLoader  # noqa: E402
from api.preprocessing import clean_prediction_data  # noqa: E402
from common import load_config  # noqa: E402

BASELINE = Path("data/processed/X_test_raw.csv")
EVOLVED = Path("data/monitoring/drift/evolved_data.csv")
REPORT = Path("outputs/drift/prediction_drift_report.csv")


def predict(loader, data):
    return np.clip(loader.model.predict(loader.preprocessor.transform(clean_prediction_data(data))), 0, None)


def main():
    config = load_config()
    threshold = config["monitoring"]["ks_threshold"]
    loader = ModelLoader()
    loader.load()
    base, evo = predict(loader, pd.read_csv(BASELINE)), predict(loader, pd.read_csv(EVOLVED))
    stat, p = ks_2samp(base, evo)
    row = {
        "prediction_distribution_ks": float(stat), "prediction_distribution_p_value": float(p),
        "prediction_distribution_status": "DRIFT" if stat >= threshold else "OK",
        "baseline_mean_prediction": float(base.mean()), "evolved_mean_prediction": float(evo.mean()),
        "mean_shift_pct": float((evo.mean() - base.mean()) / base.mean() * 100),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(REPORT, index=False)
    print("=" * 70 + "\nPREDICTION DRIFT ANALYSIS\n" + "=" * 70)
    print(f"Model v{loader.model_version} | mean prediction {base.mean():.1f} -> {evo.mean():.1f} ({row['mean_shift_pct']:+.1f}%)")
    print(f"KS={stat:.6f} (threshold {threshold}) -> {row['prediction_distribution_status']} | report: {REPORT}")


if __name__ == "__main__":
    main()
