"""Data drift: KS test (numerical) and Jensen-Shannon divergence (categorical) vs the baseline."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common import load_config

BASELINE = Path("data/processed/X_test_raw.csv")
EVOLVED = Path("data/monitoring/drift/evolved_data.csv")
REPORT = Path("outputs/drift/data_drift_report.csv")


def js_divergence(baseline, evolved):
    cats = sorted(set(baseline.dropna().unique()) | set(evolved.dropna().unique()))
    p = baseline.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(dtype=float)
    q = evolved.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(dtype=float)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, m) + kl(q, m)) / 2  # 0 (identical) .. 1 (disjoint)


def main():
    config = load_config()
    mon = config["monitoring"]
    baseline, evolved = pd.read_csv(BASELINE), pd.read_csv(EVOLVED)
    rows = []
    for col in mon["numerical_columns"]:
        stat, p = ks_2samp(baseline[col].dropna(), evolved[col].dropna())
        rows.append({"feature": col, "type": "numerical", "metric": "KS", "score": round(float(stat), 6),
                     "p_value": round(float(p), 6), "threshold": mon["ks_threshold"],
                     "status": "DRIFT" if stat >= mon["ks_threshold"] else "OK"})
    for col in mon["categorical_columns"]:
        score = js_divergence(baseline[col], evolved[col])
        rows.append({"feature": col, "type": "categorical", "metric": "JS", "score": round(score, 6),
                     "p_value": None, "threshold": mon["js_threshold"],
                     "status": "DRIFT" if score >= mon["js_threshold"] else "OK"})
    report = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    print("=" * 70 + "\nDATA DRIFT ANALYSIS\n" + "=" * 70)
    print(report.to_string(index=False))
    print(f"\nFeatures analysed: {len(report)} | drifted: {(report['status'] == 'DRIFT').sum()} | report: {REPORT}")


if __name__ == "__main__":
    main()
