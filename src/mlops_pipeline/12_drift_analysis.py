"""Combine the data, prediction and concept drift reports into one decision table."""
from pathlib import Path

import pandas as pd

DATA = Path("outputs/drift/data_drift_report.csv")
PRED = Path("outputs/drift/prediction_drift_report.csv")
CONCEPT = Path("outputs/drift/concept_drift_report.csv")
OUTPUT = Path("outputs/drift/drift_analysis_report.csv")


def main():
    from common import load_config
    load_config()
    data, pred, concept = pd.read_csv(DATA), pd.read_csv(PRED).iloc[0], pd.read_csv(CONCEPT)
    drifted = data[data["status"] == "DRIFT"]
    degraded = concept[concept["status"] == "DRIFT"]
    summary = pd.DataFrame([
        {"category": "Data Drift", "metric": "Features with drift", "value": len(drifted),
         "status": "DRIFT" if len(drifted) else "OK", "details": ", ".join(drifted["feature"])},
        {"category": "Prediction Drift", "metric": "Prediction distribution", "value": pred["prediction_distribution_ks"],
         "status": pred["prediction_distribution_status"], "details": f"KS; mean shift {pred['mean_shift_pct']:+.1f}%"},
        {"category": "Concept Drift", "metric": "Degraded performance metrics", "value": len(degraded),
         "status": "DRIFT" if len(degraded) else "OK", "details": ", ".join(degraded["metric"])},
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print("=" * 70 + "\nCOMBINED DRIFT ANALYSIS\n" + "=" * 70)
    print(summary.to_string(index=False))
    overall = "DRIFT / MODEL BEHAVIOR CHANGE OBSERVED" if (summary["status"] == "DRIFT").any() else "NO SIGNIFICANT DRIFT OBSERVED"
    print(f"\nOVERALL STATUS : {overall}\nCombined report saved : {OUTPUT}")


if __name__ == "__main__":
    main()
