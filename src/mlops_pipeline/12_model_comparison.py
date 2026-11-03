"""Compare Production V1 and retrained V2 on (a) the fixed hold-out and (b) the evolved environment."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.model_loader import ModelLoader  # noqa: E402
from common import clean_dataframe, load_config, regression_metrics, write_json  # noqa: E402

OUT_JSON = Path("outputs/retraining/model_comparison_report.json")
OUT_CSV = Path("outputs/retraining/model_comparison_report.csv")


def evaluate(model, pre, X, y, config):
    return regression_metrics(y, np.clip(model.predict(pre.transform(clean_dataframe(X, config))), 0, None))


def main():
    config = load_config()
    X_test, y_test = pd.read_csv("data/processed/X_test_raw.csv"), np.load("data/processed/y_test.npy")
    evo = pd.read_csv("data/monitoring/drift/evolved_labeled_data.csv")
    X_evo, y_evo = evo.drop(columns=["cnt"]), evo["cnt"].to_numpy(dtype=float)

    v1 = ModelLoader()
    v1.load()
    v2_model, v2_pre = joblib.load("models/retrained_model_v2.pkl"), joblib.load("models/retrained_preprocessor_v2.pkl")

    res = {
        "model_v1": {"version": v1.model_version, "holdout": evaluate(v1.model, v1.preprocessor, X_test, y_test, config),
                     "evolved": evaluate(v1.model, v1.preprocessor, X_evo, y_evo, config)},
        "model_v2": {"holdout": evaluate(v2_model, v2_pre, X_test, y_test, config),
                     "evolved": evaluate(v2_model, v2_pre, X_evo, y_evo, config)},
    }
    rows = []
    for dataset in ("holdout", "evolved"):
        for metric in ("rmse", "mae", "r2_score"):
            a, b = res["model_v1"][dataset][metric], res["model_v2"][dataset][metric]
            rows.append({"dataset": dataset, "metric": metric, "model_v1": a, "model_v2": b, "change_v2_minus_v1": b - a})
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    gate = config["quality_gate"]
    res["v2_quality_gate_passed"] = res["model_v2"]["holdout"][gate["metric"]] >= gate["minimum_value"]
    res["v2_better_on_holdout"] = res["model_v2"]["holdout"]["rmse"] < res["model_v1"]["holdout"]["rmse"]
    res["v2_better_on_evolved"] = res["model_v2"]["evolved"]["rmse"] < res["model_v1"]["evolved"]["rmse"]
    write_json(OUT_JSON, res)
    print("=" * 70 + "\nV1 VS V2 COMPARISON\n" + "=" * 70)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nV2 quality gate: {'PASS' if res['v2_quality_gate_passed'] else 'FAIL'} | "
          f"better on hold-out: {res['v2_better_on_holdout']} | better on evolved data: {res['v2_better_on_evolved']}")


if __name__ == "__main__":
    main()
