"""Stage 7 - Quality gate: stop the pipeline before lifecycle promotion if the candidate is too weak."""
import json
import os
import sys

from common import load_config, write_json


def run_quality_gate():
    print("[INFO] Starting Model Quality Gate...")
    config = load_config()

    report_path = config["paths"]["evaluation_report"]
    gate = config["quality_gate"]
    metric_name = gate["metric"]
    minimum_value = float(gate["minimum_value"])
    higher_is_better = gate.get("direction", "higher_is_better") == "higher_is_better"

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    if metric_name not in report["metrics"]:
        print(f"[ERROR] Metric '{metric_name}' not found in the evaluation report.")
        return False

    actual_value = float(report["metrics"][metric_name])
    passed = actual_value >= minimum_value if higher_is_better else actual_value <= minimum_value
    relation = ">=" if higher_is_better else "<="

    print(f"[INFO] Quality Gate: {metric_name}={actual_value:.4f} | required {relation} {minimum_value:.4f}")

    write_json(config["paths"]["quality_gate_report"], {
        "stage": "Model Quality Gate",
        "metric": metric_name,
        "direction": gate.get("direction", "higher_is_better"),
        "threshold": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED",
    })

    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
    else:
        print("[ERROR] Model Quality Gate FAILED. Pipeline will stop before promotion.")
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_quality_gate() else 1)
