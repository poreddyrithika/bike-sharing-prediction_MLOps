"""Request/response history from the inference log: recent calls, failures, single-request lookup."""
import json
import sys
from pathlib import Path

import pandas as pd

from common import load_config

LOG = Path("logs/inference/inference_log.jsonl")


def main():
    load_config()
    if not LOG.exists():
        raise FileNotFoundError(f"Log file not found: {LOG}. Call the API first.")
    df = pd.DataFrame([json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()])
    if df.empty:
        raise ValueError("Inference history is empty")
    print("=" * 60 + f"\nREQUEST-RESPONSE HISTORY ({len(df)} records)\n" + "=" * 60)
    print(df[["request_id", "timestamp", "predicted_rentals", "latency_ms", "status", "model_version"]].tail(10).to_string(index=False))
    failures = df[df["status"] == "failure"]
    print(f"\nFailed requests: {len(failures)}")
    if not failures.empty:
        print(failures[["request_id", "timestamp", "error"]].to_string(index=False))
    target = sys.argv[1] if len(sys.argv) > 1 else df["request_id"].iloc[0]
    print(f"\nDetails for request {target}:")
    print(df[df["request_id"] == target].to_string(index=False))


if __name__ == "__main__":
    main()
