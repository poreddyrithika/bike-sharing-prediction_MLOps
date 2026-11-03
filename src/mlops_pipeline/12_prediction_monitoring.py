"""Distribution of predictions the API has served (reads logs/inference/inference_log.jsonl)."""
import json
from pathlib import Path

import pandas as pd

from common import load_config

LOG = Path("logs/inference/inference_log.jsonl")


def load_logs():
    if not LOG.exists():
        raise FileNotFoundError(f"Log file not found: {LOG}. Call the API first.")
    records = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Inference log is empty")
    return pd.DataFrame(records)


def main():
    load_config()
    df = load_logs()
    ok = df[df["status"] == "success"]
    print("=" * 60 + "\nPREDICTION DISTRIBUTION MONITORING\n" + "=" * 60)
    print(f"Total logged requests: {len(df)} | successful: {len(ok)}")
    print(ok["predicted_rentals"].describe().to_string())
    print(f"\nModel: {df['model_name'].iloc[-1]} v{df['model_version'].iloc[-1]} (run {df['run_id'].iloc[-1]})")


if __name__ == "__main__":
    main()
