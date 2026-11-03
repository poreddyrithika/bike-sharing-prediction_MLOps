"""API reliability and latency statistics from the inference log."""
import json
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
        raise ValueError("Inference log is empty")
    total, ok = len(df), int((df["status"] == "success").sum())
    lat = df["latency_ms"].dropna()
    print("=" * 60 + "\nAPI PERFORMANCE MONITORING\n" + "=" * 60)
    print(f"Total requests: {total} | success: {ok} ({ok / total * 100:.2f}%) | failed: {total - ok}")
    if not lat.empty:
        print(f"Latency ms  avg={lat.mean():.3f}  median={lat.median():.3f}  p95={lat.quantile(0.95):.3f}  max={lat.max():.3f}")
    ts = pd.to_datetime(df["timestamp"], utc=True)
    span = (ts.max() - ts.min()).total_seconds()
    print(f"Throughput  : {total / span:.3f} req/s" if span > 0 else "Throughput  : n/a (single timestamp)")


if __name__ == "__main__":
    main()
