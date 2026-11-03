"""Stage 2 - Raw data validation (exit code 1 halts the pipeline)."""
import os
import sys

import pandas as pd

from common import load_config
from validation import validate_schema


def main():
    config = load_config()
    data_path = config["data"]["raw_path"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
