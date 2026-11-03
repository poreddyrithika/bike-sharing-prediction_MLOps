"""Stage 1 - Data loading, batch ingestion, hashing and lineage.

* If data/raw/new_bike.csv does not exist  -> status ``no_new_data``.
* If it exists and its SHA-256 is already in the lineage file -> ``already_ingested``.
* Otherwise the batch is schema-validated, identity-checked (unique ids/dates, strictly
  after the current data), the previous dataset is archived, rows are appended and the
  lineage is updated -> ``ingested``.
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import file_sha256, load_config, write_json
from validation import validate_schema


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []
    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_lineage(lineage_path, lineage):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)


def next_archive_version(archive_dir, stem):
    versions = []
    for file_path in archive_dir.glob(f"{stem}_v*.csv"):
        try:
            versions.append(int(file_path.stem.replace(f"{stem}_v", "")))
        except ValueError:
            continue
    return max(versions, default=0) + 1


def validate_new_batch_identity(new_df, current_df, config):
    id_col = config["data"]["id_column"]
    date_col = config["data"]["date_column"]

    if new_df.empty:
        raise ValueError("The incoming batch is empty.")
    if current_df.empty:
        raise ValueError("The current dataset is empty.")

    for name, frame in (("incoming batch", new_df), ("current dataset", current_df)):
        if not frame[id_col].is_unique:
            raise ValueError(f"The {name} contains duplicate {id_col} values.")
        if not frame[date_col].is_unique:
            raise ValueError(f"The {name} contains duplicate {date_col} values.")

    id_overlap = set(current_df[id_col]).intersection(new_df[id_col])
    if id_overlap:
        raise ValueError(f"The incoming batch has {len(id_overlap)} {id_col} values already present in the dataset.")

    date_overlap = set(current_df[date_col]).intersection(new_df[date_col])
    if date_overlap:
        raise ValueError(f"The incoming batch has {len(date_overlap)} dates already present in the dataset.")

    if new_df[id_col].min() <= current_df[id_col].max():
        raise ValueError(f"Batch {id_col} values must be greater than the current maximum ({current_df[id_col].max()}).")

    if pd.to_datetime(new_df[date_col]).min() <= pd.to_datetime(current_df[date_col]).max():
        raise ValueError("Batch dates must be later than the last date already in the dataset.")

    print("[SUCCESS] New batch identity validation passed.")


def ingest_new_data(raw_path, new_data_path, archive_dir, lineage_path, report_path, config):
    current_df = pd.read_csv(raw_path)
    new_df = pd.read_csv(new_data_path)

    print(f"[INFO] Current dataset rows: {len(current_df)}")
    print(f"[INFO] Incoming batch rows : {len(new_df)}")

    print("[INFO] Validating incoming batch schema...")
    if not validate_schema(new_df, output_report_name="new_data_validation.csv"):
        raise ValueError("Incoming batch failed schema validation. Ingestion stopped.")

    new_data_hash = file_sha256(new_data_path)
    print(f"[INFO] Batch SHA256: {new_data_hash}")

    lineage = load_lineage(lineage_path)
    if any(record.get("new_data_hash") == new_data_hash for record in lineage):
        print("[INFO] This batch has already been ingested. No archive or retraining action will be performed.")
        write_json(report_path, {
            "status": "already_ingested",
            "ingestion_performed": False,
            "new_data_hash": new_data_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_rows": len(current_df),
            "new_rows": len(new_df),
        })
        return False

    validate_new_batch_identity(new_df, current_df, config)

    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = raw_path.stem
    archive_version = next_archive_version(archive_dir, stem)
    archive_path = archive_dir / f"{stem}_v{archive_version:02d}.csv"
    shutil.copy2(raw_path, archive_path)
    print(f"[SUCCESS] Previous dataset archived as: {archive_path.name}")

    updated_df = pd.concat([current_df, new_df], ignore_index=True)
    updated_df = updated_df.sort_values(config["data"]["id_column"]).reset_index(drop=True)

    temp_path = raw_path.with_suffix(".tmp.csv")
    updated_df.to_csv(temp_path, index=False)
    os.replace(temp_path, raw_path)

    print(f"[SUCCESS] Batch appended to {raw_path}")
    print(f"[INFO] Previous rows: {len(current_df)} | Added rows: {len(new_df)} | Current rows: {len(updated_df)}")

    record = {
        "dataset_version": f"v{archive_version + 1:02d}",
        "archive_file": str(archive_path).replace("\\", "/"),
        "source_file": str(new_data_path).replace("\\", "/"),
        "new_data_hash": new_data_hash,
        "previous_rows": len(current_df),
        "new_rows": len(new_df),
        "current_rows": len(updated_df),
        "first_new_date": str(new_df[config["data"]["date_column"]].min()),
        "last_new_date": str(new_df[config["data"]["date_column"]].max()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    lineage.append(record)
    save_lineage(lineage_path, lineage)

    write_json(report_path, {"status": "ingested", "ingestion_performed": True, **record})
    print("[SUCCESS] Dataset ingestion completed.")
    return True


def main():
    config = load_config()

    raw_path = Path(config["data"]["raw_path"])
    new_data_path = Path(config["data"]["new_batch_path"])
    archive_dir = raw_path.parent / "archive"
    lineage_path = archive_dir / "dataset_lineage.json"
    report_path = Path("artifacts/data_load_report.json")

    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    if not raw_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {raw_path}")

    if new_data_path.exists():
        print(f"[INFO] Incoming batch detected: {new_data_path}")
        ingest_new_data(raw_path, new_data_path, archive_dir, lineage_path, report_path, config)
    else:
        print("[INFO] No incoming batch detected. Continuing with the current dataset.")
        current_df = pd.read_csv(raw_path)
        write_json(report_path, {
            "status": "no_new_data",
            "ingestion_performed": False,
            "current_rows": len(current_df),
            "current_columns": len(current_df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    final_df = pd.read_csv(raw_path)
    if final_df.empty:
        raise ValueError(f"Current dataset is empty: {raw_path}")

    print(f"[SUCCESS] Final dataset shape: {final_df.shape}")
    print("[SUCCESS] Data loading stage completed.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Data loading failed: {exc}")
        sys.exit(1)
