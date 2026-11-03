import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Test traffic must not pollute the real monitoring log.
os.environ.setdefault("INFERENCE_LOG_PATH", str(ROOT / "outputs" / "test_inference_log.jsonl"))


def pytest_collection_modifyitems(config, items):
    if (ROOT / "deployment" / "model.pkl").exists():
        return
    skip = pytest.mark.skip(reason="No deployment bundle. Run the pipeline (with --force) first.")
    for item in items:
        if "test_pipeline_units" not in item.nodeid:
            item.add_marker(skip)
